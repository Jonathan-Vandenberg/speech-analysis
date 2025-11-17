import hashlib
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .database import db_manager
from .middleware import APIKeyInfo, api_key_bearer
from .utils_text import (
    phonemes_from_audio,
    normalize_ipa_preserve_diphthongs,
    tokenize_words,
    phonemize_words_en,
)
from .utils_mfa import run_mfa_alignment, MFAAlignmentError

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

logger = logging.getLogger("speech_analyzer")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_TTS_MODEL = os.getenv("PRONUNCIATION_REFERENCE_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("PRONUNCIATION_REFERENCE_TTS_VOICE", "alloy")
_openai_client: Optional["OpenAI"] = None

if OPENAI_API_KEY and OpenAI:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to initialize OpenAI client: %s", exc)
        _openai_client = None


class PronunciationReferenceRequest(BaseModel):
    question_id: str = Field(..., description="Assignment question ID")
    expected_text: str = Field(..., description="Canonical sentence/phrase students must read")
    force_regenerate: bool = Field(
        default=False, description="Override cache even if text hash matches"
    )
    tts_model: Optional[str] = Field(
        default=None, description="Override OpenAI TTS model (default gpt-4o-mini-tts)"
    )
    tts_voice: Optional[str] = Field(
        default=None, description="Override OpenAI TTS voice (default alloy)"
    )


class PronunciationReferenceResponse(BaseModel):
    status: str
    reference: Optional[dict] = None
    generated_at: Optional[str] = None
    phoneme_count: Optional[int] = None


router = APIRouter(tags=["Pronunciation References"])
SAMPLE_RATE = 16000


def _normalize_text(text: str) -> str:
    stripped = text.strip()
    return " ".join(stripped.split())


def _hash_text(text: str) -> str:
    normalized = _normalize_text(text).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _ensure_openai_client() -> "OpenAI":
    if not _openai_client:
        raise HTTPException(
            status_code=503,
            detail="OpenAI TTS is not configured (missing OPENAI_API_KEY).",
        )
    return _openai_client


def _mp3_bytes_to_wav(mp3_bytes: bytes) -> tuple[bytes, np.ndarray]:
    """Convert returned MP3 bytes to 16k mono PCM for Allosaurus + storage."""
    tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp_mp3.write(mp3_bytes)
        tmp_mp3.flush()
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                tmp_mp3.name,
                "-ac",
                "1",
                "-ar",
                "16000",
                tmp_wav.name,
            ],
            check=True,
        )
        wav_bytes = Path(tmp_wav.name).read_bytes()
        audio, sr = sf.read(tmp_wav.name, dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        return wav_bytes, audio.astype(np.float32)
    finally:
        tmp_mp3.close()
        tmp_wav.close()
        try:
            os.unlink(tmp_mp3.name)
        except OSError:
            pass
        try:
            os.unlink(tmp_wav.name)
        except OSError:
            pass


def _tts_synthesize(text: str, model: str, voice: str) -> tuple[bytes, np.ndarray]:
    client = _ensure_openai_client()
    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
        mp3_bytes = response.read()
        return _mp3_bytes_to_wav(mp3_bytes)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.error("OpenAI TTS synthesis failed: %s", exc)
        raise HTTPException(status_code=502, detail="OpenAI TTS synthesis failed") from exc


@router.post(
    "/pronunciation",
    response_model=PronunciationReferenceResponse,
    summary="Ensure cached pronunciation reference assets exist",
)
async def ensure_pronunciation_reference(
    payload: PronunciationReferenceRequest,
    api_key_info: APIKeyInfo = Depends(api_key_bearer),
):
    if not db_manager.is_available():
        raise HTTPException(status_code=503, detail="Supabase is not configured.")

    normalized_text = _normalize_text(payload.expected_text)
    if not normalized_text:
        raise HTTPException(status_code=400, detail="expected_text cannot be empty.")

    text_hash = _hash_text(normalized_text)
    existing = await db_manager.get_pronunciation_reference(payload.question_id)
    if (
        existing
        and not payload.force_regenerate
        and existing.get("text_hash") == text_hash
        and existing.get("ready")
    ):
        return PronunciationReferenceResponse(status="cached", reference=existing)

    wav_bytes, audio_16k = _tts_synthesize(
        normalized_text,
        payload.tts_model or OPENAI_TTS_MODEL,
        payload.tts_voice or OPENAI_TTS_VOICE,
    )

    try:
        ipa_sequence = normalize_ipa_preserve_diphthongs(phonemes_from_audio(audio_16k))
        logger.info("Reference IPA sequence (%s): %s", payload.question_id, ipa_sequence[:40])
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to extract phonemes for %s: %s", payload.question_id, exc)
        raise HTTPException(status_code=500, detail="Phoneme extraction failed") from exc

    word_entries = []
    expected_words = tokenize_words(normalized_text)
    exp_ipas_words = phonemize_words_en(normalized_text)
    try:
        alignments = run_mfa_alignment(audio_16k, normalized_text)
    except MFAAlignmentError as exc:
        logger.warning("Skipping MFA-based segmentation for %s: %s", payload.question_id, exc)
        alignments = []

    def segment_bounds(idx: int) -> tuple[float, float]:
        if idx < len(alignments):
            return alignments[idx].start, alignments[idx].end
        span = len(audio_16k) / SAMPLE_RATE
        approx = span / max(1, len(expected_words))
        start = idx * approx
        end = start + approx
        return start, min(span, end)

    for idx, word in enumerate(expected_words):
        start_sec, end_sec = segment_bounds(idx)
        start_idx = max(0, int(max(0.0, start_sec - 0.05) * SAMPLE_RATE))
        end_idx = min(len(audio_16k), int(min(float(len(audio_16k)) / SAMPLE_RATE, end_sec + 0.05) * SAMPLE_RATE))
        if end_idx <= start_idx:
            end_idx = min(len(audio_16k), start_idx + int(0.1 * SAMPLE_RATE))
        segment_audio = audio_16k[start_idx:end_idx]
        text_phonemes_raw: List[str] = exp_ipas_words[idx] if idx < len(exp_ipas_words) else []
        segment_phonemes = normalize_ipa_preserve_diphthongs(text_phonemes_raw)
        if not segment_phonemes:
            logger.warning("Text phonemizer returned no phonemes for '%s'", word)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to phonemize word '{word}'. Please adjust the text and try again.",
            )
        word_entries.append({
            "word_text": word,
            "start": start_sec,
            "end": end_sec,
            "phonemes": segment_phonemes,
        })

    audio_duration_ms = int(len(audio_16k) / 16000 * 1000)
    phoneme_payload = {
        "expected_text": normalized_text,
        "ipa_sequence": ipa_sequence,
        "ipa_joined": " ".join(ipa_sequence),
        "generator": {
            "tts_model": payload.tts_model or OPENAI_TTS_MODEL,
            "tts_voice": payload.tts_voice or OPENAI_TTS_VOICE,
        },
        "audio_duration_ms": audio_duration_ms,
        "generated_at": datetime.utcnow().isoformat(),
        "word_phonemes": word_entries,
    }

    audio_url = await db_manager.upload_reference_audio(
        payload.question_id,
        text_hash,
        wav_bytes,
    )
    saved = await db_manager.upsert_pronunciation_reference(
        payload.question_id,
        normalized_text,
        text_hash,
        phoneme_payload,
        audio_url,
        ready=True,
    )

    return PronunciationReferenceResponse(
        status="generated",
        reference=saved,
        generated_at=phoneme_payload["generated_at"],
        phoneme_count=len(ipa_sequence),
    )
