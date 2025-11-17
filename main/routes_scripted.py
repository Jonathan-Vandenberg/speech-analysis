from typing import List, Optional
import logging
import re
from collections import defaultdict, deque
from difflib import SequenceMatcher
import numpy as np
import soundfile as sf
import io
import tempfile
import subprocess
import os
import json
import requests
import librosa
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends, Request

logger = logging.getLogger("speech_analyzer")

from .schemas import AnalyzeResponse, PronunciationResult, WordPronunciation, PhonemeScore
from .utils_text import tokenize_words, normalize_ipa, phonemize_words_en, phonemes_from_audio, normalize_ipa_preserve_diphthongs
from .utils_align import align_and_score
from .middleware import api_key_bearer, APIKeyInfo, request_tracker, generate_request_id
from .utils_mfa import run_mfa_alignment, MFAAlignmentError
from .routes_unscripted import transcribe_faster_whisper
from .database import db_manager

router = APIRouter()
SAMPLE_RATE = 16000
REFERENCE_AUDIO_CACHE: dict[str, np.ndarray] = {}
def load_audio_to_mono16k(data: bytes) -> np.ndarray:
    import librosa
    # Try soundfile first (works for WAV, FLAC, etc.)
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = 0.9 * (y / peak)
        return y.astype(np.float32)
    except Exception as e:
        logger.debug(f"Soundfile failed to read audio: {e}")
    
    # Try librosa directly (supports more formats including webm, m4a, etc.)
    # Note: librosa uses ffmpeg under the hood for webm, so this may still fail if ffmpeg is broken
    try:
        y, sr = librosa.load(io.BytesIO(data), sr=16000, mono=True)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = 0.9 * (y / peak)
        return y.astype(np.float32)
    except Exception as e:
        logger.debug(f"Librosa failed to read audio: {e}")
        # If librosa fails, it might be because it needs ffmpeg for webm
        # We'll fall through to the ffmpeg fallback
    
    # Fallback to ffmpeg (for formats that librosa doesn't support)
    # Save with a generic extension - ffmpeg will auto-detect the format
    in_f = tempfile.NamedTemporaryFile(suffix=".audio", delete=False)
    out_f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        in_f.write(data)
        in_f.flush()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", in_f.name,
            "-ac", "1", "-ar", "16000",
            "-f", "wav", out_f.name,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
        if not os.path.exists(out_f.name) or os.path.getsize(out_f.name) == 0:
            raise Exception("FFmpeg produced empty output file")
        y, sr = sf.read(out_f.name, dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = np.mean(y, axis=1)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = 0.9 * (y / peak)
        return y.astype(np.float32)
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg conversion timed out after 30 seconds")
        raise HTTPException(
            status_code=500,
            detail="Audio conversion timed out. The audio file may be corrupted or too large."
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else 'Unknown error')
        logger.error(f"FFmpeg conversion failed: {error_msg}")
        # Check if it's a library loading issue
        if "Library not loaded" in error_msg or "dyld" in error_msg:
            raise HTTPException(
                status_code=500,
                detail="Audio conversion failed due to missing system libraries. Please ensure FFmpeg and its dependencies (theora) are properly installed. Error: " + error_msg[:200]
            )
        raise HTTPException(
            status_code=500,
            detail=f"Audio conversion failed. FFmpeg error: {error_msg[:200]}"
        )
    except Exception as e:
        logger.error(f"Audio loading failed with all methods: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not process audio file. Please ensure the file is a valid audio format (WAV, MP3, M4A, WEBM, OGG). Error: {str(e)}"
        )
    finally:
        try:
            in_f.close()
            os.unlink(in_f.name)
        except Exception:
            pass
        try:
            out_f.close()
            os.unlink(out_f.name)
        except Exception:
            pass


def phonemize_words(text: str) -> List[List[str]]:
    return phonemize_words_en(text)


@router.post("/scripted", response_model=AnalyzeResponse)
async def scripted(
    request: Request,
    expected_text: str = Form(...),
    browser_transcript: Optional[str] = Form(None),
    analysis_type: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    api_key_info: Optional[APIKeyInfo] = Depends(api_key_bearer),
):
    # Start request tracking
    request_id = generate_request_id()
    request_tracker.start_request(request_id, api_key_info, "scripted", request)
    
    # Fast path: avoid decoding the audio when we only evaluate text vs text
    # Audio file is optional for text-based pronunciation analysis
    
    said_text = (browser_transcript or "").strip()
    if not said_text:
        raise HTTPException(status_code=400, detail="Missing browser_transcript for scripted mode.")
    
    # Validate audio file if provided (for backward compatibility)
    if file and file.content_type is not None:
        if not (file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))):
            raise HTTPException(status_code=400, detail="Invalid audio file format.")

    # Normalize text for comparison
    def normalize_for_comparison(text: str) -> str:
        return text.lower().strip()
    
    # Check if this is a reading assignment that requires exact text matching
    is_reading_assignment = analysis_type == "READING"
    
    # For reading assignments, check exact text match first
    text_matches_exactly = False
    if is_reading_assignment:
        expected_normalized = normalize_for_comparison(expected_text)
        said_normalized = normalize_for_comparison(said_text)
        text_matches_exactly = expected_normalized == said_normalized
        
        if not text_matches_exactly:
            # For reading assignments with wrong text, use proper word-level alignment
            exp_words = tokenize_words(expected_text)
            said_words = tokenize_words(said_text)
            exp_ipas_words = phonemize_words_en(" ".join(exp_words))
            said_ipas_words = phonemize_words_en(" ".join(said_words))
            
            # Use difflib to align words properly (handles insertions, deletions, substitutions)
            import difflib
            sm = difflib.SequenceMatcher(a=[w.lower() for w in exp_words], b=[w.lower() for w in said_words])
            out_words: List[WordPronunciation] = []

            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    # Words match - give good pronunciation scores
                    for k in range(i2 - i1):
                        ei = i1 + k
                        sj = j1 + k
                        w_text = exp_words[ei]
                        exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                        said_ipas = normalize_ipa(said_ipas_words[sj]) if sj < len(said_ipas_words) else []
                        scores, pairs = align_and_score(exp_ipas, said_ipas)
                        phonemes: List[PhonemeScore] = []
                        for a, b, sc in pairs:
                            if a not in (None, "∅") and b not in (None, "∅") and sc is not None:
                                phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=float(sc)))
                            elif a not in (None, "∅") and (b in (None, "∅")):
                                phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=0.0))
                            elif (a in (None, "∅")) and b not in (None, "∅"):
                                phonemes.append(PhonemeScore(ipa_label=b, phoneme_score=0.0))
                        word_score = float(np.mean([p.phoneme_score for p in phonemes])) if phonemes else 0.0
                        out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=word_score))
                elif tag == "replace":
                    # Words don't match - give low scores for expected words
                    for ei in range(i1, i2):
                        w_text = exp_words[ei]
                        exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                        phonemes = [PhonemeScore(ipa_label=p, phoneme_score=10.0) for p in exp_ipas]
                        word_score = 10.0 if phonemes else 0.0
                        out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=word_score))
                elif tag == "delete":
                    # Expected words not said - give low scores
                    for ei in range(i1, i2):
                        w_text = exp_words[ei]
                        exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                        phonemes = [PhonemeScore(ipa_label=p, phoneme_score=10.0) for p in exp_ipas]
                        word_score = 10.0 if phonemes else 0.0
                        out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=word_score))
                # Note: We ignore "insert" tag since we only show expected words, not extra words said
            
            # Calculate overall score as average of individual word scores
            overall_score = float(np.mean([w.word_score for w in out_words])) if out_words else 0.0
            response = AnalyzeResponse(
                pronunciation=PronunciationResult(words=out_words, overall_score=overall_score), 
                predicted_text=browser_transcript or ""
            )
            
            # Finish request tracking
            audio_data = await file.read() if file else None
            form_data = {
                "expected_text": expected_text,
                "browser_transcript": browser_transcript,
                "analysis_type": analysis_type,
                "deep_analysis": "false",
                "use_audio": "false"
            }
            await request_tracker.finish_request(request_id, response.model_dump(), form_data, audio_data)
            return response

    # Phonemize both (only for exact matches or pronunciation assignments)
    exp_words = tokenize_words(expected_text)
    said_words = tokenize_words(said_text)
    exp_ipas_words = phonemize_words_en(" ".join(exp_words))
    said_ipas_words = phonemize_words_en(" ".join(said_words))

    import difflib
    sm = difflib.SequenceMatcher(a=[w.lower() for w in exp_words], b=[w.lower() for w in said_words])
    out_words: List[WordPronunciation] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ei = i1 + k
                sj = j1 + k
                w_text = exp_words[ei]
                exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                said_ipas = normalize_ipa(said_ipas_words[sj]) if sj < len(said_ipas_words) else []
                scores, pairs = align_and_score(exp_ipas, said_ipas)
                phonemes: List[PhonemeScore] = []
                for a, b, sc in pairs:
                    if a not in (None, "∅") and b not in (None, "∅") and sc is not None:
                        phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=float(sc)))
                    elif a not in (None, "∅") and (b in (None, "∅")):
                        phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=0.0))
                    elif (a in (None, "∅")) and b not in (None, "∅"):
                        # insertion on said side → penalize as 0 for that phoneme
                        phonemes.append(PhonemeScore(ipa_label=b, phoneme_score=0.0))
                word_scores = [p.phoneme_score for p in phonemes]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(word_scores)) if word_scores else 0.0))
        elif tag == "replace":
            exp_indices = list(range(i1, i2))
            said_indices = list(range(j1, j2))
            pairs: List[tuple[float, int, int, List[float]]] = []
            for ei in exp_indices:
                exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                for sj in said_indices:
                    said_ipas = normalize_ipa(said_ipas_words[sj]) if sj < len(said_ipas_words) else []
                    scores, _ = align_and_score(exp_ipas, said_ipas)
                    mean_score = float(np.mean(scores)) if scores else 0.0
                    cost = 1.0 - (mean_score / 100.0)
                    pairs.append((cost, ei, sj, scores))
            pairs.sort(key=lambda x: x[0])
            used_e, used_s = set(), set()
            for cost, ei, sj, scores in pairs:
                if ei in used_e or sj in used_s:
                    continue
                used_e.add(ei); used_s.add(sj)
                w_text = exp_words[ei]
                exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                scores, pairs = align_and_score(exp_ipas, normalize_ipa(said_ipas_words[sj]) if sj < len(said_ipas_words) else [])
                phonemes: List[PhonemeScore] = []
                for a, b, sc in pairs:
                    if a not in (None, "∅") and b not in (None, "∅") and sc is not None:
                        phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=float(sc)))
                    elif a not in (None, "∅") and (b in (None, "∅")):
                        phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=0.0))
                    elif (a in (None, "∅")) and b not in (None, "∅"):
                        phonemes.append(PhonemeScore(ipa_label=b, phoneme_score=0.0))
                word_scores = [p.phoneme_score for p in phonemes]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(word_scores)) if word_scores else 0.0))
            for ei in exp_indices:
                if ei in used_e:
                    continue
                w_text = exp_words[ei]
                exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in exp_ipas]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=0.0))
        elif tag == "delete":
            for ei in range(i1, i2):
                w_text = exp_words[ei]
                exp_ipas = normalize_ipa(exp_ipas_words[ei]) if ei < len(exp_ipas_words) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in exp_ipas]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=0.0))

    overall = float(np.mean([w.word_score for w in out_words])) if out_words else 0.0
    response = AnalyzeResponse(pronunciation=PronunciationResult(words=out_words, overall_score=overall), predicted_text=browser_transcript or "")
    
    # Finish request tracking
    audio_data = await file.read() if file else None
    form_data = {
        "expected_text": expected_text,
        "browser_transcript": browser_transcript,
        "analysis_type": analysis_type,
        "deep_analysis": "false",  # scripted endpoint doesn't have deep analysis
        "use_audio": "false"  # scripted endpoint uses browser transcript
    }
    await request_tracker.finish_request(request_id, response.model_dump(), form_data, audio_data)
    
    return response


@router.post("/pronunciation", response_model=AnalyzeResponse)
async def pronunciation(
    request: Request,
    file: UploadFile = File(...),
    expected_text: str = Form(...),
    question_id: Optional[str] = Form(None),
    api_key_info: Optional[APIKeyInfo] = Depends(api_key_bearer),
):
    """Pronunciation analysis that uses Montreal Forced Aligner for word timing + Allosaurus scoring."""
    request_id = generate_request_id()
    request_tracker.start_request(request_id, api_key_info, "pronunciation", request)

    if file.content_type is None or not (
        file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
    ):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")
    if not expected_text.strip():
        raise HTTPException(status_code=400, detail="Expected text is required for pronunciation analysis.")

    def normalize_word_token(token: str) -> str:
        return re.sub(r"[^a-z0-9']+", "", (token or "").lower())
    
    def lexical_variants(token: str) -> set[str]:
        base = (token or "").lower()
        forms = {base}
        if base.endswith("ies") and len(base) > 3:
            forms.add(base[:-3] + "y")
        if base.endswith("es") and len(base) > 2:
            forms.add(base[:-2])
        if base.endswith("s") and len(base) > 1:
            forms.add(base[:-1])
        if base.endswith("ed") and len(base) > 2:
            forms.add(base[:-2])
            forms.add(base[:-1])
        if base.endswith("ing") and len(base) > 4:
            forms.add(base[:-3])
            forms.add(base[:-3] + "e")
        if base.endswith("er") and len(base) > 3:
            forms.add(base[:-2])
        return {f for f in forms if f}

    try:
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
        audio = load_audio_to_mono16k(audio_data)
        exp_words = tokenize_words(expected_text)
        if not exp_words:
            raise HTTPException(status_code=400, detail="Expected text must include pronounceable words.")
        exp_ipas_words = phonemize_words_en(" ".join(exp_words))

        reference_word_sequences: List[List[str]] = []
        reference_flat_sequence: Optional[List[str]] = None
        reference_record = None
        if question_id:
            try:
                reference_record = await db_manager.get_pronunciation_reference(question_id)
            except Exception as ref_exc:
                logger.warning("Failed to fetch pronunciation reference for %s: %s", question_id, ref_exc)
                reference_record = None
        reference_used = False
        if reference_record:
            payload = reference_record.get("phonemes_json")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                word_entries = payload.get("word_phonemes") or []
                if isinstance(word_entries, list):
                    for entry in word_entries:
                        seq = entry.get("phonemes") if isinstance(entry, dict) else None
                        if isinstance(seq, list):
                            cleaned = [str(p) for p in seq if isinstance(p, str) and p]
                            normalized_seq = normalize_ipa_preserve_diphthongs(cleaned)
                            reference_word_sequences.append(normalized_seq)
                sequence = payload.get("ipa_sequence")
                if isinstance(sequence, list):
                    reference_flat_sequence = [str(p) for p in sequence if isinstance(p, str) and p]
                if reference_word_sequences or reference_flat_sequence:
                    reference_used = True
            audio_url = reference_record.get("audio_url")
            if audio_url:
                logger.info("Reference audio cached at %s for %s", audio_url, question_id)

        try:
            recognized_transcript = transcribe_faster_whisper(audio)
            logger.info("📝 Whisper transcript: %s", recognized_transcript)
        except Exception as transcribe_exc:
            logger.warning("Whisper transcription failed: %s", transcribe_exc)
            recognized_transcript = ""
        recognized_words = tokenize_words(recognized_transcript)
        recognized_ipas_words = phonemize_words_en(" ".join(recognized_words)) if recognized_words else []
        recognized_inventory: List[dict[str, object]] = []
        for idx, raw_word in enumerate(recognized_words):
            normalized = normalize_word_token(raw_word)
            if not normalized:
                continue
            ipas = normalize_ipa(recognized_ipas_words[idx]) if idx < len(recognized_ipas_words) else []
            recognized_inventory.append({"normalized": normalized, "phonemes": ipas})
        lexical_strict_threshold = float(os.getenv("LEXICAL_STRICT_THRESHOLD", "0.9"))
        lexical_cursor = 0

        def consume_next(normalized_word: str) -> Optional[int]:
            nonlocal lexical_cursor
            if not normalized_word:
                return None
            for idx in range(lexical_cursor, len(recognized_inventory)):
                entry = recognized_inventory[idx]
                rec_norm = entry.get("normalized")
                if not isinstance(rec_norm, str) or not rec_norm:
                    continue
                char_ratio = SequenceMatcher(None, normalized_word, rec_norm).ratio()
                if char_ratio >= lexical_strict_threshold:
                    lexical_cursor = idx + 1
                    return idx
            return None

        # Run Montreal Forced Aligner and capture per-word timings
        logger.info("🎯 Using Montreal Forced Aligner (MFA) for alignment...")
        try:
            word_alignments = run_mfa_alignment(audio, expected_text)
        except MFAAlignmentError as err:
            logger.error("MFA alignment failed: %s", err)
            raise HTTPException(status_code=500, detail=str(err))

        per_word_phonemes: List[List[str]] = []
        reference_sequences_by_word: List[List[str]] = []
        reference_flat_sequence: Optional[List[str]] = None
        if reference_word_sequences:
            reference_sequences_by_word = [
                [str(p) for p in reference_word_sequences[idx]] if idx < len(reference_word_sequences) else []
                for idx in range(len(exp_words))
            ]
            reference_used = True
        if reference_flat_sequence is None and reference_record:
            payload = reference_record.get("phonemes_json", {}) if isinstance(reference_record, dict) else {}
            reference_ipa = payload.get("ipa_sequence")
            if isinstance(reference_ipa, list):
                reference_flat_sequence = [str(p) for p in reference_ipa if isinstance(p, str) and p]

        flat_expected: List[str] = []
        for idx, word_text in enumerate(exp_words):
            if reference_used:
                seq = reference_sequences_by_word[idx]
            else:
                seq = normalize_ipa_preserve_diphthongs(exp_ipas_words[idx]) if idx < len(exp_ipas_words) else []
            per_word_phonemes.append(seq)
            flat_expected.extend(seq)
        if reference_used and question_id:
            logger.info("Using cached reference phonemes for question %s", question_id)
        if reference_flat_sequence:
            flat_expected = reference_flat_sequence
            if question_id:
                logger.debug("Reference phonemes for %s: %s", question_id, flat_expected[:40])

        def extract_segment(idx: int) -> np.ndarray:
            if idx < len(word_alignments):
                alignment = word_alignments[idx]
                start_idx = max(0, int(max(0.0, alignment.start - 0.05) * SAMPLE_RATE))
                end_idx = min(len(audio), int(min(float(len(audio)) / SAMPLE_RATE, alignment.end + 0.05) * SAMPLE_RATE))
                if end_idx <= start_idx:
                    end_idx = min(len(audio), start_idx + int(0.1 * SAMPLE_RATE))
                return audio[start_idx:end_idx]
            return audio

        segment_cache: dict[int, List[str]] = {}

        def get_segment_phonemes(idx: int) -> List[str]:
            if idx in segment_cache:
                return segment_cache[idx]
            segment_audio = extract_segment(idx)
            try:
                recognized_segment = normalize_ipa_preserve_diphthongs(phonemes_from_audio(segment_audio))
            except Exception as seg_exc:
                logger.warning("Per-word phoneme extraction failed for '%s': %s", exp_words[idx], seg_exc)
                recognized_segment = []
            segment_cache[idx] = recognized_segment
            return recognized_segment

        reference_segments_audio: List[Optional[np.ndarray]] = []
        if reference_used and reference_record:
            audio_url = reference_record.get("audio_url")
            payload = reference_record.get("phonemes_json")
            word_entries: List[dict] = []
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                entries = payload.get("word_phonemes") or []
                if isinstance(entries, list):
                    word_entries = [entry for entry in entries if isinstance(entry, dict)]
            if audio_url and word_entries:
                try:
                    ref_audio = _load_reference_audio(audio_url)
                    reference_segments_audio = _slice_reference_segments(ref_audio, word_entries)
                except Exception as exc:
                    logger.warning("Failed to prepare reference audio segments for %s: %s", question_id, exc)
                    reference_segments_audio = []
        recognized_phonemes_flat = normalize_ipa_preserve_diphthongs(phonemes_from_audio(audio))
        logger.debug("Expected phonemes (%d): %s", len(flat_expected), flat_expected[:40])
        logger.debug("Recognized phonemes: %s", recognized_phonemes_flat[:40])

        _, pairs = align_and_score(flat_expected, recognized_phonemes_flat)
        aligned_scores: List[float] = []
        for a, b, sc in pairs:
            if a in (None, "∅"):
                continue
            score_val = float(sc) if sc is not None else 0.0
            aligned_scores.append(score_val)

        out_words: List[WordPronunciation] = []
        score_idx = 0
        for idx, word_text in enumerate(exp_words):
            expected_phonemes = per_word_phonemes[idx]
            normalized_word = normalize_word_token(word_text)
            phoneme_scores: List[PhonemeScore] = []
            match_idx = None
            if normalized_word:
                variant_list = [normalized_word]
                variants = lexical_variants(normalized_word)
                for variant in variants:
                    if variant not in variant_list:
                        variant_list.append(variant)
                for variant in variant_list:
                    match_idx = consume_next(variant)
                    if match_idx is not None:
                        break
            if match_idx is None:
                if expected_phonemes:
                    phoneme_scores = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in expected_phonemes]
                out_words.append(WordPronunciation(word_text=word_text, phonemes=phoneme_scores, word_score=0.0))
                continue

            reference_audio_seg = reference_segments_audio[idx] if idx < len(reference_segments_audio) else None
            if (
                reference_audio_seg is not None
                and reference_audio_seg.size > 0
            ):
                segment_audio = extract_segment(idx)
                if segment_audio.size > 0:
                    similarity = _audio_similarity(segment_audio, reference_audio_seg)
                    if similarity is not None:
                        phoneme_scores = [
                            PhonemeScore(ipa_label=p, phoneme_score=similarity) for p in expected_phonemes
                        ]
                        word_score = similarity if phoneme_scores else similarity or 0.0
                        out_words.append(WordPronunciation(word_text=word_text, phonemes=phoneme_scores, word_score=word_score))
                        continue
            for phoneme in expected_phonemes:
                if score_idx < len(aligned_scores):
                    phoneme_scores.append(PhonemeScore(ipa_label=phoneme, phoneme_score=aligned_scores[score_idx]))
                    score_idx += 1
                else:
                    phoneme_scores.append(PhonemeScore(ipa_label=phoneme, phoneme_score=0.0))
            word_score = float(np.mean([p.phoneme_score for p in phoneme_scores])) if phoneme_scores else 0.0
            need_segment_rescore = False
            if not phoneme_scores or any(p.phoneme_score == 0.0 for p in phoneme_scores):
                need_segment_rescore = True
            if need_segment_rescore and expected_phonemes:
                segment_phonemes = get_segment_phonemes(idx)
                if segment_phonemes:
                    _, seg_pairs = align_and_score(expected_phonemes, segment_phonemes, detailed_logging=False)
                    rescore: List[PhonemeScore] = []
                    for exp_ph, _, sc in seg_pairs:
                        if exp_ph in (None, "∅"):
                            continue
                        score_val = float(sc) if sc is not None else 0.0
                        rescore.append(PhonemeScore(ipa_label=exp_ph, phoneme_score=score_val))
                    if rescore:
                        phoneme_scores = rescore
                        word_score = float(np.mean([p.phoneme_score for p in phoneme_scores]))
                else:
                    phoneme_scores = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in expected_phonemes]
                    word_score = 0.0
            out_words.append(WordPronunciation(word_text=word_text, phonemes=phoneme_scores, word_score=word_score))

        overall = float(np.mean([w.word_score for w in out_words])) if out_words else 0.0
        logger.info("🎯 MFA pronunciation overall score: %.1f%%", overall)

        response = AnalyzeResponse(
            pronunciation=PronunciationResult(words=out_words, overall_score=overall),
            predicted_text=expected_text,
        )

        form_data = {
            "expected_text": expected_text,
            "deep_analysis": "false",
            "use_audio": "true",
        }
        await request_tracker.finish_request(request_id, response.model_dump(), form_data, audio_data)
        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pronunciation analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


def _load_reference_audio(url: str) -> np.ndarray:
    if url in REFERENCE_AUDIO_CACHE:
        return REFERENCE_AUDIO_CACHE[url]
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to download reference audio ({resp.status_code})")
    audio = load_audio_to_mono16k(resp.content)
    REFERENCE_AUDIO_CACHE[url] = audio
    return audio


def _slice_reference_segments(audio: np.ndarray, word_entries: List[dict]) -> List[Optional[np.ndarray]]:
    segments: List[Optional[np.ndarray]] = []
    total_len = len(audio)
    for entry in word_entries:
        start_sec = float(entry.get("start") or 0.0)
        end_sec = float(entry.get("end") or start_sec + 0.25)
        if end_sec <= start_sec:
            end_sec = start_sec + 0.25
        start_idx = max(0, int(start_sec * SAMPLE_RATE))
        end_idx = min(total_len, int(end_sec * SAMPLE_RATE))
        if end_idx <= start_idx:
            end_idx = min(total_len, start_idx + int(0.1 * SAMPLE_RATE))
        if end_idx <= start_idx:
            segments.append(None)
            continue
        seg = audio[start_idx:end_idx]
        segments.append(seg)
    return segments


def _audio_similarity(student_audio: np.ndarray, reference_audio: np.ndarray) -> Optional[float]:
    student_vec = _mfcc_summary(student_audio)
    reference_vec = _mfcc_summary(reference_audio)
    if student_vec is None or reference_vec is None:
        return None
    denom = float(np.linalg.norm(student_vec) * np.linalg.norm(reference_vec))
    if denom == 0.0:
        return None
    cosine = float(np.clip(np.dot(student_vec, reference_vec) / denom, -1.0, 1.0))
    similarity = (cosine + 1.0) / 2.0  # map [-1,1] -> [0,1]
    return similarity * 100.0


def _mfcc_summary(audio: np.ndarray) -> Optional[np.ndarray]:
    if audio.size < int(0.05 * SAMPLE_RATE):
        return None
    try:
        mfcc = librosa.feature.mfcc(y=audio.astype(float), sr=SAMPLE_RATE, n_mfcc=13)
        if mfcc.size == 0:
            return None
        mean = np.mean(mfcc, axis=1)
        std = np.std(mfcc, axis=1)
        return np.concatenate([mean, std])
    except Exception as exc:
        logger.warning("MFCC extraction failed: %s", exc)
        return None
