from typing import List, Optional
import logging

import numpy as np
import soundfile as sf
import io
import tempfile
import subprocess
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

logger = logging.getLogger("speech_analyzer")

from .schemas import AnalyzeResponse, PronunciationResult, WordPronunciation, PhonemeScore
from .utils_text import tokenize_words, normalize_ipa, phonemize_words_en, phonemes_from_audio
from .utils_align import align_and_score

router = APIRouter()


def load_audio_to_mono16k(data: bytes) -> np.ndarray:
    import librosa
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
    except Exception:
        pass
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as in_f, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as out_f:
        in_f.write(data)
        in_f.flush()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-y", "-i", in_f.name,
            "-ac", "1", "-ar", "16000",
            "-f", "wav", out_f.name,
        ]
        subprocess.run(cmd, check=True)
        y, sr = sf.read(out_f.name, dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = np.mean(y, axis=1)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = 0.9 * (y / peak)
        return y.astype(np.float32)


def phonemize_words(text: str) -> List[List[str]]:
    return phonemize_words_en(text)


@router.post("/scripted", response_model=AnalyzeResponse)
async def scripted(
    expected_text: str = Form(...),
    browser_transcript: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    # Fast path: avoid decoding the audio when we only evaluate text vs text
    # Audio file is optional for text-based pronunciation analysis
    
    said_text = (browser_transcript or "").strip()
    if not said_text:
        raise HTTPException(status_code=400, detail="Missing browser_transcript for scripted mode.")
    
    # Validate audio file if provided (for backward compatibility)
    if file and file.content_type is not None:
        if not (file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))):
            raise HTTPException(status_code=400, detail="Invalid audio file format.")

    # Phonemize both
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
    return AnalyzeResponse(pronunciation=PronunciationResult(words=out_words, overall_score=overall), predicted_text=browser_transcript or "")


@router.post("/pronunciation", response_model=AnalyzeResponse)
async def pronunciation(
    file: UploadFile = File(...),
    expected_text: str = Form(...),
):
    """Pronunciation analysis using audio-to-phoneme recognition (no text transcription needed)."""
    if file.content_type is None or not (
        file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
    ):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")

    if not expected_text.strip():
        raise HTTPException(status_code=400, detail="Expected text is required for pronunciation analysis.")

    try:
        # Load and process audio
        audio = load_audio_to_mono16k(await file.read())
        
        # Extract phonemes directly from audio
        recognized_phonemes = phonemes_from_audio(audio)
        
        if not recognized_phonemes:
            raise HTTPException(status_code=500, detail="Could not extract phonemes from audio. Please check audio quality.")
        
        # Get expected phonemes from text
        exp_words = tokenize_words(expected_text)
        exp_ipas_words = phonemize_words_en(" ".join(exp_words))
        
        # Flatten expected phonemes with word boundaries
        expected_phonemes_flat = []
        word_boundaries = []  # Track which phoneme belongs to which word
        
        for word_idx, word_ipas in enumerate(exp_ipas_words):
            normalized_ipas = normalize_ipa(word_ipas)
            for phoneme in normalized_ipas:
                expected_phonemes_flat.append(phoneme)
                word_boundaries.append(word_idx)
        
        if not expected_phonemes_flat:
            raise HTTPException(status_code=500, detail="Could not generate expected phonemes from text.")
        
        # Align recognized phonemes to expected phonemes
        scores, pairs = align_and_score(expected_phonemes_flat, recognized_phonemes)
        
        # Group results back into words
        word_phoneme_data = {}  # word_idx -> list of (phoneme, score)
        recognized_phoneme_idx = 0
        
        for i, (expected_ph, recognized_ph, score) in enumerate(pairs):
            if expected_ph not in (None, "∅"):
                # This is an expected phoneme, find which word it belongs to
                if i < len(word_boundaries):
                    word_idx = word_boundaries[i]
                    if word_idx not in word_phoneme_data:
                        word_phoneme_data[word_idx] = []
                    
                    if score is not None:
                        word_phoneme_data[word_idx].append((expected_ph, float(score)))
                    else:
                        # Deletion (expected but not said)
                        word_phoneme_data[word_idx].append((expected_ph, 0.0))
            
            elif recognized_ph not in (None, "∅"):
                # Insertion (said but not expected) - penalize in current word context
                if word_boundaries and recognized_phoneme_idx < len(word_boundaries):
                    word_idx = word_boundaries[min(recognized_phoneme_idx, len(word_boundaries) - 1)]
                    if word_idx not in word_phoneme_data:
                        word_phoneme_data[word_idx] = []
                    word_phoneme_data[word_idx].append((recognized_ph, 0.0))
            
            if recognized_ph not in (None, "∅"):
                recognized_phoneme_idx += 1
        
        # Build output words
        out_words: List[WordPronunciation] = []
        for word_idx, word_text in enumerate(exp_words):
            if word_idx in word_phoneme_data:
                phoneme_data = word_phoneme_data[word_idx]
                phonemes = [PhonemeScore(ipa_label=ph, phoneme_score=score) for ph, score in phoneme_data]
                word_score = float(np.mean([score for _, score in phoneme_data])) if phoneme_data else 0.0
            else:
                # Word completely missing
                expected_ipas = normalize_ipa(exp_ipas_words[word_idx]) if word_idx < len(exp_ipas_words) else []
                phonemes = [PhonemeScore(ipa_label=ph, phoneme_score=0.0) for ph in expected_ipas]
                word_score = 0.0
            
            out_words.append(WordPronunciation(word_text=word_text, phonemes=phonemes, word_score=word_score))
        
        overall = float(np.mean([w.word_score for w in out_words])) if out_words else 0.0
        
        # Create a more readable predicted text that preserves word boundaries
        predicted_words = []
        for word_idx, word_text in enumerate(exp_words):
            if word_idx in word_phoneme_data:
                word_phonemes = [ph for ph, _ in word_phoneme_data[word_idx] if ph not in (None, "∅")]
                if word_phonemes:
                    predicted_words.append("/".join(word_phonemes))
                else:
                    predicted_words.append("∅")
            else:
                predicted_words.append("∅")
        
        # Fallback to simple phoneme sequence if word grouping fails
        if not predicted_words or all(w == "∅" for w in predicted_words):
            recognized_text = " ".join(recognized_phonemes)
        else:
            recognized_text = " | ".join(predicted_words)
        
        return AnalyzeResponse(pronunciation=PronunciationResult(words=out_words, overall_score=overall), predicted_text=recognized_text)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pronunciation analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


