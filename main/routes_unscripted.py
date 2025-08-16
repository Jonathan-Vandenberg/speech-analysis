from typing import Optional, List
import logging

import numpy as np
import soundfile as sf
import io
import tempfile
import subprocess
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import AnalyzeResponse, PronunciationResult, WordPronunciation, PhonemeScore
from .utils_text import tokenize_words, normalize_ipa, phonemize_words_en, phonemes_from_audio
from .utils_align import align_and_score

router = APIRouter()
logger = logging.getLogger("speech_analyzer")


async def analyze_audio_pronunciation(pred_words: List[str], pred_ipas_words: List[List[str]], recognized_phonemes: List[str], predicted_text: str) -> AnalyzeResponse:
    """Analyze pronunciation using audio-extracted phonemes vs predicted text phonemes."""
    # Flatten expected phonemes with word boundaries
    expected_phonemes_flat = []
    word_boundaries = []  # Track which phoneme belongs to which word
    
    for word_idx, word_ipas in enumerate(pred_ipas_words):
        normalized_ipas = normalize_ipa(word_ipas)
        for phoneme in normalized_ipas:
            expected_phonemes_flat.append(phoneme)
            word_boundaries.append(word_idx)
    
    if not expected_phonemes_flat:
        raise HTTPException(status_code=500, detail="Could not generate expected phonemes from transcription.")
    
    # Align recognized phonemes to expected phonemes
    logger.debug(f"Expected phonemes from transcription: {expected_phonemes_flat}")
    logger.debug(f"Recognized phonemes from audio: {recognized_phonemes}")
    scores, pairs = align_and_score(expected_phonemes_flat, recognized_phonemes)
    logger.debug(f"Alignment pairs: {pairs}")
    
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
    for word_idx, word_text in enumerate(pred_words):
        if word_idx in word_phoneme_data:
            phoneme_data = word_phoneme_data[word_idx]
            phonemes = [PhonemeScore(ipa_label=ph, phoneme_score=score) for ph, score in phoneme_data]
            word_score = float(np.mean([score for _, score in phoneme_data])) if phoneme_data else 0.0
        else:
            # Word completely missing
            expected_ipas = normalize_ipa(pred_ipas_words[word_idx]) if word_idx < len(pred_ipas_words) else []
            phonemes = [PhonemeScore(ipa_label=ph, phoneme_score=0.0) for ph in expected_ipas]
            word_score = 0.0
        
        out_words.append(WordPronunciation(word_text=word_text, phonemes=phonemes, word_score=word_score))
    
    overall = float(np.mean([w.word_score for w in out_words])) if out_words else 0.0
    recognized_text = " ".join(recognized_phonemes)  # Show what was actually recognized
    
    return AnalyzeResponse(pronunciation=PronunciationResult(words=out_words, overall_score=overall), predicted_text=predicted_text)


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
    # Fallback via ffmpeg for webm/ogg/m4a
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


def transcribe_faster_whisper(audio: np.ndarray) -> str:
    from faster_whisper import WhisperModel
    import os
    model_id = os.getenv("WHISPER_MODEL", "small")
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    model = WhisperModel(model_id, device="auto", compute_type=compute_type)
    segments, _ = model.transcribe(audio, beam_size=beam_size, vad_filter=True, word_timestamps=False, language="en", task="transcribe")
    texts: List[str] = [seg.text for seg in segments]
    return " ".join(t.strip() for t in texts).strip()


@router.post("/unscripted", response_model=AnalyzeResponse)
async def unscripted(
    file: UploadFile = File(...),
    browser_transcript: Optional[str] = Form(None),
    use_audio: bool = Form(False),
):
    if file.content_type is None or not (
        file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
    ):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")

    audio = load_audio_to_mono16k(await file.read())
    
    # Choose between audio transcription or browser transcript based on use_audio flag
    if use_audio:
        logger.debug("Using audio mode: Whisper transcription + audio phoneme extraction")
        # Use Whisper transcription as expected text, extract actual phonemes from audio
        predicted_text = transcribe_faster_whisper(audio)
        logger.debug(f"Whisper transcription: {predicted_text}")
        if not predicted_text.strip():
            raise HTTPException(status_code=500, detail="Could not transcribe audio. Please check audio quality.")
        
        # Extract phonemes directly from audio (actual pronunciation)
        recognized_phonemes = phonemes_from_audio(audio)
        if not recognized_phonemes:
            raise HTTPException(status_code=500, detail="Could not extract phonemes from audio. Please check audio quality.")
        
        # Get expected phonemes from transcription
        pred_words = tokenize_words(predicted_text)
        pred_ipas_words = phonemize_words_en(" ".join(pred_words))
        
        # Use audio-based phoneme analysis (similar to pronunciation mode)
        return await analyze_audio_pronunciation(pred_words, pred_ipas_words, recognized_phonemes, predicted_text)
        
    else:
        logger.debug("Using browser transcript mode: text-vs-text comparison")
        # Traditional text-vs-text comparison
        predicted_text = browser_transcript or ""
        said_text = (browser_transcript or "").strip()
        logger.debug(f"Browser transcript: {said_text}")
        
        if not said_text:
            raise HTTPException(status_code=400, detail="No text available - either provide browser_transcript or set use_audio=true")

        # Phonemize both
        pred_words = tokenize_words(predicted_text)
        said_words = tokenize_words(said_text)
        pred_ipas_words = phonemize_words_en(" ".join(pred_words))
        said_ipas_words = phonemize_words_en(" ".join(said_words))

    import difflib
    sm = difflib.SequenceMatcher(a=[w.lower() for w in pred_words], b=[w.lower() for w in said_words])
    out_words: List[WordPronunciation] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ei = i1 + k
                sj = j1 + k
                w_text = pred_words[ei]
                exp_ipas = normalize_ipa(pred_ipas_words[ei]) if ei < len(pred_ipas_words) else []
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
                word_scores = [p.phoneme_score for p in phonemes]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(word_scores)) if word_scores else 0.0))
        elif tag == "replace":
            exp_indices = list(range(i1, i2))
            said_indices = list(range(j1, j2))
            pairs: List[tuple[float, int, int, List[float]]] = []
            for ei in exp_indices:
                exp_ipas = normalize_ipa(pred_ipas_words[ei]) if ei < len(pred_ipas_words) else []
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
                w_text = pred_words[ei]
                exp_ipas = normalize_ipa(pred_ipas_words[ei]) if ei < len(pred_ipas_words) else []
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
                w_text = pred_words[ei]
                exp_ipas = normalize_ipa(pred_ipas_words[ei]) if ei < len(pred_ipas_words) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in exp_ipas]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=0.0))

    overall = float(np.mean([w.word_score for w in out_words])) if out_words else 0.0
    return AnalyzeResponse(pronunciation=PronunciationResult(words=out_words, overall_score=overall), predicted_text=predicted_text)


