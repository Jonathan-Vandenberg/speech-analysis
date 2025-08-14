import io
import os
import tempfile
import subprocess
from typing import List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import difflib
import unicodedata
 

# ASR
from faster_whisper import WhisperModel
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import math

# Phone recognition

# IPA feature distances
import panphon
 


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
PHONEME_MODEL_ID = os.getenv("PHONEME_MODEL_ID", "facebook/wav2vec2-lv-60-espeak-cv-ft")
FRONTEND_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
MAX_AUDIO_SECONDS = float(os.getenv("MAX_AUDIO_SECONDS", "120"))
TARGET_SR = 16000
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ALIGN_LOG_LIMIT = int(os.getenv("ALIGN_LOG_LIMIT", "30"))
MIN_FRAMES_PER_WORD = int(os.getenv("MIN_FRAMES_PER_WORD", "1"))

# Logger
logger = logging.getLogger("speech_analyzer")
if not logger.handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


 

# Lazy-loaded singletons
_whisper_model: Optional[WhisperModel] = None
_feature_table = None
_phoneme_processor = None
_phoneme_model = None


class PhonemeScore(BaseModel):
    ipa_label: str
    phoneme_score: float


class WordPronunciation(BaseModel):
    word_text: str
    phonemes: List[PhonemeScore]
    word_score: float


class PronunciationResult(BaseModel):
    words: List[WordPronunciation]
    overall_score: float


class AnalyzeResponse(BaseModel):
    pronunciation: PronunciationResult
    predicted_text: str


def _lazy_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
    return _whisper_model


def _lazy_allosaurus():
    global _allosaurus_recognizer
    if _allosaurus_recognizer is None:
        _allosaurus_recognizer = read_recognizer()
    return _allosaurus_recognizer


def _lazy_feature_table() -> panphon.FeatureTable:
    global _feature_table
    if _feature_table is None:
        _feature_table = panphon.FeatureTable()
    return _feature_table


def _lazy_phoneme_model():
    global _phoneme_processor, _phoneme_model
    if _phoneme_processor is None or _phoneme_model is None:
        # Prefer explicit processor/model classes per HF docs
        try:
            _phoneme_processor = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL_ID)
        except Exception:
            _phoneme_processor = AutoProcessor.from_pretrained(PHONEME_MODEL_ID)
        try:
            _phoneme_model = Wav2Vec2ForCTC.from_pretrained(
                PHONEME_MODEL_ID,
                use_safetensors=True,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:
            # Fallback: load PyTorch weights (.bin). This requires torch>=2.6.
            logger.warning("Safetensors load failed (%s). Falling back to PyTorch weights.", exc)
            try:
                _phoneme_model = Wav2Vec2ForCTC.from_pretrained(
                    PHONEME_MODEL_ID,
                    use_safetensors=False,
                    low_cpu_mem_usage=True,
                )
            except Exception as exc2:
                msg2 = str(exc2)
                if "require users to upgrade torch" in msg2 or "CVE-2025-32434" in msg2:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Torch version too old for loading PyTorch weights. "
                            "Please upgrade torch to >= 2.6.0."
                        ),
                    )
                raise
        _phoneme_model.eval()
    return _phoneme_processor, _phoneme_model


TARGET_IPA_INVENTORY: List[str] = []

# Heuristic confusion similarities to soften near-matches (0..1 similarity)
CONFUSION_SIMILARITY: dict[tuple[str, str], float] = {
    ("o", "ɔ"): 0.85, ("ʌ", "ə"): 0.8, ("ɒ", "ɑ"): 0.85,
    ("a", "ɑ"): 0.9,
    ("i", "ɪ"): 0.85, ("e", "ɛ"): 0.85, ("ʊ", "u"): 0.8,
    ("j", "i"): 0.75, ("r", "ɹ"): 0.9, ("n", "ɴ"): 0.85,
    ("ʃ", "ɕ"): 0.85, ("tʃ", "tɕ"): 0.85, ("dʒ", "dʑ"): 0.85,
    ("h", "x"): 0.7, ("θ", "ð"): 0.7,
}


 


def load_audio_to_mono16k(data: bytes) -> np.ndarray:
    # First try libsndfile (wav/flac/ogg). If it fails (e.g., webm/opus), fallback to ffmpeg.
    try:
        buf = io.BytesIO(data)
        y, sr = sf.read(buf, dtype="float32", always_2d=False)
        if getattr(y, "ndim", 1) > 1:
            y = np.mean(y, axis=1)
        if sr != TARGET_SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        # peak normalize to improve recognizer robustness
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 0:
            y = 0.9 * (y / peak)
        return y.astype(np.float32)
    except Exception:
        pass

    # Fallback: use ffmpeg to decode any container/codec to 16k mono wav
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as in_f, \
        	tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as out_f:
            in_f.write(data)
            in_f.flush()
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-y", "-i", in_f.name,
                "-ac", "1", "-ar", str(TARGET_SR),
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
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unsupported or corrupt audio: {exc}")


def transcribe_with_words(audio: np.ndarray) -> tuple[str, List[dict]]:
    model = _lazy_whisper()
    # Force English to avoid language detection variability
    segments, _ = model.transcribe(audio, beam_size=5, vad_filter=True, word_timestamps=True, language="en", task="transcribe")
    texts: List[str] = []
    asr_words: List[dict] = []
    for seg in segments:
        texts.append(seg.text)
        for w in getattr(seg, "words", []) or []:
            # faster-whisper Word has .word and .prob
            try:
                asr_words.append({
                    "word": w.word,
                    "prob": float(getattr(w, "prob", 0.5)),
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0)),
                })
            except Exception:
                pass
    pred = " ".join(t.strip() for t in texts).strip()
    logger.info("ASR predicted_text='%s'", pred)
    return pred, asr_words


 


 


 


def split_diphthongs(seq: List[str]) -> List[str]:
    out: List[str] = []
    for p in seq:
        if p == "eɪ":
            out += ["e", "ɪ"]
        elif p == "oʊ":
            out += ["o", "ʊ"]
        elif p == "aɪ":
            out += ["a", "ɪ"]
        elif p == "aʊ":
            out += ["a", "ʊ"]
        elif p == "ɔɪ":
            out += ["ɔ", "ɪ"]
        elif p == "ɪə":
            out += ["ɪ", "ə"]
        elif p == "ʊə":
            out += ["ʊ", "ə"]
        else:
            out.append(p)
    return out


def ipa_feature_distance(p1: str, p2: str) -> float:
    ft = _lazy_feature_table()
    # Exact match → zero distance
    if p1 == p2:
        return 0.0
    # Heuristic confusion softening
    if (p1, p2) in CONFUSION_SIMILARITY:
        return 1.0 - CONFUSION_SIMILARITY[(p1, p2)]
    if (p2, p1) in CONFUSION_SIMILARITY:
        return 1.0 - CONFUSION_SIMILARITY[(p2, p1)]
    # Convert multi-char phones to a single averaged feature vector using word_to_vector_list
    def vec_for_phone(phone: str) -> Optional[np.ndarray]:
        try:
            vectors = ft.word_to_vector_list(phone)
            if not vectors:
                return None
            return np.mean(np.array(vectors, dtype=float), axis=0)
        except Exception:
            return None

    v1 = vec_for_phone(p1)
    v2 = vec_for_phone(p2)
    if v1 is None or v2 is None:
        logger.debug("No panphon vector for pair (%s, %s)", p1, p2)
        return 1.0
    # Cosine similarity based score → distance
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        logger.debug("Zero-norm panphon vector for pair (%s, %s)", p1, p2)
        return 1.0
    cos_sim = float(np.dot(v1, v2) / denom)
    # convert similarity [-1,1] to distance [0,1]
    return float(1.0 - (cos_sim + 1.0) / 2.0)


def ctc_phone_posteriors(audio: np.ndarray) -> tuple[np.ndarray, List[str]]:
    processor, model = _lazy_phoneme_model()
    with torch.no_grad():
        inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
        logits = model(inputs.input_values).logits[0]  # (time, vocab)
        post = torch.nn.functional.log_softmax(logits, dim=-1).exp().cpu().numpy()
        # sanitize any NaNs/Infs defensively
        post = np.nan_to_num(post, nan=0.0, posinf=0.0, neginf=0.0)
        vocab = processor.tokenizer.get_vocab()  # token -> id
        labels = [""] * (max(vocab.values()) + 1)
        for token, idx in vocab.items():
            if 0 <= idx < len(labels):
                labels[idx] = token
    return post, labels


def ctc_align_scores(expected_phones: List[str], posteriors: np.ndarray, labels: List[str]) -> List[float]:
    # Map expected phones to label ids (fallback to nearest label by simple string match)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    seq_ids = []
    for p in expected_phones:
        if p in label_to_idx:
            seq_ids.append(label_to_idx[p])
        else:
            # crude fallback: find label with max overlap
            best = max(labels, key=lambda l: sum(ch in l for ch in p))
            seq_ids.append(label_to_idx.get(best, 0))
    # Simple Viterbi over posteriors to align sequence
    T, V = posteriors.shape
    N = len(seq_ids)
    if N == 0:
        return []
    dp = np.full((T+1, N+1), -1e9)
    prev = np.full((T+1, N+1), -1, dtype=int)
    dp[0, 0] = 0.0
    for t in range(1, T+1):
        dp[t, 0] = dp[t-1, 0] + np.log(max(1e-9, posteriors[t-1, label_to_idx.get("<pad>", 0)]))
        prev[t, 0] = 0
        for n in range(1, N+1):
            stay = dp[t-1, n] + np.log(max(1e-9, posteriors[t-1, seq_ids[n-1]]))
            advance = dp[t-1, n-1] + np.log(max(1e-9, posteriors[t-1, seq_ids[n-1]]))
            if advance > stay:
                dp[t, n] = advance
                prev[t, n] = 1
            else:
                dp[t, n] = stay
                prev[t, n] = 0
    # Traceback to collect per-phone frame ranges
    t, n = T, N
    spans = [(0, 0)] * N
    cur_end = T
    while n > 0:
        while t > 0 and prev[t, n] == 0:
            t -= 1
        # now either start of advance or t==0
        start = max(0, t-1)
        spans[n-1] = (start, cur_end)
        cur_end = start
        n -= 1
        t -= 1
    # Score each phone as mean posterior over its span
    scores = []
    for i, (s, e) in enumerate(spans):
        if e <= s:
            scores.append(0.0)
        else:
            pid = seq_ids[i]
            scores.append(float(np.clip(posteriors[s:e, pid].mean()*100.0, 0.0, 100.0)))
    return scores


def _time_to_frame_indices(start_s: float, end_s: float, total_audio_seconds: float, total_frames: int) -> tuple[int, int]:
    if total_frames <= 0 or total_audio_seconds <= 0:
        return 0, 0
    # clamp within [0, total_audio_seconds]
    start_s = max(0.0, min(total_audio_seconds, float(start_s)))
    end_s = max(0.0, min(total_audio_seconds, float(end_s)))
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    sec_per_frame = total_audio_seconds / float(total_frames)
    s_idx = int(start_s / sec_per_frame)
    e_idx = int(end_s / sec_per_frame)
    s_idx = max(0, min(total_frames, s_idx))
    e_idx = max(s_idx, min(total_frames, e_idx))
    # ensure at least MIN_FRAMES_PER_WORD frames
    if e_idx - s_idx < max(1, MIN_FRAMES_PER_WORD):
        e_idx = min(total_frames, s_idx + max(1, MIN_FRAMES_PER_WORD))
    return s_idx, e_idx


def _align_words_indices(source_words: List[str], target_words: List[str]) -> List[tuple[int, int]]:
    # Return pairs of indices (i, j) where words match via LCS
    sm = difflib.SequenceMatcher(a=[w.lower() for w in source_words], b=[w.lower() for w in target_words])
    pairs: List[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(0, i2 - i1):
                pairs.append((i1 + k, j1 + k))
    return pairs


def _tokenize_words(text: str) -> List[str]:
    import re
    return [w for w in re.findall(r"[A-Za-z]+'?[A-Za-z]+|[A-Za-z]+", text)]


def phonemize_words(text: str) -> List[List[str]]:
    # Phonemize each word to IPA using espeak to keep inventories consistent
    words = _tokenize_words(text)
    ipa_per_word: List[List[str]] = []
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        for w in words:
            ipa = phonemize(
                w,
                language="en-us",
                backend="espeak",
                strip=True,
                with_stress=False,
                separator=Separator(phone=" ", word="|"),  # ensure phones split on spaces; word sep different
            )
            phones = [p for p in ipa.strip().split() if p]
            ipa_per_word.append(phones)
    except Exception as exc:
        logger.warning("Phonemizer failed: %s", exc)
        for _ in words:
            ipa_per_word.append([])
    return ipa_per_word


def score_text_vs_text(expected_text: str, said_text: str) -> PronunciationResult:
    # Tokenize
    exp_words = _tokenize_words(expected_text)
    said_words = _tokenize_words(said_text)

    # Phonemize per word
    exp_ipas_per_word = phonemize_words(" ".join(exp_words))
    said_ipas_per_word = phonemize_words(" ".join(said_words))

    def normalize_seq(seq: List[str]) -> List[str]:
        out: List[str] = []
        for p in split_diphthongs(seq):
            nf = unicodedata.normalize("NFD", p)
            base = "".join(ch for ch in nf if not unicodedata.combining(ch) and ch not in {"ː", "ˑ"})
            base = (base
                .replace("tɕ", "tʃ").replace("tɕʰ", "tʃ").replace("tʂ", "tʃ")
                .replace("dʑ", "dʒ")
                .replace("ɕ", "ʃ").replace("ʂ", "ʃ").replace("ʐ", "ʒ")
                .replace("ɹ", "r").replace("ɾ", "r")
                .replace("x", "h").replace("y", "j")
                .replace("ɴ", "n").replace("ɫ", "l")
                .replace("ɒ", "ɑ").replace("ɤ", "ʌ")
            )
            if base:
                out.append(base)
        return out

    # Align word sequences (case-insensitive)
    sm = difflib.SequenceMatcher(a=[w.lower() for w in exp_words], b=[w.lower() for w in said_words])
    words_out: List[WordPronunciation] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Fast path: same words, align 1:1
            for k in range(i2 - i1):
                exp_idx = i1 + k
                said_idx = j1 + k
                w_text = exp_words[exp_idx]
                exp_ipas = normalize_seq(exp_ipas_per_word[exp_idx]) if exp_idx < len(exp_ipas_per_word) else []
                said_ipas = normalize_seq(said_ipas_per_word[said_idx]) if said_idx < len(said_ipas_per_word) else []
                scores, _pairs = align_and_score(exp_ipas, said_ipas)  # type: ignore[misc]
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=float(scores[i]) if i < len(scores) else 0.0) for i, p in enumerate(exp_ipas)]
                wp = WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(scores)) if scores else 0.0)
                words_out.append(wp)
        elif tag == "replace":
            # Build a phoneme-similarity-based matching within the span
            exp_indices = list(range(i1, i2))
            said_indices = list(range(j1, j2))
            pairs_all: List[tuple[float, int, int, List[float]]] = []  # (cost, exp_idx, said_idx, scores)
            for ei in exp_indices:
                exp_ipas = normalize_seq(exp_ipas_per_word[ei]) if ei < len(exp_ipas_per_word) else []
                for sj in said_indices:
                    said_ipas = normalize_seq(said_ipas_per_word[sj]) if sj < len(said_ipas_per_word) else []
                    scores, _pairs = align_and_score(exp_ipas, said_ipas)  # type: ignore[misc]
                    mean_score = float(np.mean(scores)) if scores else 0.0
                    cost = 1.0 - (mean_score / 100.0)
                    pairs_all.append((cost, ei, sj, scores))
            # Greedy select minimal cost without collisions
            pairs_all.sort(key=lambda x: x[0])
            used_exp: set[int] = set()
            used_said: set[int] = set()
            matched: List[tuple[int, int, List[float]]] = []
            for cost, ei, sj, scores in pairs_all:
                if ei in used_exp or sj in used_said:
                    continue
                used_exp.add(ei)
                used_said.add(sj)
                matched.append((ei, sj, scores))
                if len(used_exp) >= len(exp_indices) or len(used_said) >= len(said_indices):
                    break
            # Emit matched word scores
            for ei, sj, scores in matched:
                w_text = exp_words[ei]
                exp_ipas = normalize_seq(exp_ipas_per_word[ei]) if ei < len(exp_ipas_per_word) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=float(scores[i]) if i < len(scores) else 0.0) for i, p in enumerate(exp_ipas)]
                words_out.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(scores)) if scores else 0.0))
            # Any expected words not matched → zero score
            for ei in exp_indices:
                if ei in used_exp:
                    continue
                w_text = exp_words[ei]
                exp_ipas = split_diphthongs(exp_ipas_per_word[ei]) if ei < len(exp_ipas_per_word) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in exp_ipas]
                words_out.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=0.0))
        elif tag == "delete":
            for exp_idx in range(i1, i2):
                w_text = exp_words[exp_idx]
                exp_ipas = normalize_seq(exp_ipas_per_word[exp_idx]) if exp_idx < len(exp_ipas_per_word) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=0.0) for p in exp_ipas]
                words_out.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=0.0))
        elif tag == "insert":
            # Insertions in said_text do not correspond to expected words, skip in expected-oriented output
            continue

    overall = float(np.mean([w.word_score for w in words_out])) if words_out else 0.0
    return PronunciationResult(words=words_out, overall_score=overall)


def forced_align_with_whisperx(audio: np.ndarray, expected_text: str) -> List[dict]:
    # Save PCM16 wav to disk for whisperx
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        sf.write(tmp.name, pcm16, TARGET_SR, subtype="PCM_16")

        import whisperx
        device = "cpu"
        batch_size = 16
        compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", "int8")
        model = whisperx.load_model(WHISPER_MODEL, device, compute_type=compute_type)
        align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
        # Build one segment with the expected transcript; whisperx will force-align it to audio
        segments = [{"text": expected_text, "start": 0.0, "end": len(audio)/TARGET_SR}]
        result_aligned = whisperx.align(segments, align_model, metadata, tmp.name, device)
        # result_aligned["segments"][i]["words"] has start/end per word
        words_out = []
        for seg in result_aligned.get("segments", []):
            for w in seg.get("words", []):
                words_out.append({
                    "word": w.get("word", ""),
                    "start": float(w.get("start", 0.0)),
                    "end": float(w.get("end", 0.0)),
                    "score": float(w.get("score", 1.0)),
                })
        return words_out


def recognize_segment_phones(audio: np.ndarray, start_s: float, end_s: float) -> List[str]:
    start_idx = max(0, int(start_s * TARGET_SR))
    end_idx = min(len(audio), int(end_s * TARGET_SR))
    if end_idx <= start_idx:
        return []
    segment = audio[start_idx:end_idx]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        pcm16 = np.clip(segment, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        sf.write(tmp.name, pcm16, TARGET_SR, subtype="PCM_16")
        try:
            rec = read_recognizer()
            if hasattr(rec, "recognize"):
                out = rec.recognize(tmp.name)
            elif hasattr(rec, "predict"):
                out = rec.predict(tmp.name)
            else:
                out = ""
            raw = [p for p in (out or "").strip().split() if p]
            # normalize to expected IPA-ish inventory to improve alignment
            import unicodedata
            def norm(ph: str) -> str:
                # strip combining diacritics/length
                nf = unicodedata.normalize("NFD", ph)
                base = "".join(ch for ch in nf if not unicodedata.combining(ch) and ch not in {"ː","ˑ"})
                return (base
                        .replace("tɕ", "tʃ").replace("tɕʰ", "tʃ").replace("tʂ", "tʃ")
                        .replace("dʑ", "dʒ")
                        .replace("ɕ", "ʃ").replace("ʂ", "ʃ").replace("ʐ", "ʒ")
                        .replace("ɹ", "r").replace("ɾ", "r")
                        .replace("x", "h").replace("y", "j")
                        .replace("ɴ", "n").replace("ɫ", "l")
                        .replace("ɒ", "ɑ").replace("ɤ", "ʌ")
                )
            phones = [norm(p) for p in raw if p]
            return phones
        except Exception as exc:
            logger.warning("segment phone recognition failed: %s", exc)
            return []


def recognize_audio_phones(audio: np.ndarray) -> List[str]:
    # Recognize phones on the whole utterance for robust alignment
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        pcm16 = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16)
        sf.write(tmp.name, pcm16, TARGET_SR, subtype="PCM_16")
        try:
            rec = read_recognizer()
            if hasattr(rec, "recognize"):
                out = rec.recognize(tmp.name)
            elif hasattr(rec, "predict"):
                out = rec.predict(tmp.name)
            else:
                out = ""
            raw = [p for p in (out or "").strip().split() if p]
            import unicodedata
            def norm(ph: str) -> str:
                nf = unicodedata.normalize("NFD", ph)
                base = "".join(ch for ch in nf if not unicodedata.combining(ch) and ch not in {"ː","ˑ"})
                return (base
                        .replace("tɕ", "tʃ").replace("tɕʰ", "tʃ").replace("tʂ", "tʃ")
                        .replace("dʑ", "dʒ")
                        .replace("ɕ", "ʃ").replace("ʂ", "ʃ").replace("ʐ", "ʒ")
                        .replace("ɹ", "r").replace("ɾ", "r")
                        .replace("x", "h").replace("y", "j")
                        .replace("ɴ", "n").replace("ɫ", "l")
                        .replace("ɒ", "ɑ").replace("ɤ", "ʌ")
                )
            return [norm(p) for p in raw if p]
        except Exception as exc:
            logger.warning("utterance phone recognition failed: %s", exc)
            return []

def align_and_score(expected_ipa: List[str], recognized_ipa: List[str]) -> tuple[List[float], List[tuple[Optional[str], Optional[str], Optional[float]]]]:
    # Needleman-Wunsch alignment with feature-based substitution cost
    gap_penalty = 0.9

    n = len(expected_ipa)
    m = len(recognized_ipa)
    if n == 0:
        return []

    # DP matrices
    score = np.zeros((n + 1, m + 1), dtype=float)
    ptr = np.zeros((n + 1, m + 1), dtype=int)
    # init
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] - gap_penalty
        ptr[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] - gap_penalty
        ptr[0, j] = 2
    # fill
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 1.0 - (1.0 - ipa_feature_distance(expected_ipa[i - 1], recognized_ipa[j - 1]))
            match_score = score[i - 1, j - 1] - sub_cost
            delete_score = score[i - 1, j] - gap_penalty
            insert_score = score[i, j - 1] - gap_penalty
            best = max(match_score, delete_score, insert_score)
            score[i, j] = best
            ptr[i, j] = 0 if best == match_score else (1 if best == delete_score else 2)
    # traceback to collect aligned pairs
    i, j = n, m
    aligned_scores: List[float] = []
    align_pairs: List[tuple[Optional[str], Optional[str], Optional[float]]] = []
    while i > 0 and j > 0:
        dir_ = ptr[i, j]
        if dir_ == 0:
            # match/substitute
            dist = ipa_feature_distance(expected_ipa[i - 1], recognized_ipa[j - 1])
            score_01 = 1.0 - dist
            aligned_scores.append(score_01)
            align_pairs.append((expected_ipa[i - 1], recognized_ipa[j - 1], score_01 * 100.0))
            i -= 1
            j -= 1
        elif dir_ == 1:
            # deletion: expected unmatched
            aligned_scores.append(0.0)
            align_pairs.append((expected_ipa[i - 1], "∅", 0.0))
            i -= 1
        else:
            # insertion: recognized extra, skip
            align_pairs.append(("∅", recognized_ipa[j - 1], None))
            j -= 1
    # if any expected remain, penalize
    while i > 0:
        aligned_scores.append(0.0)
        align_pairs.append((expected_ipa[i - 1], "∅", 0.0))
        i -= 1

    aligned_scores.reverse()
    align_pairs.reverse()
    # Map to 0-100
    scores_100 = [float(np.clip(s * 100.0, 0.0, 100.0)) for s in aligned_scores]
    return scores_100, align_pairs


def compute_scores_for_text(predicted_text: str, recognized_phones: List[str]) -> PronunciationResult:
    # Deprecated path; kept for compatibility but not used now
    return PronunciationResult(words=[], overall_score=0.0)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    language: str = Form("en"),
    expected_text: Optional[str] = Form(None),
    browser_transcript: Optional[str] = Form(None),
):
    if file.content_type is None or not (
        file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
    ):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")

    data = await file.read()
    audio = load_audio_to_mono16k(data)
    seconds = len(audio) / TARGET_SR
    if seconds > MAX_AUDIO_SECONDS:
        raise HTTPException(status_code=400, detail=f"Audio too long: {seconds:.1f}s > {MAX_AUDIO_SECONDS}s")
    logger.info("Analyze file='%s' content_type='%s' duration_s=%.2f", file.filename, file.content_type, seconds)

    # ASR using Whisper for predicted text and word timestamps
    predicted_text, asr_words = transcribe_with_words(audio)

    # If both expected_text and browser_transcript are provided and you want strict text-vs-text scoring,
    # compute purely textual phoneme alignment. This avoids model-label mismatches entirely.
    if expected_text and (browser_transcript or predicted_text):
        said_text_for_strict = browser_transcript or predicted_text
        result = score_text_vs_text(expected_text, said_text_for_strict)
        return AnalyzeResponse(pronunciation=result, predicted_text=predicted_text)

    # Otherwise fall back to acoustic alignment using phoneme posteriors
    post, labels = ctc_phone_posteriors(audio)
    total_audio_seconds = len(audio) / TARGET_SR

    words_out: List[WordPronunciation] = []

    # If browser transcript provided, align to ASR words and score matched words
    # Choose which text to score against: browser_transcript > expected_text > predicted_text
    said_words: List[str] = []
    if browser_transcript and browser_transcript.strip():
        said_words = _tokenize_words(browser_transcript)
    elif expected_text and expected_text.strip():
        said_words = _tokenize_words(expected_text)
    asr_word_texts = [str(w.get("word", "")).strip() for w in asr_words]
    asr_word_times = [(float(w.get("start", 0.0)), float(w.get("end", 0.0))) for w in asr_words]

    if said_words:
        pairs = _align_words_indices(said_words, [w.lower() for w in asr_word_texts])
        # Phonemize said words once
        phonemized_said = phonemize_words(" ".join(said_words))
        # Map original said index to its phones
        idx_to_phones = {i: split_diphthongs(phonemized_said[i]) if i < len(phonemized_said) else [] for i in range(len(said_words))}
        for i_idx, j_idx in pairs:
            said_w = said_words[i_idx]
            exp_ipas = idx_to_phones.get(i_idx, [])
            if not exp_ipas:
                continue
            # Time slice from ASR word
            if 0 <= j_idx < len(asr_word_times):
                start_s, end_s = asr_word_times[j_idx]
            else:
                start_s, end_s = 0.0, total_audio_seconds
            s_idx, e_idx = _time_to_frame_indices(start_s, end_s, total_audio_seconds, post.shape[0])
            post_slice = post[s_idx:e_idx] if e_idx > s_idx else post
            scores = ctc_align_scores(exp_ipas, post_slice, labels) if post_slice.size else []
            wp = WordPronunciation(
                word_text=said_w,
                phonemes=[PhonemeScore(ipa_label=p, phoneme_score=float(scores[k]) if k < len(scores) else 0.0) for k, p in enumerate(exp_ipas)],
                word_score=float(np.mean(scores)) if scores else 0.0,
            )
            words_out.append(wp)
    else:
        # Fallback: score all ASR words using their own timestamps
        predicted_tokens = _tokenize_words(predicted_text)
        phonemized_pred = phonemize_words(" ".join(predicted_tokens))
        for idx, tok in enumerate(predicted_tokens):
            exp_ipas = split_diphthongs(phonemized_pred[idx]) if idx < len(phonemized_pred) else []
            if not exp_ipas:
                continue
            # Find matching ASR occurrence index-wise
            # Map token occurrence to jth asr word with same lower text
            occurrences_before = sum(1 for t in predicted_tokens[:idx] if t.lower() == tok.lower())
            # collect indices where ASR words equal tok
            eq_indices = [j for j, w in enumerate(asr_word_texts) if w.lower() == tok.lower()]
            j_idx = eq_indices[occurrences_before] if occurrences_before < len(eq_indices) else -1
            if 0 <= j_idx < len(asr_word_times):
                start_s, end_s = asr_word_times[j_idx]
            else:
                start_s, end_s = 0.0, total_audio_seconds
            s_idx, e_idx = _time_to_frame_indices(start_s, end_s, total_audio_seconds, post.shape[0])
            post_slice = post[s_idx:e_idx] if e_idx > s_idx else post
            scores = ctc_align_scores(exp_ipas, post_slice, labels) if post_slice.size else []
            wp = WordPronunciation(
                word_text=tok,
                phonemes=[PhonemeScore(ipa_label=p, phoneme_score=float(scores[k]) if k < len(scores) else 0.0) for k, p in enumerate(exp_ipas)],
                word_score=float(np.mean(scores)) if scores else 0.0,
            )
            words_out.append(wp)

    # Log a couple of words
    for idx in range(min(2, len(words_out))):
        w = words_out[idx]
        def fmt_score(x: float) -> str:
            if x is None or math.isnan(x) or math.isinf(x):
                return "0"
            return str(int(max(0, min(100, x))))
        logger.info("Word %s phones: %s", w.word_text, ", ".join(f"/{p.ipa_label}/ {fmt_score(p.phoneme_score)}" for p in w.phonemes))

    # If all zeros (e.g., mismatch of IPA to label-set), try utterance-level alignment as fallback
    overall = float(np.mean([w.word_score for w in words_out])) if words_out else 0.0
    if words_out and all((w.word_score == 0.0 or math.isnan(w.word_score)) for w in words_out):
        # Build a single IPA sequence from selected text and align against the utterance posteriors
        score_text = browser_transcript or expected_text or predicted_text
        ipas_words = phonemize_words(score_text)
        expected_full = []
        for ipas in ipas_words:
            expected_full.extend(split_diphthongs(ipas))
        if expected_full:
            scores_full = ctc_align_scores(expected_full, post, labels)
            # Distribute back to words proportionally
            k = 0
            for w in words_out:
                n = len(w.phonemes)
                seg = scores_full[k:k+n]
                for i in range(n):
                    if i < len(seg):
                        w.phonemes[i].phoneme_score = float(seg[i])
                w.word_score = float(np.mean(seg)) if seg else 0.0
                k += n
            overall = float(np.mean([w.word_score for w in words_out])) if words_out else 0.0
    result = PronunciationResult(words=words_out, overall_score=overall)
    return AnalyzeResponse(pronunciation=result, predicted_text=predicted_text)


# Health
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
