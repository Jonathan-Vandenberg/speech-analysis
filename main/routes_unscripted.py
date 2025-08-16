from typing import Optional, List
import logging
import re
from collections import Counter

import numpy as np
import soundfile as sf
import io
import tempfile
import subprocess
import librosa
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import AnalyzeResponse, PronunciationResult, WordPronunciation, PhonemeScore, SpeechMetrics, PauseDetail, DiscourseMarker, FillerWordDetail, Repetition
from .utils_text import tokenize_words, normalize_ipa, phonemize_words_en, phonemes_from_audio
from .utils_align import align_and_score

router = APIRouter()
logger = logging.getLogger("speech_analyzer")


async def analyze_audio_pronunciation(pred_words: List[str], pred_ipas_words: List[List[str]], recognized_phonemes: List[str], predicted_text: str, audio: Optional[np.ndarray] = None, deep_analysis: bool = False) -> AnalyzeResponse:
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
    
    # Perform deep analysis if requested and audio is available
    metrics = None
    if deep_analysis and audio is not None:
        logger.debug("Performing deep speech fluency analysis in audio mode")
        metrics = analyze_speech_fluency(predicted_text, audio, sr=16000)
    
    return AnalyzeResponse(pronunciation=PronunciationResult(words=out_words, overall_score=overall), predicted_text=predicted_text, metrics=metrics)


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


def detect_pauses(audio: np.ndarray, sr: int = 16000, min_pause_duration: float = 1.0) -> List[PauseDetail]:
    """Detect pauses in speech (silence > 1 second)."""
    # Use librosa to detect non-silent intervals
    intervals = librosa.effects.split(audio, top_db=20, frame_length=2048, hop_length=512)
    
    pauses = []
    if len(intervals) > 1:
        for i in range(len(intervals) - 1):
            # Calculate pause between current interval end and next interval start
            current_end = intervals[i][1]
            next_start = intervals[i + 1][0]
            
            pause_frames = next_start - current_end
            pause_duration = pause_frames / sr
            
            if pause_duration >= min_pause_duration:
                # Convert frame indices to character indices (approximate)
                # This is rough - in real implementation you'd need word-level timestamps
                start_char = int(current_end * 0.05)  # Rough approximation
                end_char = start_char
                
                pauses.append(PauseDetail(
                    start_index=start_char,
                    end_index=end_char,
                    duration=round(pause_duration, 1)
                ))
    
    return pauses


def detect_filler_words(text: str, phonemes_list: List[str]) -> tuple[List[FillerWordDetail], int]:
    """Detect filler words based on phoneme patterns and context."""
    filler_patterns = {
        ('ʌ',): 'uh',
        ('ʌ', 'm'): 'um', 
        ('h', 'ə', 'm'): 'hmm',
        ('ə', 'm'): 'em',
        ('ə',): 'uh',
        ('m', 'h', 'm'): 'mhm'
    }
    
    filler_words = []
    text_words = text.lower().split()
    
    # Clear filler words that are almost always fillers
    definite_fillers = ['uh', 'um', 'hmm', 'er', 'ah', 'mm', 'erm']
    
    char_idx = 0
    for i, word in enumerate(text_words):
        clean_word = word.strip('.,!?')
        
        # Always mark definite fillers
        if clean_word in definite_fillers:
            filler_words.append(FillerWordDetail(
                text=word,
                start_index=char_idx,
                end_index=char_idx + len(word),
                phonemes=clean_word
            ))
        
        # Context-aware detection for "like"
        elif clean_word == 'like':
            is_filler = is_like_filler(text_words, i)
            if is_filler:
                filler_words.append(FillerWordDetail(
                    text=word,
                    start_index=char_idx,
                    end_index=char_idx + len(word),
                    phonemes=clean_word
                ))
        
        # Context-aware detection for "you know"
        elif clean_word == 'you' and i + 1 < len(text_words) and text_words[i + 1].strip('.,!?') == 'know':
            # "you know" as filler usually appears at sentence boundaries or with hesitation
            if is_you_know_filler(text_words, i):
                filler_words.append(FillerWordDetail(
                    text=f"{word} {text_words[i + 1]}",
                    start_index=char_idx,
                    end_index=char_idx + len(word) + 1 + len(text_words[i + 1]),
                    phonemes="you know"
                ))
        
        char_idx += len(word) + 1  # +1 for space
    
    return filler_words, len(filler_words)


def is_like_filler(words: List[str], like_index: int) -> bool:
    """Determine if 'like' is being used as a filler word based on context."""
    # Get surrounding context
    prev_word = words[like_index - 1].strip('.,!?').lower() if like_index > 0 else ""
    next_word = words[like_index + 1].strip('.,!?').lower() if like_index + 1 < len(words) else ""
    
    # "like" is likely a filler if:
    
    # 1. Followed by another filler or hesitation
    if next_word in ['uh', 'um', 'er', 'ah']:
        return True
    
    # 2. Used in quotative speech patterns: "I was like", "she was like"
    if prev_word in ['was', 'am', 'is', 'were']:
        return True
    
    # 3. Used as discourse marker: "like, you know", "like, totally"
    if next_word in ['you', 'totally', 'really', 'literally']:
        return True
    
    # 4. Repeated "like" (hesitation pattern)
    if (like_index > 0 and words[like_index - 1].strip('.,!?').lower() == 'like') or \
       (like_index + 1 < len(words) and words[like_index + 1].strip('.,!?').lower() == 'like'):
        return True
    
    # 5. At the beginning of utterances (discourse marker)
    if like_index == 0 or prev_word in ['.', '!', '?', ',']:
        return True
    
    # Otherwise, it's likely legitimate usage: "I like pizza", "it looks like rain"
    return False


def is_you_know_filler(words: List[str], you_index: int) -> bool:
    """Determine if 'you know' is being used as a filler phrase."""
    # Get surrounding context
    prev_word = words[you_index - 1].strip('.,!?').lower() if you_index > 0 else ""
    next_word = words[you_index + 2].strip('.,!?').lower() if you_index + 2 < len(words) else ""
    
    # "you know" is likely a filler if:
    
    # 1. At sentence boundaries (pause markers)
    if prev_word in ['.', '!', '?', ''] or next_word in ['.', '!', '?', '']:
        return True
    
    # 2. Surrounded by commas (parenthetical)
    if prev_word.endswith(',') or next_word.startswith(','):
        return True
    
    # 3. Followed by other fillers
    if next_word in ['like', 'uh', 'um']:
        return True
    
    # Otherwise might be legitimate: "Do you know the answer?"
    return False


def detect_repetitions(text: str) -> List[Repetition]:
    """Detect repeated words or short phrases."""
    words = text.lower().split()
    repetitions = []
    
    # Look for consecutive repeated words
    i = 0
    while i < len(words) - 1:
        current_word = words[i].strip('.,!?')
        count = 1
        j = i + 1
        
        # Count consecutive repetitions
        while j < len(words) and words[j].strip('.,!?') == current_word:
            count += 1
            j += 1
            
        if count > 1:  # Found repetition
            # Calculate character indices
            start_idx = sum(len(words[k]) + 1 for k in range(i))
            end_idx = start_idx + sum(len(words[k]) + 1 for k in range(i, j)) - 1
            
            repetitions.append(Repetition(
                text=current_word,
                start_index=start_idx,
                end_index=end_idx,
                repeat_count=count
            ))
            i = j
        else:
            i += 1
            
    return repetitions


def calculate_speech_rate(text: str, audio: np.ndarray, sr: int = 16000) -> tuple[float, List[float]]:
    """Calculate overall speech rate and speech rate over time."""
    # Remove silence to get actual speaking time
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
    duration_seconds = len(audio_trimmed) / sr
    
    # Get detected filler words using context-aware detection
    phonemes = []  # Not used for speech rate, but required by function signature
    filler_details, _ = detect_filler_words(text, phonemes)
    
    # Create set of detected filler word positions for efficient lookup
    filler_positions = set()
    for filler in filler_details:
        # Calculate word positions based on character indices
        words_before = text[:filler.start_index].split()
        filler_positions.add(len(words_before))
    
    # Count words (excluding contextually-detected filler words)
    words = text.split()
    content_words = []
    for i, word in enumerate(words):
        if i not in filler_positions:
            content_words.append(word)
    
    if duration_seconds == 0:
        return 0.0, []
    
    # Words per minute
    overall_rate = (len(content_words) / duration_seconds) * 60
    
    # Calculate rate over time (in chunks)
    chunk_duration = 3.0  # 3-second chunks
    chunks = int(duration_seconds / chunk_duration)
    rate_over_time = []
    
    if chunks > 0:
        words_per_chunk = len(content_words) / chunks
        for i in range(chunks):
            # Simple approximation - in reality you'd need word-level timestamps
            chunk_rate = (words_per_chunk / chunk_duration) * 60
            rate_over_time.append(round(chunk_rate))
    
    return round(overall_rate), rate_over_time


def detect_discourse_markers(text: str) -> List[DiscourseMarker]:
    """Detect discourse markers (connecting words/phrases)."""
    markers_map = {
        'and': 'adding information',
        'but': 'contrasting',
        'however': 'contrasting', 
        'therefore': 'concluding',
        'so': 'concluding',
        'first': 'sequencing',
        'then': 'sequencing',
        'next': 'sequencing',
        'finally': 'concluding',
        'also': 'adding information',
        'furthermore': 'adding information',
        'moreover': 'adding information'
    }
    
    discourse_markers = []
    words = text.split()
    char_idx = 0
    
    for word in words:
        clean_word = word.lower().strip('.,!?')
        if clean_word in markers_map:
            discourse_markers.append(DiscourseMarker(
                text=clean_word,
                start_index=char_idx,
                end_index=char_idx + len(clean_word),
                description=markers_map[clean_word]
            ))
        char_idx += len(word) + 1
    
    return discourse_markers


def analyze_speech_fluency(text: str, audio: np.ndarray, sr: int = 16000) -> SpeechMetrics:
    """Perform comprehensive speech fluency analysis."""
    # Get phonemes for filler word detection
    phonemes = phonemes_from_audio(audio)
    
    # Detect pauses
    pauses = detect_pauses(audio, sr)
    
    # Detect filler words
    filler_details, filler_count = detect_filler_words(text, phonemes)
    
    # Detect repetitions
    repetitions = detect_repetitions(text)
    
    # Calculate speech rate
    speech_rate, rate_over_time = calculate_speech_rate(text, audio, sr)
    
    # Detect discourse markers
    discourse_markers = detect_discourse_markers(text)
    
    # Calculate filler words per minute
    audio_duration_min = len(audio) / sr / 60
    filler_words_per_min = filler_count / audio_duration_min if audio_duration_min > 0 else 0
    
    return SpeechMetrics(
        speech_rate=speech_rate,
        speech_rate_over_time=rate_over_time,
        pauses=len(pauses),
        filler_words=filler_count,
        discourse_markers=discourse_markers,
        filler_words_per_min=round(filler_words_per_min, 1),
        pause_details=pauses,
        repetitions=repetitions,
        filler_words_details=filler_details
    )


@router.post("/unscripted", response_model=AnalyzeResponse)
async def unscripted(
    file: UploadFile = File(...),
    browser_transcript: Optional[str] = Form(None),
    use_audio: bool = Form(False),
    deep_analysis: bool = Form(False),
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
        return await analyze_audio_pronunciation(pred_words, pred_ipas_words, recognized_phonemes, predicted_text, audio, deep_analysis)
        
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
    
    return AnalyzeResponse(
        pronunciation=PronunciationResult(words=out_words, overall_score=overall), 
        predicted_text=predicted_text,
        metrics=None  # Deep analysis is only available with audio mode
    )


