from typing import Optional, List, Dict, Any
import logging
import re
import json
import os
from collections import Counter

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

import numpy as np
import soundfile as sf
import io
import tempfile
import subprocess
import librosa
from fastapi import APIRouter, File, Form, HTTPException, UploadFile


def ensure_minimum_score(score: float) -> float:
    """Ensure no score is below 10% - replace harsh zeros with encouraging 10-20%."""
    if score <= 5.0:
        from .utils_align import random_low_score
        return random_low_score()
    return score

from .schemas import (
    AnalyzeResponse, PronunciationResult, WordPronunciation, PhonemeScore, 
    SpeechMetrics, PauseDetail, DiscourseMarker, FillerWordDetail, Repetition,
    GrammarCorrection, GrammarDifference, RelevanceAnalysis, IELTSScore
)
from .utils_text import tokenize_words, normalize_ipa, normalize_ipa_preserve_diphthongs, phonemize_words_en, phonemes_from_audio
from .utils_align import align_and_score, random_low_score

router = APIRouter()
logger = logging.getLogger("speech_analyzer")

# OpenAI for grammar analysis
try:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        openai_client = OpenAI(api_key=api_key)
        OPENAI_AVAILABLE = True
    else:
        logger.warning("OPENAI_API_KEY not found in environment variables")
        openai_client = None
        OPENAI_AVAILABLE = False
except ImportError:
    logger.warning("OpenAI library not installed")
    OPENAI_AVAILABLE = False
    openai_client = None
except Exception as e:
    logger.error(f"Error initializing OpenAI client: {e}")
    OPENAI_AVAILABLE = False
    openai_client = None

# Global Whisper model for sharing across requests (MAJOR PERFORMANCE IMPROVEMENT)
_whisper_model: Optional[object] = None

def _lazy_whisper_model():
    """Lazy load Whisper model globally to avoid reloading per request."""
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            import os
            model_id = os.getenv("WHISPER_MODEL", "small")
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "auto")
            
            logger.info(f"🚀 Loading Whisper model '{model_id}' (compute_type: {compute_type}) - this happens once per server startup")
            _whisper_model = WhisperModel(model_id, device="auto", compute_type=compute_type)
            logger.info("✅ Whisper model loaded successfully and cached for all future requests")
        except ImportError:
            raise ImportError("faster-whisper not installed. Install with: pip install faster-whisper")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    return _whisper_model

# Concurrency controls for better performance
from asyncio import Semaphore
audio_processing_semaphore = Semaphore(5)  # Max 5 concurrent audio processing tasks
grammar_analysis_semaphore = Semaphore(8)  # Max 8 concurrent grammar analysis tasks


async def analyze_audio_pronunciation(pred_words: List[str], pred_ipas_words: List[List[str]], recognized_phonemes: List[str], predicted_text: str, audio: Optional[np.ndarray] = None, deep_analysis: bool = False) -> AnalyzeResponse:
    """Analyze pronunciation using audio-extracted phonemes vs predicted text phonemes with Oxford IPA."""
    # Starting Oxford IPA pronunciation analysis
    
    # If we have too few recognized phonemes, use more lenient scoring
    total_expected_phonemes = sum(len(normalize_ipa_preserve_diphthongs(word_ipas)) for word_ipas in pred_ipas_words)
    phoneme_ratio = len(recognized_phonemes) / max(1, total_expected_phonemes)
    
    # Calculate phoneme ratio for quality assessment
    
    # Flatten expected phonemes with word boundaries
    expected_phonemes_flat = []
    word_boundaries = []  # Track which phoneme belongs to which word
    
    for word_idx, word_ipas in enumerate(pred_ipas_words):
        # Use diphthong-preserving normalization for consistency with fixed recognized phonemes
        normalized_ipas = normalize_ipa_preserve_diphthongs(word_ipas)
        for phoneme in normalized_ipas:
            expected_phonemes_flat.append(phoneme)
            word_boundaries.append(word_idx)
    
    if not expected_phonemes_flat:
        raise HTTPException(status_code=500, detail="Could not generate expected phonemes from transcription.")
    
    # Check if we have reasonable phoneme data for alignment
    if len(recognized_phonemes) < 3 or phoneme_ratio < 0.3:
        # Poor phoneme extraction quality, using fallback scoring
        # Use fallback scoring based on successful Whisper transcription
        word_phoneme_data = {}
        for word_idx, word_ipas in enumerate(pred_ipas_words):
            normalized_ipas = normalize_ipa_preserve_diphthongs(word_ipas)
            # Give excellent scores since Whisper successfully transcribed the speech
            word_phoneme_data[word_idx] = [(ph, 92.0) for ph in normalized_ipas]
    else:
        # Align recognized phonemes to expected phonemes
        scores, pairs = align_and_score(expected_phonemes_flat, recognized_phonemes)
        
        # Group results back into words
        word_phoneme_data = {}  # word_idx -> list of (phoneme, score)
        expected_phoneme_idx = 0  # Track position in original expected phonemes
        
        for i, (expected_ph, recognized_ph, score) in enumerate(pairs):
            if expected_ph not in (None, "∅"):
                # This is an expected phoneme, find which word it belongs to using the correct index
                if expected_phoneme_idx < len(word_boundaries):
                    word_idx = word_boundaries[expected_phoneme_idx]
                    if word_idx not in word_phoneme_data:
                        word_phoneme_data[word_idx] = []
                    
                    if score is not None:
                        word_phoneme_data[word_idx].append((expected_ph, float(score)))
                    else:
                        # Deletion (expected but not said) - use encouraging random score
                        word_phoneme_data[word_idx].append((expected_ph, random_low_score()))
                
                # Increment expected phoneme index when we process an expected phoneme
                expected_phoneme_idx += 1
            
            elif recognized_ph not in (None, "∅"):
                # Insertion (said but not expected) - assign to current word context
                # Use the last valid expected phoneme position to determine word context
                current_word_idx = 0
                if expected_phoneme_idx > 0 and (expected_phoneme_idx - 1) < len(word_boundaries):
                    current_word_idx = word_boundaries[expected_phoneme_idx - 1]
                elif expected_phoneme_idx < len(word_boundaries):
                    current_word_idx = word_boundaries[expected_phoneme_idx]
                
                if current_word_idx not in word_phoneme_data:
                    word_phoneme_data[current_word_idx] = []
                # Insertion penalty - use encouraging random score instead of harsh 0
                word_phoneme_data[current_word_idx].append((recognized_ph, random_low_score()))
    
    # Build output words using normalized words (simpler approach)
    out_words: List[WordPronunciation] = []
    for word_idx, word_text in enumerate(pred_words):
        if word_idx in word_phoneme_data:
            phoneme_data = word_phoneme_data[word_idx]
            phonemes = [PhonemeScore(ipa_label=ph, phoneme_score=score) for ph, score in phoneme_data]
            word_score = float(np.mean([score for _, score in phoneme_data])) if phoneme_data else random_low_score()
        else:
            # Word completely missing
            expected_ipas = normalize_ipa_preserve_diphthongs(pred_ipas_words[word_idx]) if word_idx < len(pred_ipas_words) else []
            phonemes = [PhonemeScore(ipa_label=ph, phoneme_score=random_low_score()) for ph in expected_ipas]
            word_score = random_low_score()
        
        out_words.append(WordPronunciation(word_text=word_text, phonemes=phonemes, word_score=word_score))
    
    overall = float(np.mean([w.word_score for w in out_words])) if out_words else random_low_score()
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
    """Transcribe audio using shared Whisper model (MUCH faster - no model loading per request)."""
    import os
    model = _lazy_whisper_model()  # Use shared model instead of loading each time!
    beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    
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
    # Balanced filler phoneme patterns - multi-phoneme sequences and some standalone vowels
    filler_patterns = {        
        # "Um" sounds (most reliable - vowel + m combination)
        ('ʌ', 'm'): 'um',
        ('ə', 'm'): 'um', 
        ('ʊ', 'm'): 'um',
        ('a', 'm'): 'um',
        
        # "Hmm" sounds (consonant combinations)
        ('h', 'ə', 'm'): 'hmm',
        ('h', 'm'): 'hmm',
        ('m', 'm'): 'hmm',
        
        # "Er" sounds - but only longer patterns to avoid "water", "her" issues
        ('ɚ', 'r'): 'er',   # More specific pattern
        ('ə', 'r'): 'er',   # Schwa + r
        
        # Single vowel patterns (for very clear standalone hesitations)
        ('ʌ',): 'uh',   # Most common "uh" sound 
        ('ə',): 'uh',   # Schwa "uh" 
        ('a',): 'ah',   # "Ah" hesitation
    }
    
    # Note: Single phoneme patterns are now included in filler_patterns above
    
    filler_words = []
    text_words = text.lower().split()
    
    # First: Detect filler sounds directly from phonemes (more sensitive)
    logger.debug(f"Starting phoneme-based filler detection with {len(phonemes_list)} phonemes")
    logger.debug(f"Available patterns: {list(filler_patterns.keys())}")
    phoneme_fillers = detect_phoneme_fillers(phonemes_list, filler_patterns)
    logger.debug(f"Phoneme-based detection found {len(phoneme_fillers)} fillers")
    filler_words.extend(phoneme_fillers)
    
    # Second: Text-based detection for words Whisper did transcribe
    definite_fillers = ['uh', 'um', 'hmm', 'er', 'ah', 'mm', 'erm', 'eh', 'oh']
    
    char_idx = 0
    for i, word in enumerate(text_words):
        clean_word = word.strip('.,!?')
        
        # Always mark definite fillers in text
        if clean_word in definite_fillers:
            # Avoid duplicates from phoneme detection
            if not any(f.text.lower() == clean_word and abs(f.start_index - char_idx) < 10 for f in filler_words):
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
            if is_you_know_filler(text_words, i):
                filler_words.append(FillerWordDetail(
                    text=f"{word} {text_words[i + 1]}",
                    start_index=char_idx,
                    end_index=char_idx + len(word) + 1 + len(text_words[i + 1]),
                    phonemes="you know"
                ))
        
        char_idx += len(word) + 1  # +1 for space
    
    # Sort by position and remove near duplicates
    filler_words.sort(key=lambda x: x.start_index)
    unique_fillers = []
    for filler in filler_words:
        if not any(abs(filler.start_index - existing.start_index) < 5 for existing in unique_fillers):
            unique_fillers.append(filler)
    
    return unique_fillers, len(unique_fillers)


def detect_phoneme_fillers(phonemes: List[str], patterns: dict) -> List[FillerWordDetail]:
    """Detect filler words directly from phoneme sequences."""
    detected_fillers = []
    
    if not phonemes:
        logger.debug("No phonemes provided for filler detection")
        return detected_fillers
    
    logger.debug(f"Scanning {len(phonemes)} phonemes for filler patterns")
    
    i = 0
    while i < len(phonemes):
        # Check for multi-phoneme patterns first (longest first)
        max_pattern_len = min(3, len(phonemes) - i)
        found_pattern = False
        
        for pattern_len in range(max_pattern_len, 0, -1):  # Start from longest patterns down to 1
            if i + pattern_len <= len(phonemes):
                phoneme_seq = tuple(phonemes[i:i + pattern_len])
                
                if phoneme_seq in patterns:
                    filler_text = patterns[phoneme_seq]
                    estimated_char_pos = i * 3
                    
                    # Additional check: make sure this isn't part of a longer word sequence
                    # For single phonemes, use stricter isolation checking
                    is_standalone = True
                    if pattern_len == 1:
                        is_standalone = is_phoneme_isolated(phonemes, i)
                        logger.debug(f"Single phoneme '{phonemes[i]}' isolation check: {is_standalone}")
                    else:
                        is_standalone = is_likely_standalone_filler(phonemes, i, pattern_len)
                        logger.debug(f"Multi-phoneme pattern {phoneme_seq} standalone check: {is_standalone}")
                    
                    if is_standalone:
                        logger.debug(f"Found filler pattern {phoneme_seq} -> '{filler_text}' at position {i}")
                        
                        detected_fillers.append(FillerWordDetail(
                            text=filler_text,
                            start_index=estimated_char_pos,
                            end_index=estimated_char_pos + len(filler_text),
                            phonemes='/'.join(phoneme_seq)
                        ))
                        
                        i += pattern_len
                        found_pattern = True
                        break
                    else:
                        logger.debug(f"Pattern {phoneme_seq} found but not considered standalone - skipping")
        
        # No need for separate isolated pattern checking since single patterns are now in main patterns
        
        if not found_pattern:
            i += 1
    
    logger.debug(f"Detected {len(detected_fillers)} filler words from phonemes")
    return detected_fillers


def is_likely_standalone_filler(phonemes: List[str], start_index: int, pattern_length: int) -> bool:
    """Check if a multi-phoneme pattern is likely a standalone filler rather than part of a word."""
    
    pattern_phonemes = phonemes[start_index:start_index + pattern_length]
    
    # For "um" patterns (vowel + m), be very lenient - they're highly reliable
    if len(pattern_phonemes) == 2 and pattern_phonemes[1] == 'm':
        # "vowel + m" combinations are almost always fillers
        logger.debug(f"Found vowel+m pattern {pattern_phonemes} - highly likely to be filler")
        return True
    
    # For "hmm" patterns, also be lenient
    if 'h' in pattern_phonemes and 'm' in pattern_phonemes:
        logger.debug(f"Found h+m pattern {pattern_phonemes} - likely filler")
        return True
    
    # For "er" patterns, be more careful but still allow them
    if len(pattern_phonemes) == 2 and pattern_phonemes[1] == 'r':
        # Look at wider context to see if this might be part of a word like "water"
        context_window = 3
        before_start = max(0, start_index - context_window)
        after_end = min(len(phonemes), start_index + pattern_length + context_window)
        
        # Check if surrounded by many phonemes (indicating it's part of a longer word)
        phonemes_before = phonemes[before_start:start_index]
        phonemes_after = phonemes[start_index + pattern_length:after_end]
        
        # If there are 3+ phonemes on both sides, probably part of a word
        if len(phonemes_before) >= 3 and len(phonemes_after) >= 3:
            logger.debug(f"Er pattern {pattern_phonemes} surrounded by many phonemes - might be part of word")
            return False
        
        logger.debug(f"Er pattern {pattern_phonemes} appears isolated enough - likely filler")
        return True
    
    # For other patterns, be moderately permissive
    return True


def is_phoneme_isolated(phonemes: List[str], index: int) -> bool:
    """Check if a phoneme appears isolated (likely to be a filler).
    
    Balanced approach - detects genuine fillers while avoiding parts of words.
    """
    phoneme = phonemes[index]
    
    # For consonants like 'm', be more permissive as they're often standalone fillers
    if phoneme == 'm':
        # Look for gaps or pauses around the 'm' sound
        prev_phoneme = phonemes[index - 1] if index > 0 else None
        next_phoneme = phonemes[index + 1] if index + 1 < len(phonemes) else None
        
        # If it's at boundaries, likely isolated
        if index == 0 or index == len(phonemes) - 1:
            return True
        
        # If surrounded by vowels without consonants, might be part of a word
        vowels = {'a', 'e', 'i', 'o', 'u', 'ə', 'ɪ', 'ɛ', 'æ', 'ɑ', 'ɔ', 'ʌ', 'ʊ', 'ɜ', 'ɐ'}
        if prev_phoneme in vowels and next_phoneme in vowels:
            # But 'VmV' pattern could still be a filler, check context
            return len(phonemes) > 10  # Only in longer sequences
        return True
    
    # For vowels (ʌ, ə), use more nuanced detection
    if phoneme in {'ʌ', 'ə'}:
        # These are the most common hesitation sounds
        
        # Look at immediate neighbors
        prev_phoneme = phonemes[index - 1] if index > 0 else None
        next_phoneme = phonemes[index + 1] if index + 1 < len(phonemes) else None
        
        # At beginning or end of sequence - likely isolated
        if index == 0 or index == len(phonemes) - 1:
            return True
        
        # Check for patterns that suggest it's NOT part of a word
        consonants = {'t', 'd', 'k', 'g', 'p', 'b', 'f', 'v', 's', 'z', 'ʃ', 'ʒ', 'tʃ', 'dʒ', 'n', 'l', 'r', 'w', 'j', 'h'}
        
        # If preceded by a stop consonant and followed by another consonant -> likely hesitation
        stop_consonants = {'t', 'd', 'k', 'g', 'p', 'b'}
        if prev_phoneme in stop_consonants and next_phoneme in consonants:
            logger.debug(f"Vowel '{phoneme}' between stop consonant and consonant - likely hesitation")
            return True
        
        # If surrounded by pauses or at word boundaries
        if prev_phoneme in consonants and next_phoneme in consonants:
            # Check if this looks like part of a consonant cluster vs isolated
            # Look at wider context
            context_before = phonemes[max(0, index-2):index]
            context_after = phonemes[index+1:min(len(phonemes), index+3)]
            
            # If there aren't many phonemes clustered around, likely isolated
            total_context = len(context_before) + len(context_after)
            if total_context < 3:
                logger.debug(f"Vowel '{phoneme}' has minimal context - likely isolated")
                return True
        
        # If it's in a long sequence but appears to have gaps around it
        if len(phonemes) > 8:
            # Look for evidence of isolation in longer sequences
            context_window = 3
            context_phonemes = phonemes[max(0, index-context_window):min(len(phonemes), index+context_window+1)]
            # If the context window isn't completely full, suggests gaps/pauses
            if len(context_phonemes) < (context_window * 2 + 1):
                logger.debug(f"Vowel '{phoneme}' in long sequence with apparent gaps")
                return True
    
    logger.debug(f"Phoneme '{phoneme}' not considered isolated")
    return False


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
    
    # Calculate rate over time using audio analysis for more realistic fluctuations
    chunk_duration = 2.0  # 2-second chunks for better granularity
    total_audio_duration = len(audio) / sr
    chunks = max(1, int(total_audio_duration / chunk_duration))
    rate_over_time = []
    
    # Detect speaking activity in chunks using energy analysis
    for i in range(chunks):
        start_sample = int(i * chunk_duration * sr)
        end_sample = int(min((i + 1) * chunk_duration * sr, len(audio)))
        
        if end_sample > start_sample:
            chunk_audio = audio[start_sample:end_sample]
            
            # Calculate energy/activity in this chunk
            # Remove silence and measure actual speaking time
            try:
                chunk_trimmed, _ = librosa.effects.trim(chunk_audio, top_db=20)
                chunk_speaking_duration = len(chunk_trimmed) / sr
                
                # Estimate words in this chunk based on activity level
                if chunk_speaking_duration > 0:
                    # Activity ratio: how much of this chunk contains speech
                    activity_ratio = chunk_speaking_duration / chunk_duration
                    
                    # Base rate with some variation based on energy
                    energy = float(np.mean(np.abs(chunk_trimmed))) if len(chunk_trimmed) > 0 else 0
                    energy_factor = min(1.5, max(0.5, energy * 1000))  # Scale energy for variation
                    
                    # Estimate chunk rate with realistic variation
                    estimated_words_in_chunk = (len(content_words) / chunks) * activity_ratio * energy_factor
                    chunk_rate = (estimated_words_in_chunk / chunk_duration) * 60
                    
                    # Add some realistic variation based on chunk position
                    position_factor = 1.0 + 0.1 * np.sin(i * 0.5)  # Small sinusoidal variation
                    chunk_rate *= position_factor
                    
                    rate_over_time.append(max(0, round(chunk_rate)))
                else:
                    # No speech detected in this chunk
                    rate_over_time.append(0)
            except Exception:
                # Fallback to average rate if trimming fails
                avg_rate = overall_rate
                variation = 0.8 + 0.4 * (i % 3) / 2  # Simple variation pattern
                rate_over_time.append(round(avg_rate * variation))
        else:
            rate_over_time.append(0)
    
    # Ensure we have at least one data point
    if not rate_over_time:
        rate_over_time = [round(overall_rate)]
    
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


async def perform_comprehensive_ielts_analysis(text: str, question_text: Optional[str] = None, context: Optional[str] = None) -> tuple[Optional[GrammarCorrection], Optional[RelevanceAnalysis]]:
    """Perform comprehensive IELTS analysis using a single OpenAI call."""
    if not OPENAI_AVAILABLE or not openai_client:
        logger.warning("OpenAI not available for IELTS analysis")
        return None, None
    
    try:
        # Enhanced prompt that includes both grammar and relevance analysis
        prompt = f"""
        Analyze this SPOKEN text response according to IELTS assessment criteria. This is transcribed speech, so focus on actual grammatical errors, not punctuation or minor speech patterns.
        
        {f'The question being answered is: "{question_text}"' if question_text else ''}
        {f'Context: "{context}"' if context else ''}
        
        Original spoken text:
        {text}
        
        Please provide comprehensive analysis including:
        
        1. Grammar Correction: Correct ONLY obvious grammatical errors in the speech (wrong verb forms, missing articles, incorrect word order). Do NOT add punctuation or change natural speech patterns. Focus on errors that affect meaning or sound unnatural in spoken English.
        
        2. IELTS Lexical Resource Analysis: Evaluate the vocabulary, word choice, and language use according to IELTS criteria.
        
        3. Strengths: Highlight good points in the response that would earn marks in IELTS.
        
        4. Areas for Improvement: Suggest specific ways to enhance vocabulary and expression to improve IELTS score.
        
        5. Approximate Band Score for Lexical Resource (scale of 1-9): Provide an estimated band score just for vocabulary/lexical resource.
        
        6. Grammar Score (scale of 1-9): Provide an estimated band score for grammatical range and accuracy.
        
        7. Model Answers: Provide short, grammatically correct, and coherent example answers for IELTS speaking bands 4 through 9. For each band answer, include detailed markup to highlight specific language features that make it appropriate for that band level:
           - Use format: <mark type="feature_type" class="color_class" explanation="detailed explanation of why this feature earns marks at this band level">highlighted text</mark>
           - Feature types and their color classes:
             * basic_vocab (class="vocab-basic"): Simple, everyday vocabulary - blue
             * range_vocab (class="vocab-range"): Good range of vocabulary with some flexibility - green  
             * advanced_vocab (class="vocab-advanced"): Sophisticated, precise vocabulary - purple
             * expert_vocab (class="vocab-expert"): Natural, sophisticated vocabulary with subtle meanings - dark purple
             * simple_grammar (class="grammar-simple"): Basic sentence structures, simple tenses - light blue
             * complex_grammar (class="grammar-complex"): Complex sentences, variety of structures - orange
             * advanced_grammar (class="grammar-advanced"): Wide range of structures, natural and flexible - red
             * collocation (class="collocation"): Natural word combinations - teal
             * idiom (class="idiom"): Idiomatic expressions - pink
             * discourse_marker (class="discourse"): Linking words and phrases - yellow
             * precision (class="precision"): Precise and nuanced language use - indigo
           - Each model answer should have 5-8 marked features with different color classes
           - Show clear progression from basic (band 4) to sophisticated (band 9) language
           - Include a variety of feature types appropriate for each band level
           - ALWAYS include the class attribute for proper color coding
        
        8. Task Achievement Analysis: How well does the response address the question?
        
        9. Relevance Score (0-100): Overall relevance to the question asked.
        
        10. Key Points Covered: What important aspects were addressed well.
        
        11. Missing Points: What important aspects were missing or inadequately addressed.
        
        Format your response as a JSON object with keys: correctedText, lexicalAnalysis, strengths, improvements, lexicalBandScore, grammarScore, modelAnswers, relevanceScore, relevanceExplanation, keyPointsCovered, missingPoints.
        """

        # Single OpenAI API call for all analysis
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert IELTS examiner with deep knowledge of assessment criteria. Your task is to provide comprehensive analysis of written responses covering grammar, lexical resource, and task achievement. Your feedback should be constructive and specific."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        # Parse the AI response as JSON
        ai_response = json.loads(response.choices[0].message.content or '{}')
        corrected_text = ai_response.get('correctedText', '')

        # Generate differences between original and corrected text
        differences = find_differences(text, corrected_text)

        # Create a tagged version of the original text with highlighted errors
        tagged_text = create_tagged_text(text, differences)

        # Ensure strengths and improvements are arrays
        strengths = ai_response.get('strengths', [])
        if isinstance(strengths, str):
            strengths = [strengths]  # Convert string to list
        
        improvements = ai_response.get('improvements', [])
        if isinstance(improvements, str):
            improvements = [improvements]  # Convert string to list

        # Create grammar correction object
        grammar_correction = GrammarCorrection(
            original_text=text,
            corrected_text=corrected_text,
            differences=differences,
            taggedText=tagged_text,
            lexical_analysis=ai_response.get('lexicalAnalysis', ''),
            strengths=strengths,
            improvements=improvements,
            lexical_band_score=ai_response.get('lexicalBandScore', 0),
            grammar_score=ai_response.get('grammarScore', 0),
            modelAnswers=ai_response.get('modelAnswers', {})
        )

        # Create relevance analysis object if question provided
        relevance_analysis = None
        if question_text:
            # Ensure key_points_covered and missing_points are arrays
            key_points_covered = ai_response.get('keyPointsCovered', [])
            if isinstance(key_points_covered, str):
                key_points_covered = [key_points_covered]  # Convert string to list
            
            missing_points = ai_response.get('missingPoints', [])
            if isinstance(missing_points, str):
                missing_points = [missing_points]  # Convert string to list
                
            relevance_analysis = RelevanceAnalysis(
                relevance_score=ai_response.get('relevanceScore', 50),
                explanation=ai_response.get('relevanceExplanation', "Analysis unavailable"),
                key_points_covered=key_points_covered,
                missing_points=missing_points
            )

        return grammar_correction, relevance_analysis
    except Exception as e:
        logger.error(f"Error in comprehensive IELTS analysis: {e}")
        return None, None


def calculate_ielts_score(grammar_analysis: Optional[GrammarCorrection], pronunciation_result: PronunciationResult, speech_metrics: Optional[SpeechMetrics], relevance_analysis: Optional[RelevanceAnalysis]) -> IELTSScore:
    """Calculate IELTS score based on all factors."""
    # Calculate individual component scores
    fluency_score = calculate_fluency_score(speech_metrics) if speech_metrics else 5
    lexical_score = grammar_analysis.lexical_band_score if grammar_analysis else 5
    grammatical_score = grammar_analysis.grammar_score if grammar_analysis else 5
    pronunciation_score = min(9, max(1, pronunciation_result.overall_score / 11.11))

    # Calculate overall band score (average of the four criteria)
    overall_band = round((fluency_score + lexical_score + grammatical_score + pronunciation_score) / 4 * 2) / 2

    return IELTSScore(
        overall_band=min(9, max(1, overall_band)),
        fluency_coherence=min(9, max(1, fluency_score)),
        lexical_resource=min(9, max(1, lexical_score)),
        grammatical_range=min(9, max(1, grammatical_score)),
        pronunciation=min(9, max(1, pronunciation_score)),
        explanation=f"Overall band calculated from Fluency & Coherence ({fluency_score}), Lexical Resource ({lexical_score}), Grammatical Range & Accuracy ({grammatical_score}), and Pronunciation ({pronunciation_score:.1f})"
    )


def calculate_fluency_score(metrics: Optional[SpeechMetrics]) -> float:
    """Calculate fluency score based on speech metrics."""
    if not metrics:
        return 5
    
    # Calculate fluency based on speech rate, pauses, and filler words
    speech_rate = metrics.speech_rate or 0
    filler_words = metrics.filler_words or 0
    pauses = metrics.pauses or 0
    
    score = 5  # Start with middle band
    
    # Adjust based on speech rate (ideal range: 150-200 WPM)
    if 150 <= speech_rate <= 200:
        score += 1
    elif speech_rate < 100 or speech_rate > 250:
        score -= 1
    
    # Adjust based on filler words (penalize excessive fillers)
    if filler_words <= 2:
        score += 0.5
    elif filler_words > 5:
        score -= 1
    
    # Adjust based on pauses (penalize excessive hesitation)
    if pauses <= 1:
        score += 0.5
    elif pauses > 3:
        score -= 0.5
    
    return min(9, max(1, round(score)))


def find_differences(original: str, corrected: str) -> List[GrammarDifference]:
    """Find differences between original and corrected text using proper sequence alignment."""
    from difflib import SequenceMatcher
    
    original_words = original.split()
    corrected_words = corrected.split()
    
    differences: List[GrammarDifference] = []
    
    # Use SequenceMatcher for proper alignment that handles insertions/deletions correctly
    matcher = SequenceMatcher(None, original_words, corrected_words)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match, no differences
            continue
        elif tag == 'delete':
            # Words were deleted from original
            for k in range(i1, i2):
                differences.append(GrammarDifference(
                    type='deletion',
                    original=original_words[k],
                    corrected=None,
                    position=k
                ))
        elif tag == 'insert':
            # Words were inserted in corrected version
            for k in range(j1, j2):
                differences.append(GrammarDifference(
                    type='addition',
                    original=None,
                    corrected=corrected_words[k],
                    position=i1  # Insert at the position in original where change occurs
                ))
        elif tag == 'replace':
            # Words were substituted
            orig_span = original_words[i1:i2]
            corr_span = corrected_words[j1:j2]
            
            # Handle different lengths in replacement
            min_len = min(len(orig_span), len(corr_span))
            
            # Handle 1:1 substitutions first
            for k in range(min_len):
                if orig_span[k].lower() != corr_span[k].lower():
                    differences.append(GrammarDifference(
                        type='substitution',
                        original=orig_span[k],
                        corrected=corr_span[k],
                        position=i1 + k
                    ))
            
            # Handle extra deletions if original span is longer
            for k in range(min_len, len(orig_span)):
                differences.append(GrammarDifference(
                    type='deletion',
                    original=orig_span[k],
                    corrected=None,
                    position=i1 + k
                ))
            
            # Handle extra insertions if corrected span is longer
            for k in range(min_len, len(corr_span)):
                differences.append(GrammarDifference(
                    type='addition',
                    original=None,
                    corrected=corr_span[k],
                    position=i1 + min_len  # Insert after the last substituted word
                ))
    
    return differences


def create_tagged_text(original: str, differences: List[GrammarDifference]) -> str:
    """Create tagged text with grammar-mistake tags, handling proper alignment."""
    if not differences:
        return original

    words = original.split()
    
    # Group differences by position to handle multiple changes at the same location
    position_changes = {}
    for diff in differences:
        pos = diff.position
        if pos not in position_changes:
            position_changes[pos] = []
        position_changes[pos].append(diff)
    
    # Process positions in reverse order to avoid index shifting
    for pos in sorted(position_changes.keys(), reverse=True):
        changes = position_changes[pos]
        
        # Handle all changes at this position
        corrections = []
        original_word = None
        
        for diff in changes:
            if diff.type == 'substitution':
                if pos < len(words):
                    original_word = words[pos]
                    corrections.append(diff.corrected)
            elif diff.type == 'deletion':
                if pos < len(words):
                    original_word = words[pos]
                    corrections.append("")  # Empty string indicates deletion
            elif diff.type == 'addition':
                # For additions, we show what should be added at this position
                corrections.append(diff.corrected)
        
        # Apply the correction markup
        if pos < len(words) and original_word:
            # This is a substitution or deletion of an existing word
            correction_text = " ".join(filter(None, corrections)) if corrections else ""
            words[pos] = f'<grammar-mistake correction="{correction_text}">{original_word}</grammar-mistake>'
        elif corrections:
            # This is an insertion - add a marker for missing words
            # Insert at the position (or at the end if position is beyond current length)
            insertion_text = " ".join(corrections)
            marker = f'<grammar-mistake correction="{insertion_text}">...</grammar-mistake>'
            if pos >= len(words):
                words.append(marker)
            else:
                words.insert(pos, marker)
    
    return ' '.join(words)


def analyze_speech_fluency(text: str, audio: np.ndarray, sr: int = 16000) -> SpeechMetrics:
    """Perform comprehensive speech fluency analysis."""
    # Get phonemes for filler word detection
    phonemes = phonemes_from_audio(audio)
    logger.debug(f"Extracted phonemes for filler detection: {phonemes[:20]}...")  # Show first 20 for debugging
    
    # Detect pauses
    pauses = detect_pauses(audio, sr)
    
    # Detect filler words
    filler_details, filler_count = detect_filler_words(text, phonemes)
    logger.debug(f"Detected {filler_count} filler words: {[f.text for f in filler_details]}")
    
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
    use_audio: str = Form("false"),
    deep_analysis: str = Form("false"),
    question_text: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
):
    # 🚀 CONCURRENCY CONTROL - Limit concurrent processing for better performance
    async with audio_processing_semaphore:
        # Processing request with concurrency control
        
        if file.content_type is None or not (
            file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
        ):
            raise HTTPException(status_code=400, detail="Please upload an audio file.")

    # Convert string parameters to booleans
    use_audio_bool = use_audio.lower() in ('true', '1', 'yes', 'on')
    deep_analysis_bool = deep_analysis.lower() in ('true', '1', 'yes', 'on')
    
    # Convert parameters and validate input

    audio = load_audio_to_mono16k(await file.read())
    
    # Choose between audio transcription or browser transcript based on use_audio flag
    if use_audio_bool:
        # Use Whisper transcription and extract phonemes from audio
        predicted_text = transcribe_faster_whisper(audio)
        if not predicted_text.strip():
            raise HTTPException(status_code=500, detail="Could not transcribe audio. Please check audio quality.")
        
        # Extract phonemes from audio for pronunciation analysis
        recognized_phonemes = phonemes_from_audio(audio)
        
        # Show expected phonemes for comparison
        from .utils_text import normalize_text_for_phonemization, phonemize_words_en
        normalized_text = normalize_text_for_phonemization(predicted_text)
        expected_phonemes = []
        for word_phonemes in phonemize_words_en(normalized_text):
            expected_phonemes.extend(word_phonemes)
        # Calculate detection ratio for quality assessment
        detection_ratio = len(recognized_phonemes)/len(expected_phonemes) if expected_phonemes else 0
        
        # If Allosaurus over-segmented (> 130%), try to reduce alignment confusion
        if detection_ratio > 1.3:
            logger.warning(f"🦎 Allosaurus over-segmentation detected ({detection_ratio*100:.1f}%). This may cause alignment issues.")
            # Could implement filtering here if needed
        
        # Debug phoneme data available if needed
        
        # Check if fix_allosaurus_oversegmentation worked
        has_consecutive_duplicates = any(recognized_phonemes[i] == recognized_phonemes[i+1] 
                                       for i in range(len(recognized_phonemes)-1) 
                                       if i < len(recognized_phonemes)-1)
        # Check for consecutive duplicates (handled by oversegmentation fix)
        
        if not recognized_phonemes:
            # Failed to extract phonemes, using fallback scoring
            # Fallback: give reasonable scores without phoneme details
            pred_words = tokenize_words(predicted_text)
            out_words = []
            for word in pred_words:
                # Give good scores for clear speech (based on successful Whisper transcription)
                word_score = 85.0
                phonemes = [PhonemeScore(ipa_label="transcribed", phoneme_score=85.0)]
                out_words.append(WordPronunciation(word_text=word, phonemes=phonemes, word_score=word_score))
            
            pronunciation_result = PronunciationResult(words=out_words, overall_score=85.0)
        else:
            # Use real phoneme-level analysis with Oxford IPA system
            from .utils_text import normalize_text_for_phonemization
            
            # Normalize text for consistent phonemization (41 -> forty one)
            normalized_text = normalize_text_for_phonemization(predicted_text)
            
            # Get expected phonemes from normalized transcription
            pred_words = tokenize_words(normalized_text)
            pred_ipas_words = phonemize_words_en(normalized_text)
            
            # Prepare words and phonemes for alignment
            
            # Use improved phoneme-level analysis
            pronunciation_analysis = await analyze_audio_pronunciation(pred_words, pred_ipas_words, recognized_phonemes, predicted_text, audio, deep_analysis_bool)
            pronunciation_result = pronunciation_analysis.pronunciation
        
        # Get speech metrics from audio for fluency analysis
        metrics = None
        if deep_analysis_bool and audio is not None:
            # Performing speech fluency analysis
            metrics = analyze_speech_fluency(predicted_text, audio, sr=16000)
        
        # Create final analysis response
        pronunciation_analysis = AnalyzeResponse(
            pronunciation=pronunciation_result, 
            predicted_text=predicted_text,
            metrics=metrics
        )
        
        # Add grammar analysis if deep_analysis is enabled
        if deep_analysis_bool:
            async with grammar_analysis_semaphore:
                grammar_analysis, relevance_analysis = await perform_comprehensive_ielts_analysis(predicted_text, question_text, context)
            ielts_score = calculate_ielts_score(grammar_analysis, pronunciation_analysis.pronunciation, pronunciation_analysis.metrics, relevance_analysis)
            
            return AnalyzeResponse(
                pronunciation=pronunciation_analysis.pronunciation,
                predicted_text=pronunciation_analysis.predicted_text,
                metrics=pronunciation_analysis.metrics,
                grammar=grammar_analysis,
                relevance=relevance_analysis,
                ielts_score=ielts_score
            )
        else:
            return pronunciation_analysis
        
    else:
        logger.info("📝 Using browser transcript mode: text-vs-text comparison")
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

    from difflib import SequenceMatcher
    sm = SequenceMatcher(a=[w.lower() for w in pred_words], b=[w.lower() for w in said_words])
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
                        phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=random_low_score()))
                    elif (a in (None, "∅")) and b not in (None, "∅"):
                        phonemes.append(PhonemeScore(ipa_label=b, phoneme_score=random_low_score()))
                word_scores = [p.phoneme_score for p in phonemes]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(word_scores)) if word_scores else random_low_score()))
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
                        phonemes.append(PhonemeScore(ipa_label=a, phoneme_score=random_low_score()))
                    elif (a in (None, "∅")) and b not in (None, "∅"):
                        phonemes.append(PhonemeScore(ipa_label=b, phoneme_score=random_low_score()))
                word_scores = [p.phoneme_score for p in phonemes]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=float(np.mean(word_scores)) if word_scores else random_low_score()))
            for ei in exp_indices:
                if ei in used_e:
                    continue
                w_text = pred_words[ei]
                exp_ipas = normalize_ipa(pred_ipas_words[ei]) if ei < len(pred_ipas_words) else []
                phonemes = [PhonemeScore(ipa_label=p, phoneme_score=random_low_score()) for p in exp_ipas]
                out_words.append(WordPronunciation(word_text=w_text, phonemes=phonemes, word_score=random_low_score()))

    overall = float(np.mean([w.word_score for w in out_words])) if out_words else random_low_score()
    pronunciation_result = PronunciationResult(words=out_words, overall_score=overall)
    
    # Add grammar analysis if deep_analysis is enabled
    if deep_analysis_bool:
        async with grammar_analysis_semaphore:
            grammar_analysis, relevance_analysis = await perform_comprehensive_ielts_analysis(predicted_text, question_text, context)
        ielts_score = calculate_ielts_score(grammar_analysis, pronunciation_result, None, relevance_analysis)
        
        return AnalyzeResponse(
            pronunciation=pronunciation_result,
            predicted_text=predicted_text,
            metrics=None,
            grammar=grammar_analysis,
            relevance=relevance_analysis,
            ielts_score=ielts_score
        )
    else:
        return AnalyzeResponse(
            pronunciation=pronunciation_result, 
            predicted_text=predicted_text,
            metrics=None
        )


