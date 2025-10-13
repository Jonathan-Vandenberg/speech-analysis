from typing import List, Optional
import logging

import numpy as np
import soundfile as sf
import io
import tempfile
import subprocess
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends, Request

logger = logging.getLogger("speech_analyzer")

from .schemas import AnalyzeResponse, PronunciationResult, WordPronunciation, PhonemeScore
from .utils_text import tokenize_words, normalize_ipa, phonemize_words_en, phonemes_from_audio
from .utils_align import align_and_score
from .middleware import api_key_bearer, APIKeyInfo, request_tracker, generate_request_id

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
    api_key_info: Optional[APIKeyInfo] = Depends(api_key_bearer),
):
    """Pronunciation analysis using audio-to-phoneme recognition (no text transcription needed)."""
    # Start request tracking
    request_id = generate_request_id()
    request_tracker.start_request(request_id, api_key_info, "pronunciation", request)
    
    if file.content_type is None or not (
        file.content_type.startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
    ):
        raise HTTPException(status_code=400, detail="Please upload an audio file.")
    print("Audio: VALID")

    if not expected_text.strip():
        raise HTTPException(status_code=400, detail="Expected text is required for pronunciation analysis.")

    try:
        # Load and process audio
        audio = load_audio_to_mono16k(await file.read())

        # SIMPLIFIED APPROACH: Use Whisper for transcription, then word-level comparison
        logger.info("🔄 Using simplified word-level pronunciation analysis")
        
        # Get transcription using Whisper (much more reliable than phoneme extraction)
        import tempfile
        import soundfile as sf
        
        # Save audio temporarily for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, audio, 16000)
            temp_wav_path = temp_wav.name
        
        try:
            # Use Whisper for transcription
            from .routes_unscripted import transcribe_faster_whisper
            actual_text = transcribe_faster_whisper(audio)
            logger.info(f"📝 Whisper transcription: '{actual_text}'")
            
            if not actual_text.strip():
                raise HTTPException(status_code=500, detail="Could not transcribe audio. Please check audio quality.")
            
            # Simple word-level comparison
            expected_words = [w.lower().strip() for w in tokenize_words(expected_text)]
            actual_words = [w.lower().strip() for w in tokenize_words(actual_text)]
            
            logger.info(f"🎯 Expected words: {expected_words}")
            logger.info(f"🎤 Actual words: {actual_words}")
            
            # Build pronunciation result with simple word matching
            out_words: List[WordPronunciation] = []
            
            for i, expected_word in enumerate(expected_words):
                word_text = expected_word
                
                # Check if this word was said correctly
                if i < len(actual_words):
                    actual_word = actual_words[i]
                    
                    # Exact match = excellent score
                    if expected_word == actual_word:
                        word_score = 95.0
                        phonemes = [PhonemeScore(ipa_label="✓", phoneme_score=95.0)]
                        logger.info(f"✅ Word match: '{expected_word}' = '{actual_word}' (95%)")
                    
                    # Similar word (edit distance) = medium score  
                    elif len(expected_word) > 2 and len(actual_word) > 2:
                        # Calculate similarity using edit distance
                        import difflib
                        similarity = difflib.SequenceMatcher(None, expected_word, actual_word).ratio()
                        if similarity > 0.7:  # 70% similar
                            word_score = 60.0 + (similarity * 30.0)  # 60-90% range
                            phonemes = [PhonemeScore(ipa_label="~", phoneme_score=word_score)]
                            logger.info(f"📊 Similar word: '{expected_word}' ≈ '{actual_word}' ({word_score:.1f}%)")
                        else:
                            word_score = 15.0  # Very different words
                            phonemes = [PhonemeScore(ipa_label="✗", phoneme_score=15.0)]
                            logger.info(f"❌ Different word: '{expected_word}' ≠ '{actual_word}' (15%)")
                    else:
                        word_score = 10.0  # Short words that don't match
                        phonemes = [PhonemeScore(ipa_label="✗", phoneme_score=10.0)]
                        logger.info(f"❌ Word mismatch: '{expected_word}' ≠ '{actual_word}' (10%)")
                else:
                    # Word not said at all
                    word_score = 5.0
                    phonemes = [PhonemeScore(ipa_label="∅", phoneme_score=5.0)]
                    logger.info(f"❌ Missing word: '{expected_word}' (5%)")
                
                out_words.append(WordPronunciation(
                    word_text=word_text,
                    phonemes=phonemes,
                    word_score=word_score
                ))
            
            # Calculate overall score
            overall = float(np.mean([w.word_score for w in out_words])) if out_words else 5.0
            
            logger.info(f"🎯 Overall pronunciation score: {overall:.1f}%")
            
            response = AnalyzeResponse(
                pronunciation=PronunciationResult(words=out_words, overall_score=overall), 
                predicted_text=actual_text
            )
            
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_wav_path)
            except:
                pass
        
        # Finish request tracking
        audio_data = await file.read()
        form_data = {
            "expected_text": expected_text,
            "deep_analysis": "false",  # pronunciation endpoint doesn't have deep analysis
            "use_audio": "true"  # pronunciation endpoint uses audio
        }
        await request_tracker.finish_request(request_id, response.model_dump(), form_data, audio_data)
        
        return response
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pronunciation analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(exc)}")


