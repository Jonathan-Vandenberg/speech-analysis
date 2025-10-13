# Language Analysis API Endpoints - Technical Documentation

## Overview
The speech analysis system has multiple endpoints for different types of pronunciation and language analysis. This document provides a technical analysis of each endpoint, their parameters, processing logic, and current issues.

## Endpoint Analysis

### 1. `/scripted` Route (routes_scripted.py)

#### Purpose
Analyzes pronunciation when users read predetermined text. Designed for reading exercises where the expected text is known.

#### Parameters
- `expected_text` (str, required): The text the user is supposed to read
- `browser_transcript` (str, optional): Speech-to-text transcript from browser
- `analysis_type` (str, optional): Type of analysis ("READING" or "PRONUNCIATION")
- `file` (UploadFile, optional): Audio file containing the user's speech
- `api_key_info` (APIKeyInfo, dependency): API authentication

#### Current Processing Logic
1. **Text-Based Analysis Only**: Currently uses browser transcript for pronunciation scoring
2. **Reading vs Pronunciation Logic**:
   - `READING` type: Performs exact text matching, penalizes incorrect words
   - Non-reading: Uses phonetic comparison between expected and transcribed text
3. **Phonemization**: Converts both expected and "said" text to phonemes using text-to-phoneme models
4. **Alignment**: Uses Needleman-Wunsch algorithm for phoneme alignment and scoring

#### Critical Issues Identified
- **MAJOR BUG**: Audio file is uploaded but never used for pronunciation analysis
- Uses browser transcript instead of actual audio phoneme extraction
- If transcript is wrong (e.g., "Drink the juice" → "Eat the cake"), evaluates wrong pronunciation
- Line 252: `"use_audio": "false"` - explicitly indicates audio not used

#### Return Format
```json
{
  "pronunciation": {
    "words": [
      {
        "word_text": "example",
        "phonemes": [{"ipa_label": "ɪ", "phoneme_score": 95.2}],
        "word_score": 95.2
      }
    ],
    "overall_score": 80.2
  },
  "predicted_text": "transcript text"
}
```

### 2. `/pronunciation` Route (routes_scripted.py)

#### Purpose
Proper audio-based pronunciation analysis using ML models. This is the **correct implementation** that the `/scripted` route should emulate.

#### Parameters
- `expected_text` (str, required): Text user should pronounce
- `file` (UploadFile, required): Audio file (mandatory)
- `api_key_info` (APIKeyInfo, dependency): API authentication

#### Processing Logic (CORRECT APPROACH)
1. **Audio Processing**: Converts audio to 16kHz mono format
2. **ML Phoneme Extraction**: Uses `phonemes_from_audio(audio)` function
3. **Models Used**:
   - Primary: Allosaurus (superior phoneme recognition)
   - Fallback: wav2vec2 (Facebook's model)
4. **Expected Phoneme Generation**: Converts expected text to phonemes
5. **Audio-Text Alignment**: Aligns extracted audio phonemes with expected phonemes
6. **Word-Level Scoring**: Groups phoneme scores back into word-level scores

#### Key Features
- **Direct audio analysis** - no reliance on transcription
- **Word boundary preservation** - maintains word structure in results  
- **ML-based extraction** - uses state-of-the-art phoneme recognition
- Line 381: `"use_audio": "true"` - confirms audio-based analysis

### 3. `/unscripted` Route (routes_unscripted.py)

#### Purpose
Comprehensive speech analysis for open-ended speech (e.g., IELTS speaking, conversations). Most feature-rich endpoint.

#### Parameters
- `file` (UploadFile, required): Audio file containing speech
- `browser_transcript` (str, optional): Pre-existing transcript
- `use_audio` (str, default="false"): Whether to use audio analysis or text comparison
- `deep_analysis` (str, default="false"): Enable comprehensive IELTS analysis
- `question_text` (str, optional): Question being answered (for relevance analysis)
- `context` (str, optional): Additional context for analysis
- `api_key_info` (APIKeyInfo, dependency): API authentication

#### Processing Modes

##### Mode 1: Audio-Based Analysis (`use_audio=true`)
1. **Whisper Transcription**: Uses cached Whisper model for speech-to-text
2. **Phoneme Extraction**: Extracts phonemes directly from audio using Allosaurus/wav2vec2
3. **Pronunciation Analysis**: Compares audio phonemes vs expected phonemes from transcript
4. **Fluency Analysis** (if `deep_analysis=true`):
   - Pause detection and timing
   - Filler word detection (um, ah, like, you know)
   - Speech rate calculation (words per minute)
   - Repetition detection
   - Discourse marker analysis

##### Mode 2: Text-Based Analysis (`use_audio=false`)
1. **Browser Transcript**: Uses provided transcript for analysis
2. **Text-vs-Text Comparison**: Phonetic comparison similar to `/scripted` route
3. **Limited Features**: No audio-specific metrics available

#### Deep Analysis Features (IELTS-focused)
When `deep_analysis=true`:

1. **Grammar Analysis**: 
   - OpenAI GPT-4 integration for error correction
   - Grammar mistake highlighting with corrections
   - Lexical resource evaluation

2. **IELTS Scoring**:
   - Fluency & Coherence (based on speech metrics)
   - Lexical Resource (vocabulary analysis)
   - Grammatical Range & Accuracy (error analysis)
   - Pronunciation (phoneme-based scoring)

3. **Relevance Analysis** (if question provided):
   - Content relevance scoring
   - Key points coverage assessment
   - Missing points identification

4. **Model Answers Generation**:
   - Provides example answers for IELTS bands 4-9
   - Includes detailed markup highlighting language features

#### Return Format (Full Analysis)
```json
{
  "pronunciation": { /* standard pronunciation result */ },
  "predicted_text": "transcribed speech",
  "metrics": {
    "speech_rate": 180,
    "pauses": 2,
    "filler_words": 3,
    "filler_words_per_min": 5.2
  },
  "grammar": {
    "original_text": "...",
    "corrected_text": "...", 
    "lexical_band_score": 6.5,
    "grammar_score": 7.0
  },
  "relevance": {
    "relevance_score": 85,
    "key_points_covered": ["topic coverage", "examples provided"]
  },
  "ielts_score": {
    "overall_band": 6.5,
    "fluency_coherence": 6.0,
    "lexical_resource": 6.5,
    "grammatical_range": 7.0,
    "pronunciation": 6.5
  }
}
```

## Technical Infrastructure

### Audio Processing
- **Load Function**: `load_audio_to_mono16k()` - converts any audio format to 16kHz mono
- **Fallback Support**: Uses ffmpeg for unsupported formats (webm, m4a, ogg)
- **Normalization**: Amplitude normalization and peak limiting

### Phoneme Recognition Models
1. **Allosaurus** (Primary):
   - Multilingual phoneme recognition
   - Superior accuracy for pronunciation assessment
   - Requires audio file path (uses temporary files)

2. **wav2vec2** (Fallback):
   - Facebook's pre-trained model
   - Works with numpy audio arrays
   - Less accurate but more accessible

### Phoneme Processing Pipeline
1. **Raw Extraction**: Model outputs raw phoneme sequence
2. **Normalization**: `normalize_ipa()` - standardizes to Oxford IPA
3. **Diphthong Reconstruction**: Fixes model over-segmentation
4. **Alignment**: Needleman-Wunsch algorithm with feature-based costs

### Text Processing
- **Tokenization**: `tokenize_words()` - handles alphanumeric tokens
- **Number Normalization**: Converts "17th" → "seventeenth"
- **Phonemization**: Uses espeak-ng or g2p_en for text-to-phoneme conversion

## Current Issues Summary

### 1. `/scripted` Route Critical Bug
- **Problem**: Analyzes transcript instead of audio for pronunciation
- **Impact**: Incorrect pronunciation scores when transcript is wrong
- **Solution Needed**: Implement audio-based analysis like `/pronunciation` route

### 2. Inconsistent Architecture
- `/pronunciation`: Proper audio analysis ✅
- `/scripted`: Text-based analysis ❌
- `/unscripted`: Both modes available ✅

### 3. Parameter Confusion
- `browser_transcript` vs actual audio usage unclear
- `analysis_type` parameter only affects `/scripted` behavior
- No clear distinction between reading and pronunciation tasks

## Recommendations

### 1. Fix `/scripted` Route
- For `analysis_type="PRONUNCIATION"`: Use audio-based analysis
- For `analysis_type="READING"`: Keep text-based analysis  
- Maintain backward compatibility

### 2. Standardize Audio Processing
- All pronunciation analysis should use audio when available
- Transcript should be secondary/fallback only

### 3. Clear Parameter Semantics
- Document when audio vs transcript is used
- Provide clear guidance on parameter combinations

### 4. Consider Route Consolidation
- `/pronunciation` and `/scripted` have overlapping functionality
- Could be unified with better parameter handling
