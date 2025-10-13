# Analysis of /scripted Route Pronunciation Issues

## Problem Summary
The /scripted route is producing incorrect pronunciation analysis. User said "Drink the juice" but the system analyzed "Eat the cake" and provided pronunciation scores.

## Current System Analysis

### Critical Issue Identified: Text-Based Analysis Only
The /scripted route is NOT using the audio file for pronunciation analysis. Instead, it's:
1. Using `browser_transcript` (speech-to-text result) as the "said text"
2. Comparing this transcribed text phonetically against `expected_text`
3. **Completely ignoring the actual audio pronunciation**

### Code Flow Analysis:
1. Line 75: `said_text = (browser_transcript or "").strip()`
2. Line 169-171: Phonemizes both expected and said text using text-to-phoneme conversion
3. Lines 173-256: Performs text-based phoneme alignment and scoring

### The Core Problem:
- Speech-to-text can be wrong (user said "Drink the juice", transcript shows "Eat the cake")
- System then evaluates pronunciation of wrong words
- Audio file is uploaded but never analyzed for actual pronunciation

## Proper Solution: Use Audio-Based ML Model

### Current /pronunciation Route (Lines 259-394):
- DOES use actual audio analysis via `phonemes_from_audio(audio)`
- Uses Allosaurus or wav2vec2 ML models for direct audio-to-phoneme extraction
- Provides accurate pronunciation scoring based on actual speech sounds

### Implementation Plan:

#### Phase 1: Fix /scripted Route Logic
1. For PRONUNCIATION analysis type, use audio-based analysis like /pronunciation route
2. Keep READING analysis type as text-based (working correctly per user)
3. Add proper audio processing for pronunciation evaluation

#### Phase 2: ML Model Integration
1. Use existing Allosaurus model (preferred) or wav2vec2 fallback
2. Extract phonemes directly from audio using `phonemes_from_audio()`
3. Align extracted phonemes with expected phonemes from text
4. Provide accurate pronunciation scoring

#### Phase 3: Hybrid Approach
1. Use audio analysis for pronunciation scoring
2. Use transcript for text accuracy (separate metric)
3. Provide both pronunciation and content accuracy scores

## Next Steps:
1. Modify /scripted route to detect PRONUNCIATION analysis type
2. Implement audio-based phoneme extraction for pronunciation tasks
3. Keep existing text-based logic for READING tasks
4. Test with actual audio samples

## Key Files to Modify:
- `routes_scripted.py` - Main logic fix
- Leverage existing `utils_text.py` phoneme extraction functions
- Use existing `utils_align.py` alignment and scoring functions
