# Pronunciation Analysis Improvements

## Current Issues

1. **Whisper transcription is incomplete**: Only transcribing "The" from a 6-second audio file
2. **No forced alignment**: We're relying on Whisper transcription to know what words were said, but this fails when transcription is incomplete
3. **Phoneme extraction works but needs better alignment**: Allosaurus/wav2vec2 extract phonemes, but we need to know which phonemes belong to which words

## Recommended Solutions

### Option 1: Montreal Forced Aligner (MFA) - **RECOMMENDED**

**What it does:**
- Takes expected text + audio file
- Aligns text to audio at word and phoneme level with timestamps
- Provides exact phoneme boundaries for each word
- Industry standard for pronunciation assessment

**Pros:**
- Most accurate forced alignment tool
- Handles multiple languages
- Provides word-level and phoneme-level timestamps
- Widely used in research and production

**Cons:**
- Requires installation and model downloads
- Slightly more complex setup

**Installation (already wired into `/pronunciation`):**
```bash
pip install montreal-forced-aligner
mfa model download dictionary english_us_arpa
mfa model download acoustic english_mfa
```

**Usage (CLI call invoked by the API):**
```bash
mfa align /tmp/corpus english_us_arpa english_mfa /tmp/output \
  --num_jobs 1 --clean --overwrite --disable_mp
```

### Option 2: Bournemouth Forced Aligner (BFA) - **LEGACY (REMOVED)**

**What it does:**
- Newer tool that integrates with Whisper
- Extracts phoneme-level timestamps
- Frame-wise phoneme alignment

**Pros:**
- Integrates with Whisper (which we already use)
- Good for noisy audio
- Active development

**Cons:**
- Less mature than MFA
- Smaller community

> This aligner is no longer part of the production stack; we keep the notes here for historical context in case we ever need to evaluate it again.

### Option 3: Gentle Forced Aligner - **SIMPLE BUT LESS ACCURATE**

**What it does:**
- Web-based forced aligner
- Simple API interface
- Good for quick prototyping

**Pros:**
- Very easy to use
- Web API available
- Quick setup

**Cons:**
- Less accurate than MFA
- Requires internet connection (if using web API)
- May not handle all audio formats well

## Recommended Implementation Strategy

### Phase 1: Integrate MFA for Forced Alignment

1. **Install MFA** in the audio-analysis environment
2. **Modify `/pronunciation` route** to:
   - Use MFA to align expected text with audio
   - Get word-level and phoneme-level timestamps
   - Extract phonemes from audio segments using timestamps
   - Compare expected vs actual phonemes per word

3. **Benefits:**
   - No longer dependent on Whisper transcription accuracy
   - Exact word boundaries from forced alignment
   - Better phoneme-to-word mapping
   - Works even if user says different words (alignment will show where they differ)

### Phase 2: Hybrid Approach (Best of Both Worlds)

1. **Use Whisper for transcription** (to know what was actually said)
2. **Use MFA for forced alignment** (to get exact word/phoneme boundaries)
3. **Combine both**:
   - Whisper tells us: "The bus" (what was said)
   - MFA tells us: word boundaries for expected text "The bus arrives before the class begins"
   - We can then:
     - Identify which expected words were said vs not said
     - Analyze pronunciation for words that were said
     - Mark words that weren't said as missing

## Implementation Example

```python
from montreal_forced_alignment import align_corpus
import tempfile
import os

def forced_align_text_to_audio(expected_text: str, audio_path: str):
    """Use MFA to align expected text with audio."""
    # Create temporary directory structure for MFA
    with tempfile.TemporaryDirectory() as temp_dir:
        # MFA expects specific directory structure
        corpus_dir = os.path.join(temp_dir, "corpus")
        os.makedirs(corpus_dir)
        
        # Copy audio file
        audio_name = "audio.wav"
        audio_dest = os.path.join(corpus_dir, audio_name)
        shutil.copy(audio_path, audio_dest)
        
        # Create text file with expected text
        text_file = os.path.join(corpus_dir, audio_name.replace(".wav", ".txt"))
        with open(text_file, "w") as f:
            f.write(expected_text)
        
        # Run alignment
        alignment = align_corpus(
            corpus_directory=corpus_dir,
            dictionary_path="english_us_arpa",
            acoustic_model_path="english_us_arpa",
            output_directory=os.path.join(temp_dir, "output")
        )
        
        # Extract word and phoneme alignments
        return alignment
```

## Next Steps

1. **Test MFA installation** in the audio-analysis environment
2. **Create a proof-of-concept** integration with `/pronunciation` route
3. **Compare results** with current approach
4. **If successful, replace current alignment logic** with MFA-based approach

## Alternative: Fix Current Approach

If MFA is too complex to integrate, we can improve the current approach:

1. **Better Whisper model**: Use larger Whisper model for better transcription
2. **Audio preprocessing**: Improve audio quality before transcription
3. **Fallback handling**: When Whisper fails, use forced alignment as fallback
4. **Partial transcription handling**: Better logic to handle when only part of text is transcribed
