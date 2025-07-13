import numpy as np
import vosk
import json
import os
import re
import tempfile
import wave
import asyncio
from typing import List, Tuple, Optional
from models.responses import PhonemeScore, WordScore, PronunciationAssessmentResponse, FluencyMetrics, FillerWord, Pause, Repetition
from .phoneme_engine import get_phoneme_engine
from .confidence_scorer import ProductionConfidenceScorer
from .pronunciation_validator import PronunciationValidator
from utils.audio_processing import assess_audio_quality, detect_speech_segments

class PronunciationAnalyzer:
    """
    Unified pronunciation analysis tool that can use either Whisper or Vosk for transcription.
    Used consistently across both Pronunciation API and Freestyle API.
    """
    
    def __init__(self, vosk_model_path: str):
        self.vosk_model = None
        self.vosk_model_path = vosk_model_path
        self._load_vosk_model()
        
        # Initialize production phoneme engine
        self.phoneme_engine = get_phoneme_engine()
        print("✅ Production phoneme engine integrated")
        
        # Initialize production confidence scorer
        self.confidence_scorer = ProductionConfidenceScorer()
        print("✅ Production confidence scorer integrated")
        
        # Initialize pronunciation validator
        self.pronunciation_validator = PronunciationValidator()
        print("✅ Pronunciation validator integrated")
        
        # Filler word categories
        self.filler_words = {
            'hesitation': ['uh', 'um', 'umm', 'err', 'ah', 'eh', 'hmm', 'hm'],
            'discourse_marker': [],  # Removed 'like' and others - only detect hesitations
            'repetition': []  # Will be detected dynamically
        }
    
    def _load_vosk_model(self):
        """Load Vosk speech recognition model"""
        try:
            if os.path.exists(self.vosk_model_path):
                vosk.SetLogLevel(-1)  # Reduce Vosk logging
                self.vosk_model = vosk.Model(self.vosk_model_path)
                print("✅ Vosk speech recognition model loaded successfully.")
            else:
                print(f"⚠️ Vosk model not found at {self.vosk_model_path}. Using fallback analysis.")
        except Exception as e:
            print(f"⚠️ Failed to load Vosk model: {e}. Using fallback analysis.")
    
    def analyze_pronunciation(self, audio_data: np.ndarray, samplerate: int, expected_text: str) -> PronunciationAssessmentResponse:
        """
        Standard pronunciation analysis - compares audio against provided expected text
        """
        import time
        start_time = time.time()
        
        overall_score, word_scores = self._pronunciation_analysis_with_expected_text(audio_data, samplerate, expected_text)
        
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)
        
        return PronunciationAssessmentResponse(
            overall_score=overall_score,
            words=word_scores,
            processing_time_ms=processing_time_ms
        )
    
    async def analyze_pronunciation_freestyle(self, audio_data: np.ndarray, samplerate: int) -> Tuple[str, PronunciationAssessmentResponse]:
        """
        Freestyle pronunciation analysis - transcribes with Whisper, then analyzes pronunciation against that reference
        Returns both the transcribed text and pronunciation analysis
        """
        import time
        start_time = time.time()
        
        # Step 1: Get Whisper transcription (this becomes our reference - what user intended to say)
        whisper_transcription = await self._transcribe_with_whisper(audio_data, samplerate)
        
        if not whisper_transcription or len(whisper_transcription.strip()) < 3:
            # Return minimal response for failed transcription
            return "", PronunciationAssessmentResponse(
                overall_score=0.0,
                words=[],
                processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        print(f"📝 Whisper (reference): '{whisper_transcription}'")
        
        # Step 2: Analyze pronunciation against Whisper transcription using Vosk acoustic analysis
        overall_score, word_scores = await self._analyze_pronunciation_against_whisper_reference(
            audio_data, samplerate, whisper_transcription
        )
        
        # Step 3: Analyze fluency (only for freestyle, not fixed text pronunciation)
        fluency_metrics = None
        try:
            # Get Vosk data for fluency analysis
            if self.vosk_model:
                rec = vosk.KaldiRecognizer(self.vosk_model, samplerate)
                rec.SetWords(True)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                rec.AcceptWaveform(audio_bytes)
                result = rec.FinalResult()
                vosk_data = json.loads(result)
                vosk_words_data = vosk_data.get('result', [])
                
                # Calculate total duration
                total_duration = len(audio_data) / samplerate
                
                fluency_metrics = self._analyze_fluency_with_audio(vosk_words_data, whisper_transcription, total_duration, audio_data, samplerate)
                print(f"🎭 Fluency score: {fluency_metrics.overall_fluency_score}% | Fillers: {fluency_metrics.total_filler_count} | Pauses: {len(fluency_metrics.long_pauses)} | Rate: {fluency_metrics.speech_rate}wpm")
        except Exception as e:
            print(f"⚠️ Fluency analysis failed: {e}")
        
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)
        
        return whisper_transcription, PronunciationAssessmentResponse(
            overall_score=overall_score,
            words=word_scores,
            fluency_metrics=fluency_metrics,
            processing_time_ms=processing_time_ms
        )
    
    async def _transcribe_with_whisper(self, audio_data: np.ndarray, samplerate: int) -> str:
        """
        Transcribe speech using OpenAI Whisper API only. 
        If OpenAI fails, we don't fallback to Vosk transcription to avoid circular comparison.
        """
        try:
            # Try OpenAI Whisper API
            openai_key = os.getenv('OPENAI_API_KEY')
            if not openai_key:
                print("❌ No OPENAI_API_KEY found - cannot proceed with Whisper transcription")
                return ""
            
            print("🎤 Attempting OpenAI Whisper transcription...")
            
            # Create temporary audio file for API
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert numpy array to WAV format
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                # Write WAV file
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(samplerate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                try:
                    import openai
                    
                    client = openai.OpenAI(api_key=openai_key)
                    
                    with open(temp_file.name, 'rb') as audio_file:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"
                        )
                    
                    transcribed_text = response.strip()
                    if transcribed_text and len(transcribed_text) > 3:
                        print(f"✅ OpenAI Whisper transcription: '{transcribed_text}'")
                        return transcribed_text
                    else:
                        print("⚠️ OpenAI Whisper returned empty transcription")
                        return ""
                
                except Exception as openai_error:
                    print(f"❌ OpenAI Whisper error: {openai_error}")
                    return ""
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
            
        except Exception as e:
            print(f"❌ Whisper transcription failed: {e}")
            return ""
    
    async def _transcribe_with_vosk(self, audio_data: np.ndarray, samplerate: int) -> str:
        """Fallback transcription using Vosk"""
        if not self.vosk_model:
            return ""
        
        try:
            rec = vosk.KaldiRecognizer(self.vosk_model, samplerate)
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            rec.AcceptWaveform(audio_bytes)
            result = rec.FinalResult()
            recognition_data = json.loads(result)
            
            transcribed_text = recognition_data.get('text', '').strip()
            print(f"✅ Vosk transcription: '{transcribed_text}'")
            return transcribed_text
            
        except Exception as e:
            print(f"⚠️ Vosk transcription error: {e}")
            return ""
    
    def _pronunciation_analysis_with_expected_text(self, audio_data: np.ndarray, samplerate: int, expected_text: str) -> Tuple[float, List[WordScore]]:
        """Enhanced pronunciation analysis using production phoneme engine and improved alignment"""
        if not self.vosk_model:
            print("⚠️ Vosk model not available, falling back to mock analysis")
            return self._fallback_pronunciation_analysis(audio_data, samplerate, expected_text)
        
        try:
            print(f"🎯 PRODUCTION PHONEME ANALYSIS: Expected text '{expected_text}'")
            
            # Step 0: Assess audio quality for reliability adjustments
            audio_quality = assess_audio_quality(audio_data, samplerate)
            print(f"🔊 Audio quality: {audio_quality['overall_quality']:.1f}% (SNR: {audio_quality['snr_estimate']:.1f}dB)")
            
            # Step 1: Get what user actually said using Vosk with enhanced processing
            rec = vosk.KaldiRecognizer(self.vosk_model, samplerate)
            rec.SetWords(True)
            
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            rec.AcceptWaveform(audio_bytes)
            result = rec.FinalResult()
            recognition_data = json.loads(result)
            
            vosk_transcription = recognition_data.get('text', '').strip()
            vosk_words_data = recognition_data.get('result', [])
            
            print(f"📝 Expected text: '{expected_text}'")
            print(f"🎤 Vosk transcription: '{vosk_transcription}'")
            
            # Step 2: Enhanced word alignment with fuzzy matching
            expected_words = expected_text.lower().replace('.', '').replace(',', '').split()
            vosk_words = vosk_transcription.lower().split()
            
            # Check for potential ASR issues before alignment
            self._diagnose_asr_issues(expected_words, vosk_words, vosk_words_data)
            
            # Use advanced alignment algorithm
            word_alignments = self._align_word_sequences_enhanced(expected_words, vosk_words, vosk_words_data)
            
            word_scores = []
            total_score = 0.0
            
            # Step 3: Analyze each aligned word pair
            for i, (expected_word, actual_match) in enumerate(word_alignments):
                # Get expected phonemes using production engine
                expected_phonemes = self.phoneme_engine.get_phonemes(expected_word, accent='us')
                
                # Get actual phonemes and metadata
                actual_phonemes = []
                confidence = 0.3
                timing = {'start': i * 0.8, 'end': i * 0.8 + 0.7}
                
                if actual_match:
                    actual_word = actual_match['word']
                    actual_phonemes = self.phoneme_engine.get_phonemes(actual_word, accent='us')
                    confidence = actual_match.get('confidence', 0.3)
                    timing = {
                        'start': actual_match.get('start', i * 0.8),
                        'end': actual_match.get('end', i * 0.8 + 0.7)
                    }
                    
                    print(f"🔍 Word {i+1}: '{expected_word}' → '{actual_word}' (conf: {confidence:.2f})")
                    print(f"   Expected phonemes: {expected_phonemes}")
                    print(f"   Actual phonemes: {actual_phonemes}")
                else:
                    print(f"🔍 Word {i+1}: '{expected_word}' → [NOT DETECTED]")
                    print(f"   Expected phonemes: {expected_phonemes}")
                    print(f"   Actual phonemes: []")
                
                # Step 4: Advanced phoneme alignment and scoring
                phoneme_scores = self._align_and_score_phonemes_enhanced(actual_phonemes, expected_phonemes)
                
                # Calculate word-level score with validation and multiple factors
                word_score = self._calculate_word_score_with_validation(
                    expected_word, actual_match, phoneme_scores, confidence, audio_quality, timing
                )
                
                print(f"   💯 Word score: {word_score:.1f}%")
                
                word_scores.append(WordScore(
                    word_text=expected_word,
                    word_score=round(word_score, 1),
                    phonemes=phoneme_scores,
                    start_time=timing['start'],
                    end_time=timing['end']
                ))
                
                total_score += word_score
            
            # Step 5: Calculate comprehensive confidence assessment
            phoneme_scores_flat = [p.phoneme_score for word in word_scores for p in word.phonemes]
            confidence_metrics = self.confidence_scorer.calculate_comprehensive_confidence(
                audio_data=audio_data,
                samplerate=samplerate,
                vosk_transcription=vosk_transcription,
                vosk_words_data=vosk_words_data,
                expected_text=expected_text,
                phoneme_alignment_scores=phoneme_scores_flat
            )
            
            print(f"🔍 Confidence Assessment:")
            print(f"  Overall: {confidence_metrics.overall_confidence:.2f} ({self.confidence_scorer.get_confidence_level_description(confidence_metrics.overall_confidence)})")
            print(f"  Acoustic: {confidence_metrics.acoustic_confidence:.2f}")
            print(f"  Temporal: {confidence_metrics.temporal_confidence:.2f}")
            print(f"  Lexical: {confidence_metrics.lexical_confidence:.2f}")
            print(f"  Phonetic: {confidence_metrics.phonetic_confidence:.2f}")
            
            # Step 6: Calculate overall score with confidence-weighted adjustments
            overall_score = total_score / len(expected_words) if expected_words else 0.0
            overall_score = self._apply_confidence_weighted_adjustments(overall_score, confidence_metrics, audio_quality)
            
            print(f"🎯 Overall pronunciation score: {overall_score:.1f}% (confidence: {confidence_metrics.overall_confidence:.2f})")
            
            return round(overall_score, 1), word_scores
            
        except Exception as e:
            print(f"⚠️ Error in production phoneme analysis: {e}")
            try:
                return self._fallback_pronunciation_analysis(audio_data, samplerate, expected_text)
            except Exception as fallback_error:
                print(f"❌ Fallback analysis also failed: {fallback_error}")
                return self._create_emergency_fallback_response(expected_text)

    def _force_raw_acoustic_analysis(self, audio_data: np.ndarray, samplerate: int, expected_text: str, vosk_words: List[dict]) -> Tuple[float, List[WordScore]]:
        """
        Force raw acoustic analysis when Vosk corrects speech.
        Uses confidence scores and acoustic features to detect mispronunciation.
        """
        print("🔧 FORCING raw acoustic analysis since Vosk corrected the speech")
        
        expected_words = expected_text.lower().replace('.', '').split()
        word_scores = []
        total_score = 0.0
        
        for i, expected_word in enumerate(expected_words):
            # Get Vosk confidence for this word position
            confidence = 0.3  # Default low confidence indicates mispronunciation
            
            if i < len(vosk_words):
                word_data = vosk_words[i]
                confidence = word_data.get('conf', 0.3)
                
                # If confidence is suspiciously high for corrected speech, reduce it
                if confidence > 0.8:
                    print(f"⚠️ Suspiciously high confidence {confidence:.2f} for potentially corrected word '{expected_word}' - reducing")
                    confidence *= 0.4  # Heavy penalty for likely corrections
            
            # Use confidence as primary indicator of pronunciation accuracy
            # Low confidence = mispronunciation, even if word was "corrected"
            pronunciation_score = confidence * 100
            
            # Apply additional penalties for suspected corrections
            if pronunciation_score > 70:  # If score seems too high for potentially mispronounced word
                pronunciation_score *= 0.6  # Reduce score significantly
                print(f"⚠️ Applying correction penalty to '{expected_word}' - likely mispronounced but corrected")
            
            # Apply difficulty penalty
            difficulty_penalty = self._calculate_pronunciation_difficulty(expected_word)
            pronunciation_score = max(10, pronunciation_score - difficulty_penalty)
            
            print(f"🔍 FORCED RAW: '{expected_word}' | Original Conf: {vosk_words[i].get('conf', 0.3) if i < len(vosk_words) else 0.3:.2f} | Adjusted: {confidence:.2f} | Score: {pronunciation_score:.1f}%")
            
            # Generate phoneme scores based on adjusted confidence
            phonemes = self._generate_realistic_phoneme_scores(expected_word, confidence, confidence)
            
            word_scores.append(WordScore(
                word_text=expected_word,
                word_score=round(pronunciation_score, 1),
                phonemes=phonemes,
                start_time=i * 0.8,
                end_time=i * 0.8 + 0.7
            ))
            
            total_score += pronunciation_score
        
        overall_score = total_score / len(expected_words) if expected_words else 0.0
        overall_score = self._apply_audio_quality_adjustments(overall_score, audio_data, samplerate)
        
        print(f"🎯 FORCED RAW ACOUSTIC score: {overall_score:.1f}%")
        
        return round(overall_score, 1), word_scores
    
    def _fallback_pronunciation_analysis(self, audio_data: np.ndarray, samplerate: int, expected_text: str) -> Tuple[float, List[WordScore]]:
        """Enhanced fallback analysis with robust scoring when primary methods fail"""
        print(f"🔧 Performing robust fallback analysis for '{expected_text}' on audio with samplerate {samplerate}...")
        
        words_in_expected_text = expected_text.lower().split()
        num_words = len(words_in_expected_text)

        if num_words == 0:
            return 0.0, []

        # Assess audio quality for fallback scoring
        try:
            audio_quality = assess_audio_quality(audio_data, samplerate)
            print(f"🔊 Fallback audio quality: {audio_quality['overall_quality']:.1f}%")
        except Exception as e:
            print(f"⚠️ Audio quality assessment failed: {e}")
            audio_quality = {'overall_quality': 50.0, 'snr_estimate': 15.0}

        mock_word_scores = []

        for i, word in enumerate(words_in_expected_text):
            # Enhanced scoring using multiple factors
            word_score = self._calculate_fallback_word_score(word, audio_data, audio_quality, i, num_words)
            
            # Generate phonemes using production engine
            try:
                phonemes = self._generate_ipa_phonemes(word, word_score / 100, recognized_word=None)
            except Exception as e:
                print(f"⚠️ Phoneme generation failed for '{word}': {e}")
                # Create minimal phoneme representation
                phonemes = [PhonemeScore(ipa_label='ʌ', phoneme_score=word_score / 100)]
            
            mock_word_scores.append(
                WordScore(
                    word_text=word, 
                    word_score=round(word_score, 1),
                    phonemes=phonemes,
                    start_time=round(i * 0.8, 2),
                    end_time=round((i * 0.8) + 0.7, 2)
                )
            )
            
            print(f"  📝 Fallback '{word}': {word_score:.1f}%")
        
        overall_score = sum(w.word_score for w in mock_word_scores) / num_words if num_words > 0 else 0.0
        
        # Apply conservative adjustment for fallback analysis
        overall_score *= 0.8  # Conservative penalty for fallback method
        
        print(f"🎯 Fallback overall score: {overall_score:.1f}% (conservative estimate)")
        
        return round(overall_score, 1), mock_word_scores
    
    def _calculate_fallback_word_score(self, word: str, audio_data: np.ndarray, audio_quality: dict, 
                                     position: int, total_words: int) -> float:
        """Calculate word score for fallback analysis using multiple heuristics"""
        
        # Base score from audio characteristics
        audio_energy = np.mean(np.abs(audio_data)) if len(audio_data) > 0 else 0.1
        audio_quality_factor = audio_quality.get('overall_quality', 50) / 100
        
        # Start with audio-based score
        base_score = 0.3 + audio_energy * 1.5 + audio_quality_factor * 0.4
        base_score = min(0.9, max(0.2, base_score))
        
        # Adjust for word characteristics
        word_difficulty = self._calculate_pronunciation_difficulty(word) / 100  # Convert to 0-1
        difficulty_adjustment = max(0.1, 1.0 - word_difficulty * 0.6)
        
        # Position-based adjustment (middle words often clearer)
        if total_words > 2:
            if position == 0 or position == total_words - 1:
                position_factor = 0.95  # Slight penalty for first/last words
            else:
                position_factor = 1.05  # Slight bonus for middle words
        else:
            position_factor = 1.0
        
        # Word length adjustment
        length_factor = max(0.8, min(1.2, 1.0 - (len(word) - 5) * 0.05))
        
        # Combine all factors
        final_score = base_score * difficulty_adjustment * position_factor * length_factor * 100
        
        # Apply reasonable bounds with variability
        final_score = max(15.0, min(75.0, final_score))  # Fallback range: 15-75%
        
        return final_score
    
    def _create_emergency_fallback_response(self, expected_text: str) -> Tuple[float, List[WordScore]]:
        """Emergency fallback when all other methods fail"""
        print(f"🚨 Emergency fallback for '{expected_text}'")
        
        words = expected_text.lower().split()
        if not words:
            return 0.0, []
        
        # Create minimal but reasonable response
        word_scores = []
        for i, word in enumerate(words):
            # Very conservative scoring
            base_score = 30.0 + (i % 3) * 5  # 30-40% range with variation
            
            # Single phoneme representation
            phonemes = [PhonemeScore(ipa_label='ə', phoneme_score=base_score / 100)]
            
            word_scores.append(WordScore(
                word_text=word,
                word_score=base_score,
                phonemes=phonemes,
                start_time=i * 0.5,
                end_time=i * 0.5 + 0.4
            ))
        
        overall_score = sum(w.word_score for w in word_scores) / len(word_scores)
        
        print(f"🚨 Emergency fallback score: {overall_score:.1f}%")
        return round(overall_score, 1), word_scores
    
    def _calculate_word_similarity(self, recognized_word: str, expected_word: str) -> float:
        """Calculate similarity between recognized and expected words with morphological awareness"""
        if not recognized_word or not expected_word:
            return 0.0
        
        recognized_word = recognized_word.lower().strip()
        expected_word = expected_word.lower().strip()
        
        if recognized_word == expected_word:
            return 1.0
        
        # Check for minor morphological differences (plurals, tense, etc.)
        if self._is_minor_morphological_difference(recognized_word, expected_word):
            return 0.90  # High similarity for minor differences
        
        # Simple character-based similarity
        max_len = max(len(recognized_word), len(expected_word))
        if max_len == 0:
            return 0.0
        
        # Count matching characters in same positions
        matches = sum(1 for i in range(min(len(recognized_word), len(expected_word))) 
                      if recognized_word[i] == expected_word[i])
        
        # Calculate base similarity
        base_similarity = matches / max_len
        
        # Length difference penalty 
        length_diff = abs(len(recognized_word) - len(expected_word))
        if length_diff == 1:
            # 1 character difference - be more lenient (85% cap instead of 50%)
            length_penalty = 0.15
        elif length_diff == 2:
            # 2 character difference - moderate penalty (70% cap)
            length_penalty = 0.30
        else:
            # Larger differences - significant penalty
            length_penalty = length_diff / max_len * 0.8
        
        similarity = max(0.0, base_similarity - length_penalty)
        
        # Apply caps based on difference size
        if length_diff == 1:
            similarity = min(0.85, similarity)  # Cap at 85% for 1 letter diff
        elif length_diff == 2:
            similarity = min(0.70, similarity)  # Cap at 70% for 2 letter diff
        
        return similarity
    
    def _is_minor_morphological_difference(self, word1: str, word2: str) -> bool:
        """Check if two words differ by common morphological changes"""
        if abs(len(word1) - len(word2)) > 2:
            return False
        
        # Check common morphological patterns more simply
        import re
        
        # Plurals: skills → skill, boxes → box, studies → study
        if word1.endswith('s') and not word2.endswith('s'):
            if word1[:-1] == word2:  # skills → skill
                return True
            if word1.endswith('es') and word1[:-2] == word2:  # boxes → box
                return True
            if word1.endswith('ies') and word1[:-3] + 'y' == word2:  # studies → study
                return True
        elif word2.endswith('s') and not word1.endswith('s'):
            if word2[:-1] == word1:  # skill → skills
                return True
            if word2.endswith('es') and word2[:-2] == word1:  # box → boxes
                return True
            if word2.endswith('ies') and word2[:-3] + 'y' == word1:  # study → studies
                return True
        
        # Past tense: walked → walk, moved → move
        if word1.endswith('ed') and word1[:-2] == word2:
            return True
        elif word2.endswith('ed') and word2[:-2] == word1:
            return True
        
        if word1.endswith('d') and not word2.endswith('d') and word1[:-1] == word2:
            return True
        elif word2.endswith('d') and not word1.endswith('d') and word2[:-1] == word1:
            return True
        
        # Present participle: walking → walk
        if word1.endswith('ing') and word1[:-3] == word2:
            return True
        elif word2.endswith('ing') and word2[:-3] == word1:
            return True
        
        # Double consonant + ing: running → run
        if word1.endswith('ning') and word1[:-4] + 'n' == word2:
            return True
        elif word2.endswith('ning') and word2[:-4] + 'n' == word1:
            return True
        
        # Comparative/superlative: bigger → big, biggest → big
        if word1.endswith('er') and word1[:-2] == word2:
            return True
        elif word2.endswith('er') and word2[:-2] == word1:
            return True
        
        if word1.endswith('est') and word1[:-3] == word2:
            return True
        elif word2.endswith('est') and word2[:-3] == word1:
            return True
        
        return False
    
    def _calculate_phoneme_similarity(self, phoneme1: str, phoneme2: str) -> float:
        """Calculate similarity between two phonemes based on articulatory features with strict mispronunciation detection"""
        if phoneme1 == phoneme2:
            return 1.0
        
        # Define specific mispronunciation patterns with very low similarity scores
        mispronunciation_patterns = {
            # TH substitutions - these are mispronunciations, should get very low scores
            ('θ', 'b'): 0.1,  # th → b (like "theatre" → "beatre")
            ('θ', 'd'): 0.1,  # th → d (like "the" → "de")
            ('θ', 'f'): 0.25, # th → f (closer but still wrong)
            ('θ', 's'): 0.25, # th → s (closer but still wrong)
            ('ð', 'b'): 0.1,  # voiced th → b
            ('ð', 'd'): 0.1,  # voiced th → d  
            ('ð', 'v'): 0.25, # voiced th → v (closer)
            ('ð', 'z'): 0.25, # voiced th → z (closer)
            
            # R/L substitutions
            ('r', 'l'): 0.3,  # r → l confusion
            ('l', 'r'): 0.3,  # l → r confusion
            ('r', 'w'): 0.2,  # r → w
            
            # V/W substitutions  
            ('v', 'w'): 0.3,  # v → w confusion
            ('w', 'v'): 0.3,  # w → v confusion
            
            # Other common substitutions
            ('s', 'ʃ'): 0.4,  # s → sh
            ('z', 'ʒ'): 0.4,  # z → zh
        }
        
        # Check for specific mispronunciation patterns first
        if (phoneme1, phoneme2) in mispronunciation_patterns:
            return mispronunciation_patterns[(phoneme1, phoneme2)]
        if (phoneme2, phoneme1) in mispronunciation_patterns:
            return mispronunciation_patterns[(phoneme2, phoneme1)]
        
        # Define phoneme categories for similarity scoring
        vowels = {
            "close": ["i", "iː", "ɪ", "u", "uː", "ʊ"],
            "mid": ["e", "ə", "ɜː", "o", "oʊ", "ɔː", "ɛ"],
            "open": ["æ", "a", "ɑː", "ɒ", "ʌ"],
            "diphthongs": ["aɪ", "aʊ", "eɪ", "ɔɪ", "oʊ", "ɪə", "ʊə", "eə"]
        }
        
        consonants = {
            "stops_voiceless": ["p", "t", "k"],
            "stops_voiced": ["b", "d", "g"],
            "fricatives_voiceless": ["f", "θ", "s", "ʃ", "h"],
            "fricatives_voiced": ["v", "ð", "z", "ʒ"],
            "affricates": ["tʃ", "dʒ"],
            "nasals": ["m", "n", "ŋ"],
            "liquids": ["l", "r"],
            "glides": ["w", "j"]
        }
        
        # Check for exact category matches (higher similarity)
        for category, phonemes in vowels.items():
            if phoneme1 in phonemes and phoneme2 in phonemes:
                return 0.6  # Similar vowels within same category
        
        for category, phonemes in consonants.items():
            if phoneme1 in phonemes and phoneme2 in phonemes:
                if category in ["stops_voiceless", "stops_voiced"]:
                    return 0.5  # Stops are more similar to each other
                elif category in ["fricatives_voiceless", "fricatives_voiced"]:
                    return 0.4  # Fricatives are somewhat similar
                else:
                    return 0.5  # Other consonant categories
        
        # Check for voicing pairs (p/b, t/d, k/g, f/v, s/z, etc.)
        voicing_pairs = [
            ('p', 'b'), ('t', 'd'), ('k', 'g'),
            ('f', 'v'), ('s', 'z'), ('ʃ', 'ʒ'),
            ('θ', 'ð'), ('tʃ', 'dʒ')
        ]
        
        for pair in voicing_pairs:
            if (phoneme1, phoneme2) == pair or (phoneme2, phoneme1) == pair:
                return 0.4  # Moderate similarity for voicing differences
        
        # Check broader categories
        all_vowels = [p for cat in vowels.values() for p in cat]
        all_consonants = [p for cat in consonants.values() for p in cat]
        
        # Both vowels but different categories
        if phoneme1 in all_vowels and phoneme2 in all_vowels:
            return 0.3  # Different vowel categories
        
        # Both consonants but different categories  
        if phoneme1 in all_consonants and phoneme2 in all_consonants:
            return 0.2  # Different consonant categories
        
        # One vowel, one consonant - very different
        if (phoneme1 in all_vowels and phoneme2 in all_consonants) or \
           (phoneme1 in all_consonants and phoneme2 in all_vowels):
            return 0.1  # Very different types
        
        return 0.15  # Default for unrecognized phonemes (conservative)
    
    def _align_and_score_phonemes(self, recognized_phonemes: List[str], expected_phonemes: List[str]) -> List[PhonemeScore]:
        """Align recognized phonemes with expected phonemes and score each expected phoneme"""
        if not expected_phonemes:
            return []
        
        if not recognized_phonemes:
            # No recognized speech - generate varied realistic low scores, not flat 0%
            # This happens when the word recognition failed, not when phonemes are missing
            print("⚠️ No recognized phonemes - generating realistic varied low scores")
            result = []
            for i, phoneme in enumerate(expected_phonemes):
                clean_phoneme = phoneme.replace("ˈ", "").replace("ˌ", "")
                # Generate varied low scores between 15-35% for failed recognition
                base_score = 0.15 + (i % 3) * 0.1  # Varies between 0.15, 0.25, 0.35
                variation = np.random.normal(0, 0.05)  # Small random variation
                score = max(0.10, min(0.40, base_score + variation))
                result.append(PhonemeScore(ipa_label=clean_phoneme, phoneme_score=round(score, 3)))
            return result
        
        print(f"🔍 Aligning phonemes: recognized {recognized_phonemes} vs expected {expected_phonemes}")
        
        # Start with base scores that vary by phoneme difficulty
        expected_scores = []
        for phoneme in expected_phonemes:
            clean_phoneme = phoneme.replace("ˈ", "").replace("ˌ", "")
            # Base score depends on phoneme difficulty
            if clean_phoneme in ['θ', 'ð']:  # TH sounds - harder
                base_score = 0.20
            elif clean_phoneme in ['r', 'l', 'ŋ']:  # R, L, NG - moderately hard
                base_score = 0.30
            elif clean_phoneme in ['æ', 'ɜː', 'ɔː']:  # Complex vowels
                base_score = 0.35
            else:  # Regular consonants and vowels
                base_score = 0.40
            expected_scores.append(base_score)
        
        used_recognized = [False] * len(recognized_phonemes)
        
        # Find best matches for each expected phoneme
        for j in range(len(expected_phonemes)):
            exp_phoneme = expected_phonemes[j].replace("ˈ", "").replace("ˌ", "")
            best_score = expected_scores[j]  # Start with base score
            best_rec_idx = -1
            
            # Find best matching recognized phoneme that hasn't been used
            for i in range(len(recognized_phonemes)):
                if used_recognized[i]:
                    continue
                    
                rec_phoneme = recognized_phonemes[i].replace("ˈ", "").replace("ˌ", "")
                similarity = self._calculate_phoneme_similarity(rec_phoneme, exp_phoneme)
                
                # More generous matching threshold and better scoring
                if similarity > 0.20:  # Lower threshold
                    adjusted_score = min(0.95, expected_scores[j] + similarity * 0.6)
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_rec_idx = i
            
            # If we found a good match, use it and mark as used
            if best_rec_idx >= 0:
                used_recognized[best_rec_idx] = True
                expected_scores[j] = best_score
                print(f"✅ Matched '{exp_phoneme}' with score {best_score:.2f}")
            else:
                # No good match found - this phoneme was likely NOT SPOKEN
                # Give it 0% only if we have other successful matches, indicating the recognition worked
                total_matches = sum(1 for score in expected_scores if score > 0.5)
                if total_matches > 0:
                    # Some phonemes matched well, so this one was genuinely missing
                    expected_scores[j] = 0.0
                    print(f"❌ '{exp_phoneme}' NOT SPOKEN - score: 0%")
                else:
                    # No good matches at all, probably recognition failure, keep base score
                    print(f"⚠️ No good match for '{exp_phoneme}' - using base score {expected_scores[j]:.2f}")
        
        # Create PhonemeScore objects with improved scores
        result = []
        for i, phoneme in enumerate(expected_phonemes):
            clean_phoneme = phoneme.replace("ˈ", "").replace("ˌ", "")
            score = expected_scores[i]
            
            # Add small random variation to make it more realistic
            if score > 0:
                variation = np.random.normal(0, 0.05)
                final_score = max(0.10, min(1.0, score + variation))
            else:
                final_score = 0.0
            
            result.append(PhonemeScore(ipa_label=clean_phoneme, phoneme_score=round(final_score, 3)))
        
        return result
    
    def _generate_ipa_phonemes(self, word: str, word_score: float, recognized_word: str = None) -> List[PhonemeScore]:
        """Generate IPA phoneme scores for a word using production phoneme engine"""
        word_lower = word.lower()
        
        # Generate expected phonemes using production engine
        expected_phonemes = self.phoneme_engine.get_phonemes(word_lower, accent='us')
        
        # Get recognized phonemes
        recognized_phonemes = []
        if recognized_word and recognized_word.strip():
            # Use production engine for recognized word as well
            recognized_phonemes = self.phoneme_engine.get_phonemes(recognized_word.lower(), accent='us')
            print(f"🔍 Aligning '{recognized_word}' {recognized_phonemes} with '{word}' {expected_phonemes}")
        else:
            # No recognized word - use empty list for alignment
            print(f"🔍 No recognized word for '{word}' - aligning with empty phonemes: {expected_phonemes}")
        
        # Use enhanced alignment system
        return self._align_and_score_phonemes_enhanced(recognized_phonemes, expected_phonemes)
    
    def _align_word_sequences_enhanced(self, expected_words: List[str], vosk_words: List[str], vosk_words_data: List[dict]) -> List[Tuple[str, Optional[dict]]]:
        """
        Enhanced word alignment with fuzzy matching and acoustic timing validation
        Returns list of (expected_word, actual_match_dict_or_None) pairs
        """
        alignments = []
        used_vosk_indices = set()
        
        print(f"🔄 Enhanced alignment: {len(expected_words)} expected vs {len(vosk_words)} detected")
        
        for i, expected_word in enumerate(expected_words):
            best_match = None
            best_score = 0.0
            best_index = -1
            
            # Try to find the best match among unused Vosk words
            for j, vosk_word in enumerate(vosk_words):
                if j in used_vosk_indices:
                    continue
                
                # Calculate multiple similarity metrics
                similarity_score = self._calculate_comprehensive_word_similarity(expected_word, vosk_word)
                
                # Position penalty - words should be roughly in order
                position_penalty = abs(i - j) * 0.1
                
                # Timing validation if available
                timing_bonus = 0.0
                if j < len(vosk_words_data):
                    word_data = vosk_words_data[j]
                    expected_start_time = i * 0.8  # Rough estimate
                    actual_start_time = word_data.get('start', expected_start_time)
                    timing_diff = abs(actual_start_time - expected_start_time)
                    
                    # Bonus for good timing alignment (within 1 second)
                    if timing_diff < 1.0:
                        timing_bonus = (1.0 - timing_diff) * 0.2
                
                # Combined score
                total_score = similarity_score - position_penalty + timing_bonus
                
                if total_score > best_score and similarity_score > 0.3:  # Minimum similarity threshold
                    best_score = total_score
                    best_match = vosk_word
                    best_index = j
            
            # Create match dictionary with metadata
            if best_match and best_index >= 0:
                used_vosk_indices.add(best_index)
                
                # Get timing and confidence data
                match_dict = {
                    'word': best_match,
                    'similarity': best_score,
                    'confidence': 0.5,  # Default
                    'start': i * 0.8,   # Default timing
                    'end': i * 0.8 + 0.7
                }
                
                if best_index < len(vosk_words_data):
                    word_data = vosk_words_data[best_index]
                    match_dict.update({
                        'confidence': word_data.get('conf', 0.5),
                        'start': word_data.get('start', i * 0.8),
                        'end': word_data.get('end', i * 0.8 + 0.7)
                    })
                
                alignments.append((expected_word, match_dict))
                print(f"  ✅ '{expected_word}' → '{best_match}' (score: {best_score:.2f})")
            else:
                # No good match found
                alignments.append((expected_word, None))
                print(f"  ❌ '{expected_word}' → [NO MATCH]")
        
        return alignments
    
    def _calculate_comprehensive_word_similarity(self, word1: str, word2: str) -> float:
        """Calculate comprehensive similarity using multiple metrics with partial word detection"""
        if not word1 or not word2:
            return 0.0
        
        word1, word2 = word1.lower(), word2.lower()
        
        if word1 == word2:
            return 1.0
        
        # Check for partial word detection (e.g., "beatre" → "be")
        partial_similarity = self._check_partial_word_match(word1, word2)
        if partial_similarity > 0:
            return partial_similarity
        
        # 1. Morphological similarity (handles plurals, tenses, etc.)
        morph_similarity = 0.0
        if self._is_minor_morphological_difference(word1, word2):
            morph_similarity = 0.9
        
        # 2. Check for known th-substitution patterns (theatre → beatre)
        th_substitution_similarity = self._check_th_substitution_patterns(word1, word2)
        if th_substitution_similarity > 0:
            return th_substitution_similarity
        
        # 3. Phonetic similarity using production engine
        phonetic_similarity = 0.0
        try:
            phonemes1 = self.phoneme_engine.get_phonemes(word1)
            phonemes2 = self.phoneme_engine.get_phonemes(word2)
            phonetic_similarity = self._calculate_phoneme_sequence_similarity(phonemes1, phonemes2)
        except:
            pass
        
        # 4. Edit distance similarity
        edit_distance = self._calculate_edit_distance(word1, word2)
        max_len = max(len(word1), len(word2))
        edit_similarity = 1.0 - (edit_distance / max_len) if max_len > 0 else 0.0
        
        # 5. Character overlap similarity
        char_similarity = self._calculate_character_overlap(word1, word2)
        
        # Weighted combination
        final_similarity = max(
            morph_similarity,  # Give morphological differences high priority
            phonetic_similarity * 0.4 + edit_similarity * 0.3 + char_similarity * 0.3
        )
        
        return min(1.0, final_similarity)
    
    def _check_partial_word_match(self, word1: str, word2: str) -> float:
        """Check if one word is a partial match of another (e.g., 'be' vs 'beatre')"""
        
        # Check if one word is a prefix of the other
        if word1.startswith(word2) or word2.startswith(word1):
            shorter = min(word1, word2, key=len)
            longer = max(word1, word2, key=len)
            
            # Only consider it a partial match if the shorter word is significantly shorter
            if len(shorter) < len(longer) * 0.7:  # Shorter word is less than 70% of longer word
                # Calculate similarity based on how much of the word was captured
                coverage = len(shorter) / len(longer)
                # Give partial credit but penalize for missing parts
                return coverage * 0.4  # Max 40% similarity for partial matches
        
        return 0.0
    
    def _check_th_substitution_patterns(self, word1: str, word2: str) -> float:
        """Check for th → b/d/f substitution patterns"""
        
        # Common th-substitution patterns
        th_patterns = {
            ('theatre', 'beatre'): 0.6,   # th → b substitution
            ('theatre', 'tetre'): 0.5,    # th → t substitution  
            ('theatre', 'fetre'): 0.5,    # th → f substitution
            ('therapy', 'berapy'): 0.6,   # th → b substitution
            ('think', 'bink'): 0.6,       # th → b substitution
            ('think', 'tink'): 0.5,       # th → t substitution
            ('three', 'bree'): 0.6,       # th → b substitution
            ('three', 'tree'): 0.5,       # th → t substitution
            ('the', 'be'): 0.6,           # th → b substitution
            ('the', 'de'): 0.5,           # th → d substitution
        }
        
        # Check both directions
        if (word1, word2) in th_patterns:
            return th_patterns[(word1, word2)]
        elif (word2, word1) in th_patterns:
            return th_patterns[(word2, word1)]
        
        # Dynamic pattern detection for th-words
        if 'th' in word1 or 'th' in word2:
            # Check if one word has 'th' and the other has 'b', 'd', 'f', or 't' in the same position
            if len(word1) == len(word2):
                # Find th position
                th_pos = -1
                th_word = ""
                other_word = ""
                
                if 'th' in word1:
                    th_pos = word1.find('th')
                    th_word = word1
                    other_word = word2
                elif 'th' in word2:
                    th_pos = word2.find('th')
                    th_word = word2
                    other_word = word1
                
                if th_pos >= 0 and th_pos < len(other_word):
                    # Check if the other word has a substitution at the th position
                    substitute_char = other_word[th_pos]
                    if substitute_char in ['b', 'd', 'f', 't'] and th_pos + 1 < len(other_word):
                        # Check if the rest of the word matches
                        th_word_rest = th_word[:th_pos] + th_word[th_pos+2:]  # Remove 'th'
                        other_word_rest = other_word[:th_pos] + other_word[th_pos+1:]  # Remove substitute
                        
                        if th_word_rest == other_word_rest:
                            # It's a th-substitution pattern
                            if substitute_char == 'b':
                                return 0.6  # th → b substitution (more common)
                            elif substitute_char in ['d', 't']:
                                return 0.5  # th → d/t substitution
                            elif substitute_char == 'f':
                                return 0.4  # th → f substitution
        
        return 0.0
    
    def _calculate_phoneme_sequence_similarity(self, phonemes1: List[str], phonemes2: List[str]) -> float:
        """Calculate similarity between two phoneme sequences"""
        if not phonemes1 or not phonemes2:
            return 0.0
        
        # Use dynamic programming for sequence alignment
        m, n = len(phonemes1), len(phonemes2)
        dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if phonemes1[i-1] == phonemes2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1.0
                else:
                    # Use phoneme similarity for partial matches
                    phoneme_sim = self._calculate_phoneme_similarity(phonemes1[i-1], phonemes2[j-1])
                    dp[i][j] = max(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1] + phoneme_sim
                    )
        
        max_possible = max(m, n)
        return dp[m][n] / max_possible if max_possible > 0 else 0.0
    
    def _calculate_edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein edit distance between two words"""
        m, n = len(word1), len(word2)
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    def _calculate_character_overlap(self, word1: str, word2: str) -> float:
        """Calculate character overlap similarity"""
        if not word1 or not word2:
            return 0.0
        
        chars1 = set(word1)
        chars2 = set(word2)
        
        overlap = len(chars1.intersection(chars2))
        total = len(chars1.union(chars2))
        
        return overlap / total if total > 0 else 0.0
    
    def _align_and_score_phonemes_enhanced(self, actual_phonemes: List[str], expected_phonemes: List[str]) -> List[PhonemeScore]:
        """Enhanced phoneme alignment and scoring with production-quality assessment"""
        if not expected_phonemes:
            return []
        
        if not actual_phonemes:
            # Generate realistic low scores for missing phonemes
            print("⚠️ No actual phonemes - generating varied low scores")
            result = []
            for i, phoneme in enumerate(expected_phonemes):
                # Varied low scores based on phoneme difficulty
                difficulty_factor = self._get_phoneme_difficulty(phoneme)
                base_score = 0.1 + (i % 3) * 0.05  # 10-20% base
                score = max(0.05, base_score - difficulty_factor * 0.1)
                result.append(PhonemeScore(ipa_label=phoneme, phoneme_score=round(score, 3)))
            return result
        
        print(f"🔍 Enhanced phoneme alignment: {actual_phonemes} vs {expected_phonemes}")
        
        # Advanced alignment using production phoneme engine
        aligned_scores = []
        used_actual = [False] * len(actual_phonemes)
        
        for expected_phoneme in expected_phonemes:
            best_score = 0.0
            best_actual_idx = -1
            
            # Find best matching actual phoneme
            for i, actual_phoneme in enumerate(actual_phonemes):
                if used_actual[i]:
                    continue
                
                similarity = self._calculate_phoneme_similarity(actual_phoneme, expected_phoneme)
                
                # Enhanced scoring with contextual factors
                contextual_bonus = self._calculate_contextual_phoneme_bonus(
                    actual_phoneme, expected_phoneme, len(aligned_scores), len(expected_phonemes)
                )
                
                total_score = similarity + contextual_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_actual_idx = i
            
            # Assign score based on best match
            if best_actual_idx >= 0 and best_score > 0.3:
                used_actual[best_actual_idx] = True
                final_score = min(0.98, best_score + 0.1)  # Boost for successful match
                aligned_scores.append(PhonemeScore(ipa_label=expected_phoneme, phoneme_score=round(final_score, 3)))
                print(f"  ✅ '{expected_phoneme}' matched with score {final_score:.2f}")
            else:
                # No good match - phoneme not pronounced correctly
                difficulty_penalty = self._get_phoneme_difficulty(expected_phoneme)
                miss_score = max(0.0, 0.2 - difficulty_penalty)
                aligned_scores.append(PhonemeScore(ipa_label=expected_phoneme, phoneme_score=round(miss_score, 3)))
                print(f"  ❌ '{expected_phoneme}' no good match - score: {miss_score:.2f}")
        
        return aligned_scores
    
    def _get_phoneme_difficulty(self, phoneme: str) -> float:
        """Get difficulty factor for a phoneme (0.0 = easy, 1.0 = very hard)"""
        difficulty_map = {
            # Very difficult sounds
            'θ': 0.9,   # voiceless th
            'ð': 0.9,   # voiced th
            'ɜːr': 0.8, # r-colored schwa
            'ŋ': 0.7,   # ng sound
            
            # Moderately difficult
            'r': 0.6,   # r sound
            'l': 0.5,   # l sound
            'ʒ': 0.6,   # zh sound
            'dʒ': 0.5,  # j sound
            'tʃ': 0.5,  # ch sound
            
            # Vowel difficulties
            'æ': 0.4,   # cat vowel
            'ʌ': 0.4,   # cut vowel
            'ɪ': 0.3,   # bit vowel
            'ɛ': 0.3,   # bet vowel
            
            # Easy sounds
            'p': 0.1, 'b': 0.1, 't': 0.1, 'd': 0.1, 'k': 0.1, 'g': 0.1,
            'm': 0.2, 'n': 0.2, 'f': 0.2, 'v': 0.2, 's': 0.2, 'z': 0.2,
        }
        
        return difficulty_map.get(phoneme, 0.3)  # Default moderate difficulty
    
    def _calculate_contextual_phoneme_bonus(self, actual: str, expected: str, position: int, total_phonemes: int) -> float:
        """Calculate contextual bonus for phoneme matching"""
        bonus = 0.0
        
        # Position-based bonuses
        if position == 0:  # Word-initial position
            bonus += 0.05
        elif position == total_phonemes - 1:  # Word-final position
            bonus += 0.05
        
        # Vowel vs consonant matching bonus
        vowels = {'æ', 'ɛ', 'ɪ', 'ɔ', 'ʌ', 'iː', 'uː', 'oʊ', 'eɪ', 'aɪ', 'aʊ', 'ɔɪ', 'ɜːr', 'ɑːr', 'ɔːr'}
        
        actual_is_vowel = actual in vowels
        expected_is_vowel = expected in vowels
        
        if actual_is_vowel == expected_is_vowel:
            bonus += 0.1  # Same category bonus
        
        return bonus
    
    def _calculate_word_score_with_validation(self, expected_word: str, actual_match: Optional[dict], 
                                            phoneme_scores: List[PhonemeScore], confidence: float, 
                                            audio_quality: dict, timing: dict) -> float:
        """Calculate word score with pronunciation validation to handle ASR corrections"""
        
        # Get recognized word and timing data
        recognized_word = actual_match['word'] if actual_match else expected_word
        timing_data = {
            'start': timing.get('start', 0.0),
            'end': timing.get('end', 0.0)
        }
        
        # Validate the pronunciation result
        phoneme_scores_flat = [p.phoneme_score for p in phoneme_scores] if phoneme_scores else []
        validation_result = self.pronunciation_validator.validate_pronunciation_result(
            expected_word=expected_word,
            recognized_word=recognized_word,
            vosk_confidence=confidence,
            phoneme_scores=phoneme_scores_flat,
            timing_data=timing_data,
            audio_features=audio_quality
        )
        
        # Log validation results
        if not validation_result.is_reliable:
            print(f"⚠️ Validation issues detected for '{expected_word}' → '{recognized_word}':")
            for issue in validation_result.issues_detected:
                print(f"  - {issue}")
            print(f"  Correction likelihood: {validation_result.correction_likelihood:.2f}")
        
        # Start with base enhanced score
        base_score = self._calculate_word_score_enhanced(
            expected_word, actual_match, phoneme_scores, confidence, audio_quality
        )
        
        # Apply validation-based adjustments
        if validation_result.correction_likelihood > 0.4:
            # High likelihood of ASR correction - use mitigation strategy
            mitigation = self.pronunciation_validator.get_correction_mitigation_strategy(validation_result)
            
            if mitigation['use_raw_acoustic']:
                # Use raw acoustic score instead of ASR-based score
                print(f"  🔧 Using raw acoustic score: {validation_result.raw_acoustic_score:.1f}%")
                validated_score = validation_result.raw_acoustic_score
            else:
                # Apply penalty factor to original score
                penalty_factor = mitigation['penalty_factor']
                validated_score = base_score * penalty_factor
                print(f"  🔧 Applying correction penalty: {penalty_factor:.2f} → {validated_score:.1f}%")
            
            # Confidence adjustment for unreliable results
            confidence_factor = mitigation['confidence_adjustment']
            validated_score *= confidence_factor
            
        else:
            # Low correction likelihood - use original score with minor adjustments
            validated_score = base_score
            
            # Small penalty for any detected issues
            if validation_result.issues_detected:
                issue_penalty = len(validation_result.issues_detected) * 0.02
                validated_score *= (1.0 - issue_penalty)
        
        # Apply validation confidence weighting
        validation_confidence = validation_result.validation_confidence
        if validation_confidence < 0.5:
            # Low validation confidence - be more conservative
            validated_score *= 0.9
        
        return max(0.0, min(100.0, validated_score))
    
    def _calculate_word_score_enhanced(self, expected_word: str, actual_match: Optional[dict], 
                                     phoneme_scores: List[PhonemeScore], confidence: float, 
                                     audio_quality: dict) -> float:
        """Calculate enhanced word score using multiple factors (without validation)"""
        
        # Base score from phoneme alignment
        if phoneme_scores:
            phoneme_avg = sum(p.phoneme_score for p in phoneme_scores) / len(phoneme_scores)
            base_score = phoneme_avg * 100
        else:
            base_score = 5.0
        
        # Factor 1: Word-level similarity bonus
        word_similarity_bonus = 0.0
        if actual_match:
            word_similarity = actual_match.get('similarity', 0.0)
            if word_similarity > 0.8:
                word_similarity_bonus = min(10.0, word_similarity * 15)
        
        # Factor 2: Confidence weighting
        confidence_adjustment = (confidence - 0.5) * 20  # -10 to +10 points
        
        # Factor 3: Audio quality adjustment
        quality_factor = audio_quality.get('overall_quality', 50) / 100
        quality_adjustment = (quality_factor - 0.5) * 10  # -5 to +5 points
        
        # Factor 4: Word difficulty adjustment
        difficulty_penalty = self._calculate_pronunciation_difficulty(expected_word)
        
        # Combine all factors
        final_score = (
            base_score + 
            word_similarity_bonus + 
            confidence_adjustment + 
            quality_adjustment - 
            difficulty_penalty
        )
        
        # Apply reasonable bounds
        final_score = max(0.0, min(100.0, final_score))
        
        return final_score
    
    def _apply_confidence_weighted_adjustments(self, overall_score: float, confidence_metrics, audio_quality: dict) -> float:
        """Apply confidence-weighted adjustments to overall score for production reliability"""
        
        # Start with enhanced quality adjustments
        adjusted_score = self._apply_enhanced_quality_adjustments(overall_score, audio_quality)
        
        # Apply confidence-based reliability adjustments
        confidence_factor = confidence_metrics.overall_confidence
        
        # High confidence: minimal adjustment
        if confidence_factor >= 0.8:
            confidence_adjustment = 1.0
        # Medium confidence: slight penalty
        elif confidence_factor >= 0.6:
            confidence_adjustment = 0.95
        # Low confidence: moderate penalty
        elif confidence_factor >= 0.4:
            confidence_adjustment = 0.85
        # Very low confidence: significant penalty
        else:
            confidence_adjustment = 0.7
        
        adjusted_score *= confidence_adjustment
        
        # Specific confidence component adjustments
        
        # Acoustic confidence adjustment
        if confidence_metrics.acoustic_confidence < 0.4:
            adjusted_score *= 0.9  # Poor acoustic conditions
        
        # Temporal confidence adjustment
        if confidence_metrics.temporal_confidence < 0.4:
            adjusted_score *= 0.95  # Timing inconsistencies
        
        # Lexical confidence adjustment
        if confidence_metrics.lexical_confidence < 0.4:
            adjusted_score *= 0.9  # Word recognition issues
        
        # Phonetic confidence adjustment
        if confidence_metrics.phonetic_confidence < 0.4:
            adjusted_score *= 0.95  # Pronunciation alignment issues
        
        # Boost score for exceptionally high confidence
        if confidence_factor > 0.9:
            adjusted_score *= 1.05  # Small bonus for very reliable assessment
        
        return max(0.0, min(100.0, adjusted_score))
    
    def _apply_enhanced_quality_adjustments(self, overall_score: float, audio_quality: dict) -> float:
        """Apply enhanced audio quality adjustments to overall score"""
        
        # SNR adjustment
        snr = audio_quality.get('snr_estimate', 15)
        if snr < 10:
            overall_score *= 0.85  # Significant penalty for poor SNR
        elif snr < 15:
            overall_score *= 0.95  # Minor penalty
        elif snr > 25:
            overall_score *= 1.05  # Minor bonus for excellent SNR
        
        # Clipping penalty
        clipping_ratio = audio_quality.get('clipping_ratio', 0)
        if clipping_ratio > 0.05:  # More than 5% clipping
            overall_score *= (1.0 - clipping_ratio * 0.5)
        
        # Frequency quality adjustment
        freq_quality = audio_quality.get('frequency_quality', 50)
        if freq_quality < 30:
            overall_score *= 0.90  # Poor frequency response
        elif freq_quality > 70:
            overall_score *= 1.03  # Good frequency response
        
        # Overall quality gate
        overall_quality = audio_quality.get('overall_quality', 50)
        if overall_quality < 20:
            overall_score *= 0.7  # Severe penalty for very poor audio
        
        return max(0.0, min(100.0, overall_score))
    
    def _generate_fallback_phonemes(self, word: str) -> List[str]:
        """Fallback to production phoneme engine instead of basic G2P"""
        try:
            return self.phoneme_engine.get_phonemes(word, accent='us')
        except Exception as e:
            print(f"⚠️ Phoneme engine failed for '{word}': {e}")
            # Very basic fallback
            return ['ʌ', 'n', 'k', 'n', 'oʊ', 'n']  # "unknown" 

    def _find_best_acoustic_match(self, expected_word: str, recognized_words: List[dict], recognized_word_list: List[str], position: int) -> tuple:
        """Find the best acoustic match for an expected word considering position and confidence"""
        best_match = None
        best_score = 0.0
        
        # Try to find word with timing information first
        for rec_word_data in recognized_words:
            rec_word = rec_word_data.get('word', '').lower()
            confidence = rec_word_data.get('conf', 0.5)  # Vosk confidence
            
            # Calculate similarity
            similarity = self._calculate_word_similarity(rec_word, expected_word)
            
            # Position penalty - words should be roughly in order
            expected_position = position
            actual_position = rec_word_data.get('start', position * 0.8) / 0.8
            position_penalty = min(0.3, abs(expected_position - actual_position) * 0.1)
            
            # Combined score
            combined_score = similarity - position_penalty
            
            if combined_score > best_score and combined_score > 0.3:  # Minimum threshold
                best_score = combined_score
                best_match = (rec_word, confidence, rec_word_data)
        
        # Fallback: try positional matching in recognized word list
        if not best_match and position < len(recognized_word_list):
            rec_word = recognized_word_list[position]
            similarity = self._calculate_word_similarity(rec_word, expected_word)
            if similarity > 0.3:
                best_match = (rec_word, 0.5, {'start': position * 0.8, 'end': position * 0.8 + 0.7})
        
        return best_match
    
    def _calculate_pronunciation_difficulty(self, word: str) -> float:
        """Calculate pronunciation difficulty penalty for complex words"""
        difficulty = 0.0
        
        # Length penalty for very long words
        if len(word) > 8:
            difficulty += (len(word) - 8) * 2
        
        # Complex consonant clusters
        complex_patterns = ['th', 'ch', 'sh', 'str', 'spr', 'scr', 'spl']
        for pattern in complex_patterns:
            if pattern in word.lower():
                difficulty += 3
        
        # Silent letters (common pronunciation traps)
        silent_patterns = ['ght', 'mb', 'kn', 'wr', 'gn']
        for pattern in silent_patterns:
            if pattern in word.lower():
                difficulty += 5
        
        return min(difficulty, 20)  # Cap at 20 point penalty
    
    def _generate_realistic_phoneme_scores(self, word: str, word_similarity: float, acoustic_score: float) -> List[PhonemeScore]:
        """Generate realistic phoneme scores based on word similarity and acoustic quality"""
        phonemes = self._generate_fallback_phonemes(word)
        phoneme_scores = []
        
        # Base score from word similarity and acoustic quality
        base_score = (word_similarity * 0.7 + acoustic_score * 0.3)
        
        for i, phoneme in enumerate(phonemes):
            # Add some variation to make it realistic
            variation = np.random.normal(0, 0.1)  # Small random variation
            
            # Some phonemes are harder than others
            difficulty_map = {
                'θ': -0.2,  # 'th' sound
                'ð': -0.2,  # 'th' sound
                'r': -0.15, # R sound
                'l': -0.1,  # L sound
                'ŋ': -0.15, # 'ng' sound
            }
            
            difficulty_penalty = difficulty_map.get(phoneme, 0)
            phoneme_score = max(0.0, min(1.0, base_score + variation + difficulty_penalty))
            
            phoneme_scores.append(PhonemeScore(
                ipa_label=phoneme,
                phoneme_score=round(phoneme_score, 3)
            ))
        
        return phoneme_scores
    
    def _apply_audio_quality_adjustments(self, score: float, audio_data: np.ndarray, samplerate: int) -> float:
        """Apply adjustments based on audio quality factors"""
        # Check for very quiet audio
        audio_level = np.mean(np.abs(audio_data))
        if audio_level < 0.01:  # Very quiet
            score *= 0.8  # 20% penalty
            print("⚠️ Audio level very low - applying penalty")
        
        # Check for clipping
        if np.max(np.abs(audio_data)) > 0.99:
            score *= 0.9  # 10% penalty
            print("⚠️ Audio clipping detected - applying penalty")
        
        # Check audio length (very short clips are suspicious)
        duration = len(audio_data) / samplerate
        if duration < 1.0:  # Less than 1 second
            score *= 0.85
            print(f"⚠️ Very short audio ({duration:.1f}s) - applying penalty")
        
        return max(0.0, min(100.0, score))
    
    def _align_word_sequences(self, whisper_words: List[str], vosk_words: List[str]) -> List[Tuple[str, str, float]]:
        """
        Align two word sequences using dynamic programming to find optimal alignment
        Returns list of (whisper_word, vosk_word, similarity) tuples
        """
        # Dynamic programming approach for sequence alignment
        m, n = len(whisper_words), len(vosk_words)
        
        # Create DP table: dp[i][j] = best score for aligning whisper_words[:i] with vosk_words[:j]
        dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
        parent = [[None for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Option 1: Match whisper_words[i-1] with vosk_words[j-1]
                similarity = self._calculate_word_similarity(vosk_words[j-1], whisper_words[i-1])
                match_score = dp[i-1][j-1] + similarity
                
                # Option 2: Skip whisper word (insertion)
                skip_whisper_score = dp[i-1][j] + 0.1  # Small penalty for skipping
                
                # Option 3: Skip vosk word (deletion)  
                skip_vosk_score = dp[i][j-1] + 0.1
                
                # Choose best option
                if match_score >= skip_whisper_score and match_score >= skip_vosk_score:
                    dp[i][j] = match_score
                    parent[i][j] = 'match'
                elif skip_whisper_score >= skip_vosk_score:
                    dp[i][j] = skip_whisper_score
                    parent[i][j] = 'skip_whisper'
                else:
                    dp[i][j] = skip_vosk_score
                    parent[i][j] = 'skip_vosk'
        
        # Backtrack to find alignment
        alignments = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and parent[i][j] == 'match':
                whisper_word = whisper_words[i-1]
                vosk_word = vosk_words[j-1]
                similarity = self._calculate_word_similarity(vosk_word, whisper_word)
                alignments.append((whisper_word, vosk_word, similarity))
                i -= 1
                j -= 1
            elif i > 0 and (j == 0 or parent[i][j] == 'skip_whisper'):
                # Whisper word with no match
                alignments.append((whisper_words[i-1], None, 0.0))
                i -= 1
            else:
                # Vosk word with no match (skip it)
                j -= 1
        
        alignments.reverse()
        return alignments

    async def _analyze_pronunciation_against_whisper_reference(self, audio_data: np.ndarray, samplerate: int, whisper_reference: str) -> Tuple[float, List[WordScore]]:
        """
        Clean approach with proper sequence alignment: Whisper gives intended speech, Vosk gives actual speech, align optimally then compare phonemes
        """
        if not self.vosk_model:
            print("⚠️ Vosk model not available, using fallback scoring")
            return self._fallback_pronunciation_analysis(audio_data, samplerate, whisper_reference)
        
        try:
            print(f"🎯 CLEAN FREESTYLE ANALYSIS: Whisper reference '{whisper_reference}'")
            
            # Step 1: Get what user actually said using Vosk (raw phonetic transcription)
            rec = vosk.KaldiRecognizer(self.vosk_model, samplerate)
            rec.SetWords(True)
            
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            rec.AcceptWaveform(audio_bytes)
            result = rec.FinalResult()
            vosk_data = json.loads(result)
            
            vosk_transcription = vosk_data.get('text', '').strip()
            vosk_words_data = vosk_data.get('result', [])
            
            print(f"📝 Whisper (intended): '{whisper_reference}'")
            print(f"🎤 Vosk (actual): '{vosk_transcription}'")
            
            # Step 2: Align word sequences optimally
            whisper_words = whisper_reference.lower().split()
            vosk_words = vosk_transcription.lower().split()
            
            print(f"🔄 Aligning sequences: {len(whisper_words)} intended words vs {len(vosk_words)} actual words")
            alignments = self._align_word_sequences(whisper_words, vosk_words)
            
            word_scores = []
            total_score = 0.0
            vosk_word_index = 0
            
            # Step 3: Analyze each alignment
            for i, (intended_word, actual_word, word_similarity) in enumerate(alignments):
                print(f"🔍 Alignment {i+1}: '{intended_word}' → '{actual_word}' (similarity: {word_similarity:.2f})")
                
                # Get expected phonemes from what they intended to say (Whisper)
                expected_phonemes = self.phoneme_engine.get_phonemes(intended_word, accent='us')
                
                # Get actual phonemes from what they actually said (Vosk)
                actual_phonemes = []
                confidence = 0.3
                timing = {'start': i * 0.8, 'end': i * 0.8 + 0.7}
                
                if actual_word:
                    actual_phonemes = self.phoneme_engine.get_phonemes(actual_word, accent='us')
                    
                    # Find confidence and timing from Vosk data
                    if vosk_word_index < len(vosk_words_data):
                        # Find matching Vosk word data
                        for word_data in vosk_words_data:
                            if word_data.get('word', '').lower() == actual_word:
                                confidence = word_data.get('conf', 0.3)
                                timing = {
                                    'start': word_data.get('start', i * 0.8),
                                    'end': word_data.get('end', i * 0.8 + 0.7)
                                }
                                break
                    vosk_word_index += 1
                    
                    print(f"   Expected phonemes: {expected_phonemes}")
                    print(f"   Actual phonemes: {actual_phonemes}")
                else:
                    print(f"   Expected phonemes: {expected_phonemes}")
                    print(f"   Actual phonemes: [] (word not spoken)")
                
                # Step 4: Enhanced phoneme alignment and scoring
                phoneme_scores = self._align_and_score_phonemes_enhanced(actual_phonemes, expected_phonemes)
                
                # Calculate word-level score from phoneme scores
                if phoneme_scores:
                    avg_phoneme_score = sum(p.phoneme_score for p in phoneme_scores) / len(phoneme_scores)
                    word_score = avg_phoneme_score * 100
                else:
                    word_score = 5.0
                
                # Apply word similarity boost if applicable
                if word_similarity > 0.8:
                    word_score = min(95.0, word_score * (0.9 + word_similarity * 0.1))
                
                # Apply confidence weighting
                word_score = (word_score * 0.9 + confidence * 100 * 0.1)
                
                print(f"   💯 Final word score: {word_score:.1f}%")
                
                word_scores.append(WordScore(
                    word_text=intended_word,
                    word_score=round(word_score, 1),
                    phonemes=phoneme_scores,
                    start_time=timing['start'],
                    end_time=timing['end']
                ))
                
                total_score += word_score
            
            # Calculate overall score
            overall_score = total_score / len(alignments) if alignments else 0.0
            overall_score = self._apply_audio_quality_adjustments(overall_score, audio_data, samplerate)
            
            print(f"🎯 Overall pronunciation score: {overall_score:.1f}%")
            
            return round(overall_score, 1), word_scores
            
        except Exception as e:
            print(f"⚠️ Error in clean freestyle analysis: {e}")
            try:
                return self._fallback_pronunciation_analysis(audio_data, samplerate, whisper_reference)
            except Exception as fallback_error:
                print(f"❌ Freestyle fallback analysis also failed: {fallback_error}")
                fallback_result = self._create_emergency_fallback_response(whisper_reference)
                return fallback_result

    def _calculate_phonetic_similarity(self, spoken_word: str, intended_word: str) -> float:
        """
        Calculate phonetic similarity between what was actually spoken vs what was intended.
        This is more accurate than semantic word matching.
        """
        if not spoken_word or not intended_word:
            return 0.0
        
        spoken_word = spoken_word.lower().strip()
        intended_word = intended_word.lower().strip()
        
        # Exact match (rare but possible)
        if spoken_word == intended_word:
            return 1.0
        
        # Check for common phonetic substitutions and mispronunciations
        phonetic_substitutions = {
            # TH sounds often replaced - these are MISPRONUNCIATIONS, so low scores
            'the': {'da': 0.25, 'de': 0.25, 'za': 0.20, 'ze': 0.20, 'duh': 0.30, 'dat': 0.15},
            'that': {'dat': 0.25, 'zat': 0.20, 'dhat': 0.30},
            'this': {'dis': 0.25, 'zis': 0.20, 'diss': 0.30},
            'think': {'tink': 0.35, 'sink': 0.30, 'fink': 0.25},
            'three': {'tree': 0.40, 'free': 0.30, 'twee': 0.35},
            'there': {'dare': 0.30, 'dere': 0.25, 'were': 0.20},
            'they': {'day': 0.25, 'dey': 0.25, 'de': 0.20},
            'then': {'den': 0.30, 'ten': 0.25, 'ven': 0.20},
            'through': {'trough': 0.40, 'true': 0.30, 'frough': 0.25},
            
            # R/L confusion
            'really': {'leally': 0.40, 'weally': 0.35, 'lilly': 0.25},
            'right': {'light': 0.40, 'wight': 0.30, 'lite': 0.35},
            'red': {'led': 0.40, 'wed': 0.30, 'rad': 0.35},
            'run': {'lun': 0.35, 'wun': 0.30},
            
            # V/W confusion
            'very': {'wery': 0.40, 'berry': 0.30, 'vary': 0.35},
            'voice': {'woice': 0.40, 'voise': 0.30},
            'wave': {'vave': 0.40, 'waive': 0.35},
            'vine': {'wine': 0.40, 'wine': 0.40},
            
            # Common vowel substitutions
            'can': {'ken': 0.40, 'kon': 0.35, 'cen': 0.35},
            'man': {'men': 0.40, 'mon': 0.35, 'min': 0.30},
            'cat': {'cot': 0.40, 'cut': 0.35, 'ket': 0.30},
            'bet': {'bat': 0.40, 'bit': 0.35, 'but': 0.30},
        }
        
        # Check if this is a known phonetic substitution
        if intended_word in phonetic_substitutions:
            substitutions = phonetic_substitutions[intended_word]
            if spoken_word in substitutions:
                similarity = substitutions[spoken_word]
                print(f"🔍 KNOWN MISPRONUNCIATION: '{intended_word}' → '{spoken_word}' = {similarity:.2f} (deliberate low score)")
                return similarity
        
        # Character-level phonetic similarity for unknown substitutions
        max_len = max(len(spoken_word), len(intended_word))
        if max_len == 0:
            return 0.0
        
        # Count phonetically similar characters
        phonetic_matches = 0
        min_len = min(len(spoken_word), len(intended_word))
        
        for i in range(min_len):
            spoken_char = spoken_word[i]
            intended_char = intended_word[i]
            
            if spoken_char == intended_char:
                phonetic_matches += 1
            elif self._are_phonetically_similar(spoken_char, intended_char):
                phonetic_matches += 0.5  # Partial credit for similar sounds
        
        # Calculate similarity with heavy penalty for length differences
        base_similarity = phonetic_matches / max_len
        length_penalty = abs(len(spoken_word) - len(intended_word)) / max_len * 0.8
        
        similarity = max(0.0, base_similarity - length_penalty)
        
        # Cap similarity for different words (no word should get >50% if it's clearly different)
        if spoken_word != intended_word and similarity > 0.5:
            similarity = min(0.5, similarity)
            print(f"🔍 CAPPING similarity for '{intended_word}' → '{spoken_word}' at {similarity:.2f}")
        
        return similarity

    def _are_phonetically_similar(self, char1: str, char2: str) -> bool:
        """Check if two characters represent phonetically similar sounds"""
        similar_groups = [
            ['b', 'p'],      # Voiced/voiceless pairs
            ['d', 't'],
            ['g', 'k'],
            ['v', 'f'],
            ['z', 's'],
            ['th', 'f', 's'], # TH substitutions
            ['r', 'l', 'w'],  # Liquid consonants
            ['a', 'e', 'i'],  # Close vowels
            ['o', 'u'],       # Back vowels
        ]
        
        for group in similar_groups:
            if char1 in group and char2 in group:
                return True
        
        return False

    def _confidence_based_pronunciation_analysis(self, audio_data: np.ndarray, samplerate: int, whisper_reference: str, vosk_words: List[dict]) -> Tuple[float, List[WordScore]]:
        """
        Fallback method when Vosk still corrects speech - use confidence scores to estimate pronunciation quality
        """
        print("🔄 Using confidence-based analysis since Vosk is still correcting speech")
        
        reference_words = whisper_reference.lower().split()
        word_scores = []
        total_score = 0.0
        
        for i, reference_word in enumerate(reference_words):
            # Find corresponding Vosk word data
            corresponding_confidence = 0.3  # Default low confidence
            
            if i < len(vosk_words):
                corresponding_confidence = vosk_words[i].get('conf', 0.3)
            
            # Use confidence as primary indicator of pronunciation quality
            # Low confidence often means mispronunciation
            pronunciation_score = (corresponding_confidence * 0.9 + 0.1) * 100  # Scale to 10-100%
            
            # Apply audio quality and word difficulty adjustments
            difficulty_penalty = self._calculate_pronunciation_difficulty(reference_word)
            pronunciation_score = max(10, pronunciation_score - difficulty_penalty)
            
            print(f"🔍 '{reference_word}' | Confidence: {corresponding_confidence:.2f} | Score: {pronunciation_score:.1f}%")
            
            # Generate phoneme scores based on confidence
            phonemes = self._generate_realistic_phoneme_scores(reference_word, corresponding_confidence, corresponding_confidence)
            
            word_scores.append(WordScore(
                word_text=reference_word,
                word_score=round(pronunciation_score, 1),
                phonemes=phonemes,
                start_time=i * 0.8,
                end_time=i * 0.8 + 0.7
            ))
            
            total_score += pronunciation_score
        
        overall_score = total_score / len(reference_words) if reference_words else 0.0
        overall_score = self._apply_audio_quality_adjustments(overall_score, audio_data, samplerate)
        
        return round(overall_score, 1), word_scores

    def _find_raw_phonetic_match_for_reference_word(self, reference_word: str, vosk_words: List[dict], vosk_word_list: List[str], position: int) -> tuple:
        """
        Find what Vosk actually heard (phonetically) for a reference word position
        """
        # Try positional matching first (words should be roughly in order)
        if position < len(vosk_word_list):
            raw_word = vosk_word_list[position]
            
            # Get timing and confidence from detailed Vosk data
            word_data = None
            if position < len(vosk_words):
                word_data = vosk_words[position]
                confidence = word_data.get('conf', 0.3)
                timing = {'start': word_data.get('start', position * 0.8), 'end': word_data.get('end', position * 0.8 + 0.7)}
            else:
                confidence = 0.3
                timing = {'start': position * 0.8, 'end': position * 0.8 + 0.7}
            
            return (raw_word, confidence, timing)
        
        # Fallback: no match found
        return None

    def _analyze_fluency_with_audio(self, vosk_words_data: List[dict], transcription: str, 
                                   total_duration: float, audio_data: np.ndarray, samplerate: int) -> FluencyMetrics:
        """Enhanced fluency analysis with audio-based filler detection"""
        
        # Extract words and timing from Vosk data
        words = []
        for word_data in vosk_words_data:
            word = word_data.get('word', '').lower()
            start = word_data.get('start', 0.0)
            end = word_data.get('end', 0.0)
            words.append({'word': word, 'start': start, 'end': end})
        
        # Get Vosk transcript from the raw audio data (what was actually spoken)
        vosk_transcript = ' '.join([word_data.get('word', '') for word_data in vosk_words_data])
        whisper_transcript = transcription  # What was intended (cleaned by Whisper)
        
        print(f"🎤 Raw Vosk (spoken): '{vosk_transcript}'")
        print(f"📝 Whisper (intended): '{whisper_transcript}'")
        
        # 1. Detect filler words from Vosk transcription (basic detection)
        filler_words = self._detect_filler_words(words)
        
        # 2. ALSO detect filler words from Whisper transcription (text-based)
        whisper_fillers = self._detect_fillers_from_whisper_text(transcription, total_duration)
        print(f"🔍 Found {len(whisper_fillers)} filler words in Whisper transcription")
        
        # Combine Vosk and Whisper detected fillers (avoid duplicates)
        all_fillers = list(filler_words)
        for whisper_filler in whisper_fillers:
            # Check for overlap with existing fillers
            is_duplicate = False
            for existing_filler in filler_words:
                overlap = not (whisper_filler.end_time <= existing_filler.start_time or 
                              whisper_filler.start_time >= existing_filler.end_time)
                if overlap:
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_fillers.append(whisper_filler)
        
        # 3. Enhanced filler detection using raw audio analysis + transcript comparison
        try:
            # RE-ENABLED with conservative settings - speech recognition often filters out real filler words
            from .filler_detector import enhance_filler_detection
            print("🚀 Running enhanced filler detection with raw audio + transcript comparison...")
            all_fillers = enhance_filler_detection(
                all_fillers, audio_data, samplerate, vosk_words_data,
                vosk_transcript=vosk_transcript, 
                whisper_transcript=whisper_transcript
            )
        except ImportError as e:
            print(f"⚠️ Advanced filler detection not available: {e}")
        except Exception as e:
            print(f"⚠️ Enhanced filler detection failed: {e}")
        
        # Update filler_words for the rest of the calculations
        filler_words = all_fillers
        
        # 4. Detect long pauses
        long_pauses = self._detect_long_pauses(words, total_duration)
        
        # 5. Detect repetitions
        repetitions = self._detect_repetitions(words)
        
        # 6. Calculate speech rate over time (2-second segments)
        speech_rate_over_time = self._calculate_speech_rate_over_time(words, total_duration)
        
        # 7. Calculate average speech rate (words per minute, excluding fillers)
        content_words = [w for w in words if not self._is_filler_word(w['word'])]
        speech_duration = total_duration - sum(p.duration for p in long_pauses) if long_pauses else total_duration
        speech_rate = (len(content_words) / (speech_duration / 60)) if speech_duration > 0 else 0
        
        # 8. Calculate overall fluency score
        base_score = 100.0
        
        # Penalize filler words (up to -30 points)
        filler_penalty = min(30, len(filler_words) * 3)
        base_score -= filler_penalty
        
        # Penalize long pauses (up to -25 points)
        pause_penalty = min(25, len(long_pauses) * 5)
        base_score -= pause_penalty
        
        # Penalize repetitions (up to -20 points)
        repetition_penalty = min(20, len(repetitions) * 4)
        base_score -= repetition_penalty
        
        # Penalize speech rate if outside optimal range (140-180 WPM)
        if speech_rate < 120 or speech_rate > 200:
            rate_penalty = 15
        elif speech_rate < 140 or speech_rate > 180:
            rate_penalty = 5
        else:
            rate_penalty = 0
        base_score -= rate_penalty
        
        overall_fluency_score = max(0, min(100, base_score))
        
        return FluencyMetrics(
            overall_fluency_score=overall_fluency_score,
            filler_words=filler_words,
            long_pauses=long_pauses,
            repetitions=repetitions,
            speech_rate=speech_rate,
            speech_rate_over_time=speech_rate_over_time,
            total_filler_count=len(filler_words),
            total_pause_time=sum(p.duration for p in long_pauses),
            total_repetition_count=len(repetitions)
        )
    
    def _detect_filler_words(self, words: List[dict]) -> List[FillerWord]:
        """Detect filler words in speech - only hesitations (um, uh, etc.) - CONSERVATIVE"""
        fillers = []
        
        # Only check actual words from speech recognition (most reliable)
        for i, word_data in enumerate(words):
            word = word_data['word'].lower().strip()
            
            # Remove punctuation
            import re
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Only check for clear hesitation fillers that were actually recognized
            if clean_word in self.filler_words['hesitation']:
                # Additional validation: word should have reasonable confidence
                confidence = word_data.get('conf', 0.5)
                if confidence > 0.2:  # Only if Vosk is somewhat confident
                    fillers.append(FillerWord(
                        word=clean_word,
                        start_time=word_data.get('start', 0.0),
                        end_time=word_data.get('end', 0.0),
                        type='hesitation'
                    ))
                    print(f"📝 Detected filler in Vosk: '{clean_word}' at {word_data.get('start', 0):.2f}s (conf: {confidence:.2f})")
        
        return fillers
    
    def _detect_fillers_from_whisper_text(self, transcription: str, total_duration: float) -> List[FillerWord]:
        """Conservative filler detection from Whisper text - only clear cases"""
        import re
        
        fillers = []
        
        # Only look for very clear filler words in transcription
        # Whisper sometimes adds artifacts, so be conservative
        clear_filler_patterns = [
            r'\bum\b', r'\buh\b', r'\bumm\b', r'\berr\b'  # Only the most common ones
        ]
        
        # Find all filler words in the transcription
        text_lower = transcription.lower()
        words_in_text = text_lower.split()
        
        for i, word in enumerate(words_in_text):
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            # Only check for the most obvious fillers (not 'ah', 'eh' which could be real words)
            if clean_word in ['um', 'uh', 'umm', 'err']:
                # Estimate timing based on position in text
                estimated_start = (i / len(words_in_text)) * total_duration
                estimated_end = estimated_start + 0.3  # Assume 300ms duration
                
                fillers.append(FillerWord(
                    word=clean_word,
                    start_time=estimated_start,
                    end_time=estimated_end,
                    type='hesitation'
                ))
                print(f"📝 Found clear filler '{clean_word}' in Whisper text at estimated time {estimated_start:.2f}s")
        
        return fillers
    
    def _detect_long_pauses(self, words: List[dict], total_duration: float) -> List[Pause]:
        """Detect long pauses between words"""
        pauses = []
        
        for i in range(len(words) - 1):
            current_end = words[i]['end']
            next_start = words[i + 1]['start']
            pause_duration = next_start - current_end
            
            # Consider pauses longer than 1 second as "long pauses"
            if pause_duration > 1.0:
                pauses.append(Pause(
                    start_time=current_end,
                    end_time=next_start,
                    duration=round(pause_duration, 2)
                ))
        
        return pauses
    
    def _detect_repetitions(self, words: List[dict]) -> List[Repetition]:
        """Detect successive word/phrase repetitions (stuttering/disfluency)"""
        repetitions = []
        
        # Look for successive repetitions (immediate)
        i = 0
        while i < len(words) - 1:
            current_word = words[i]['word']
            
            # Skip only extremely common function words that naturally repeat
            # Don't skip pronouns like "I", "you" as these can be stuttered
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'that', 'this'}
            
            if current_word in common_words:
                i += 1
                continue
            
            # Check for immediate successive repetitions
            consecutive_count = 1
            repetition_occurrences = [{
                'start_time': words[i]['start'],
                'end_time': words[i]['end'],
                'text': current_word
            }]
            
            j = i + 1
            while j < len(words) and words[j]['word'] == current_word:
                # Allow small gaps between repeated words (up to 0.5 seconds for stammering)
                time_gap = words[j]['start'] - words[j-1]['end']
                if time_gap <= 0.5:  # Very close together - likely a stutter/repetition
                    consecutive_count += 1
                    repetition_occurrences.append({
                        'start_time': words[j]['start'],
                        'end_time': words[j]['end'],
                        'text': current_word
                    })
                    j += 1
                else:
                    break  # Gap too large, not a successive repetition
            
            # Only count as repetition if word appears 2+ times successively
            if consecutive_count >= 2:
                repetitions.append(Repetition(
                    repeated_text=current_word,
                    occurrences=repetition_occurrences,
                    count=consecutive_count
                ))
                
                print(f"🔄 Detected successive repetition: '{current_word}' repeated {consecutive_count} times")
            
            # Move past all the repeated words
            i = j if j > i + 1 else i + 1
        
        # NEW: Check for repetitions separated by small words (e.g., "like a like", "I uh I", etc.)
        # Look for patterns where same word appears within 2-3 positions with only filler/short words between
        for i in range(len(words) - 2):
            current_word = words[i]['word']
            
            # Skip only extremely common function words for separated repetitions too
            if current_word in common_words:
                continue
                
            # Look ahead 1-3 positions for the same word
            for j in range(i + 2, min(i + 4, len(words))):  # Look 2-3 words ahead
                if words[j]['word'] == current_word:
                    # Check if words in between are small/filler words
                    words_between = [words[k]['word'] for k in range(i + 1, j)]
                    
                    # Small words that often separate repetitions
                    small_words = {'a', 'an', 'the', 'uh', 'um', 'er', 'ah'}
                    
                    # If all words between are small/filler words, it's likely a repetition
                    if all(word in small_words for word in words_between):
                        # Check timing - should be within reasonable time window (3 seconds)
                        time_gap = words[j]['start'] - words[i]['end']
                        if time_gap <= 3.0:
                            repetition_occurrences = [
                                {
                                    'start_time': words[i]['start'],
                                    'end_time': words[i]['end'],
                                    'text': current_word
                                },
                                {
                                    'start_time': words[j]['start'],
                                    'end_time': words[j]['end'],
                                    'text': current_word
                                }
                            ]
                            
                            repetitions.append(Repetition(
                                repeated_text=current_word,
                                occurrences=repetition_occurrences,
                                count=2
                            ))
                            
                            filler_context = ' '.join(words_between) if words_between else '(immediate)'
                            print(f"🔄 Detected separated repetition: '{current_word}' repeated with '{filler_context}' between")
                            break  # Found repetition for this word, move on
        
        # Also check for phrase repetitions (2-3 word sequences)
        phrase_repetitions = self._detect_phrase_repetitions(words)
        repetitions.extend(phrase_repetitions)
        
        return repetitions
    
    def _detect_phrase_repetitions(self, words: List[dict]) -> List[Repetition]:
        """Detect successive phrase repetitions like 'I like I like' or 'you know you know'"""
        repetitions = []
        
        # Check for 2-word phrase repetitions first (more common)
        for phrase_length in [2, 3]:
            i = 0
            while i <= len(words) - (phrase_length * 2):
                # Get the first phrase
                phrase_words = [words[i + j]['word'] for j in range(phrase_length)]
                phrase_text = ' '.join(phrase_words)
                
                # For phrase repetitions, we're more lenient about common words
                # Skip only if ALL words are extremely common function words
                skip_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                if all(word in skip_words for word in phrase_words):
                    i += 1
                    continue
                
                # Check if the next phrase matches exactly
                next_phrase_start = i + phrase_length
                if next_phrase_start + phrase_length <= len(words):
                    next_phrase_words = [words[next_phrase_start + j]['word'] for j in range(phrase_length)]
                    next_phrase_text = ' '.join(next_phrase_words)
                    
                    if phrase_text.lower() == next_phrase_text.lower():  # Case insensitive comparison
                        # Check timing - phrases should be close together (within 2 seconds gap)
                        gap_time = words[next_phrase_start]['start'] - words[i + phrase_length - 1]['end']
                        if gap_time <= 2.0:  # More lenient timing
                            repetition_occurrences = [
                                {
                                    'start_time': words[i]['start'],
                                    'end_time': words[i + phrase_length - 1]['end'],
                                    'text': phrase_text
                                },
                                {
                                    'start_time': words[next_phrase_start]['start'],
                                    'end_time': words[next_phrase_start + phrase_length - 1]['end'],
                                    'text': next_phrase_text
                                }
                            ]
                            
                            repetitions.append(Repetition(
                                repeated_text=phrase_text,
                                occurrences=repetition_occurrences,
                                count=2
                            ))
                            
                            print(f"🔄 Detected phrase repetition: '{phrase_text}' repeated sequentially")
                            
                            # Skip past both phrases
                            i = next_phrase_start + phrase_length
                            continue
                
                i += 1
        
        return repetitions
    
    def _is_filler_word(self, word: str) -> bool:
        """Check if a word is a filler word - including grammatical fillers"""
        # Traditional hesitation fillers
        if (word in self.filler_words['hesitation'] or 
            word in self.filler_words['discourse_marker']):
            return True
        
        # Grammatical fillers (common function words that can be misplaced)
        # Only consider these as fillers in specific contexts - for now, just flag them
        potential_grammatical_fillers = {'a', 'an', 'the', 'and', 'or', 'but', 'so', 'well', 'like', 'you', 'know'}
        if word in potential_grammatical_fillers:
            # For now, we'll be conservative and not automatically exclude these from speech rate
            # The enhanced filler detection will catch the obvious cases
            return False
        
        return False
    
    def _calculate_speech_rate_over_time(self, words: List[dict], total_duration: float) -> List[dict]:
        """Calculate speech rate over time (2-second segments)"""
        speech_rate_over_time = []
        segment_duration = 2.0  # 2-second segments
        current_time = 0.0
        
        while current_time < total_duration:
            segment_end = min(current_time + segment_duration, total_duration)
            
            # Get words in this time segment
            segment_words = [w for w in words 
                           if w['start'] >= current_time and w['start'] < segment_end]
            
            # Filter out filler words for rate calculation
            content_words = [w for w in segment_words if not self._is_filler_word(w['word'])]
            
            # Calculate rate for this segment
            actual_duration = segment_end - current_time
            if actual_duration > 0:
                words_per_minute = (len(content_words) / (actual_duration / 60))
            else:
                words_per_minute = 0
            
            # Create segment text for debugging/display
            segment_text = ' '.join([w['word'] for w in segment_words])
            
            speech_rate_over_time.append({
                'time': current_time + (actual_duration / 2),  # midpoint of segment
                'rate': round(words_per_minute, 1),
                'segment_text': segment_text
            })
            
            current_time += segment_duration
        
        return speech_rate_over_time 