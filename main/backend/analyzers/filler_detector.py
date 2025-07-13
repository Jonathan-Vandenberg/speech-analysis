import numpy as np
import librosa
import webrtcvad
import json
import tempfile
import wave
import os
from typing import List, Tuple, Dict, Optional
from models.responses import FillerWord
import scipy.signal
import re

# Try to import forced alignment library
try:
    from forcealign import ForceAlign
    FORCEALIGN_AVAILABLE = True
    print("✅ ForceAlign library loaded successfully")
except ImportError as e:
    FORCEALIGN_AVAILABLE = False
    print(f"⚠️ ForceAlign not available: {e}")

class AdvancedFillerDetector:
    """
    Conservative filler word detection focused on accuracy over completeness.
    Prioritizes avoiding false positives by using multiple validation layers.
    """
    
    def __init__(self):
        self.vad = webrtcvad.Vad(1)  # Reduced aggressiveness to avoid false positives
        
        # More restrictive filler signatures - narrower ranges for better accuracy
        self.filler_signatures = {
            'um': {
                'f1_range': (300, 500),    # Narrower, more typical range
                'f2_range': (900, 1100),   # Narrower range
                'duration_range': (0.2, 0.6),  # Typical filler duration
                'energy_pattern': 'low_stable',
                'min_energy': 0.01  # Higher minimum energy threshold
            },
            'uh': {
                'f1_range': (350, 550),    # Narrower range
                'f2_range': (1100, 1300),  
                'duration_range': (0.15, 0.5),   
                'energy_pattern': 'steady_low',
                'min_energy': 0.008
            },
            'err': {
                'f1_range': (450, 700),    # More restrictive
                'f2_range': (1300, 1500), 
                'duration_range': (0.2, 0.7),  
                'energy_pattern': 'variable',
                'min_energy': 0.012
            }
        }
        
        # Known filler words for text-based detection
        self.text_fillers = {
            'um', 'uh', 'uhh', 'umm', 'err', 'ah', 'eh', 'hmm', 'hm'
        }
        
        # Common grammatical function words that can become fillers when misplaced
        self.potential_grammatical_fillers = {
            'a', 'an', 'the', 'and', 'or', 'but', 'so', 'well', 'like', 'you', 'know'
        }
    
    def detect_audio_based_fillers(self, audio_data: np.ndarray, sample_rate: int, 
                                  existing_words: List[dict]) -> List[FillerWord]:
        """
        Conservative audio-based filler detection with multiple validation layers
        """
        print("🔍 Starting conservative audio-based filler detection...")
        
        detected_fillers = []
        
        try:
            # LAYER 1: Text-based detection from existing words (most reliable)
            text_fillers = self._detect_text_based_fillers(existing_words)
            detected_fillers.extend(text_fillers)
            print(f"📝 Text-based detection: {len(text_fillers)} fillers")
            
            # LAYER 2: ALWAYS run acoustic analysis since speech recognition often filters out real fillers
            print("🎧 Running acoustic detection - speech recognition may have filtered out real filler words")
            acoustic_fillers = self._detect_acoustic_fillers_conservative(
                audio_data, sample_rate, existing_words
            )
            detected_fillers.extend(acoustic_fillers)
            print(f"🎧 Acoustic detection: {len(acoustic_fillers)} fillers")
            
            print(f"✅ Conservative detection found {len(detected_fillers)} total fillers")
            return detected_fillers
            
        except Exception as e:
            print(f"⚠️ Conservative filler detection failed: {e}")
            return []
    
    def _detect_text_based_fillers(self, existing_words: List[dict]) -> List[FillerWord]:
        """Detect fillers from speech recognition text - most reliable method"""
        fillers = []
        
        for word_data in existing_words:
            word = word_data.get('word', '').lower().strip()
            
            # Clean the word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in self.text_fillers:
                fillers.append(FillerWord(
                    word=clean_word,
                    start_time=word_data.get('start', 0.0),
                    end_time=word_data.get('end', 0.0),
                    type='hesitation'
                ))
                print(f"📝 Detected text filler: '{clean_word}' at {word_data.get('start', 0):.2f}s")
        
        return fillers
    
    def detect_grammatical_fillers(self, vosk_transcript: str, whisper_transcript: str, vosk_words: List[dict]) -> List[FillerWord]:
        """
        Detect grammatical fillers by comparing Vosk (what was actually spoken) vs Whisper (intended speech)
        This catches cases like 'love a playing' where 'a' is a grammatical filler
        """
        fillers = []
        
        if not vosk_transcript or not whisper_transcript:
            return fillers
        
        vosk_words_list = vosk_transcript.lower().split()
        whisper_words_list = whisper_transcript.lower().split()
        
        print(f"🔍 Comparing transcripts for grammatical fillers:")
        print(f"   Vosk (spoken): '{vosk_transcript}'")
        print(f"   Whisper (intended): '{whisper_transcript}'")
        
        # Find words in Vosk that are not in Whisper - potential fillers
        vosk_set = set(vosk_words_list)
        whisper_set = set(whisper_words_list)
        extra_words = vosk_set - whisper_set
        
        # Filter to only potential grammatical fillers
        grammatical_extra_words = extra_words & self.potential_grammatical_fillers
        
        if grammatical_extra_words:
            print(f"🎯 Found potential grammatical fillers: {grammatical_extra_words}")
            
            # Find positions of these words in Vosk transcript
            for filler_word in grammatical_extra_words:
                for word_data in vosk_words:
                    if word_data.get('word', '').lower() == filler_word:
                        # Check if this word makes sense grammatically by looking at context
                        if self._is_likely_grammatical_filler(filler_word, vosk_transcript, whisper_transcript):
                            fillers.append(FillerWord(
                                word=filler_word,
                                start_time=word_data.get('start', 0.0),
                                end_time=word_data.get('end', 0.0),
                                type='grammatical_filler'
                            ))
                            print(f"📝 Detected grammatical filler: '{filler_word}' at {word_data.get('start', 0):.2f}s")
        
        return fillers
    
    def _is_likely_grammatical_filler(self, word: str, vosk_transcript: str, whisper_transcript: str) -> bool:
        """
        Determine if a word is likely a grammatical filler based on context
        """
        # Simple heuristics for common cases
        if word == 'a':
            # Check for patterns like "love a playing" where 'a' is clearly wrong
            if 'love a playing' in vosk_transcript.lower():
                return True
            if 'like a going' in vosk_transcript.lower():
                return True
            # More general pattern: verb + a + -ing verb (usually wrong)
            vosk_words = vosk_transcript.lower().split()
            for i, w in enumerate(vosk_words):
                if w == 'a' and i > 0 and i < len(vosk_words) - 1:
                    prev_word = vosk_words[i-1]
                    next_word = vosk_words[i+1]
                    # If previous word is a verb and next word ends in -ing, 'a' is likely a filler
                    if next_word.endswith('ing') and prev_word in ['love', 'like', 'enjoy', 'hate', 'prefer', 'start', 'stop', 'keep', 'continue']:
                        return True
        
        return False
    
    def _detect_acoustic_fillers_conservative(self, audio_data: np.ndarray, sample_rate: int,
                                           existing_words: List[dict]) -> List[FillerWord]:
        """
        Very conservative acoustic detection - only runs when no text fillers found
        """
        detected_fillers = []
        
        try:
            # Only look for fillers in gaps between words where speech might have been missed
            word_gaps = self._find_word_gaps(existing_words, len(audio_data) / sample_rate)
            
            if len(word_gaps) == 0:
                print("📊 No word gaps found - no acoustic analysis needed")
                return []
            
            print(f"📊 Found {len(word_gaps)} gaps to analyze acoustically")
            
            for gap_start, gap_end in word_gaps:
                # Only analyze significant gaps (> 300ms)
                if gap_end - gap_start < 0.3:
                    continue
                
                # Extract audio segment
                start_sample = int(gap_start * sample_rate)
                end_sample = int(gap_end * sample_rate)
                gap_audio = audio_data[start_sample:end_sample]
                
                # Check if this gap contains potential filler
                filler_type = self._analyze_gap_for_filler(
                    gap_audio, sample_rate, gap_start, gap_end
                )
                
                if filler_type:
                    detected_fillers.append(FillerWord(
                        word=filler_type,
                        start_time=gap_start,
                        end_time=gap_end,
                        type='hesitation'
                    ))
                    print(f"🎯 Acoustic filler in gap: '{filler_type}' at {gap_start:.2f}-{gap_end:.2f}s")
            
            return detected_fillers
            
        except Exception as e:
            print(f"⚠️ Conservative acoustic detection failed: {e}")
            return []
    
    def _find_word_gaps(self, existing_words: List[dict], total_duration: float) -> List[Tuple[float, float]]:
        """Find gaps between recognized words where fillers might be hiding"""
        if len(existing_words) == 0:
            return [(0.0, total_duration)]
        
        gaps = []
        
        # Sort words by start time
        sorted_words = sorted(existing_words, key=lambda w: w.get('start', 0))
        
        # Gap before first word
        first_start = sorted_words[0].get('start', 0)
        if first_start > 0.5:  # Significant gap at start
            gaps.append((0.0, first_start))
        
        # Gaps between words
        for i in range(len(sorted_words) - 1):
            current_end = sorted_words[i].get('end', 0)
            next_start = sorted_words[i + 1].get('start', 0)
            
            # Only consider significant gaps
            if next_start - current_end > 0.3:
                gaps.append((current_end, next_start))
        
        # Gap after last word
        last_end = sorted_words[-1].get('end', 0)
        if total_duration - last_end > 0.5:  # Significant gap at end
            gaps.append((last_end, total_duration))
        
        return gaps
    
    def _analyze_gap_for_filler(self, gap_audio: np.ndarray, sample_rate: int,
                               gap_start: float, gap_end: float) -> Optional[str]:
        """Analyze a word gap for potential filler sounds - conservative but more sensitive"""
        try:
            duration = gap_end - gap_start
            
            # Must have minimum energy to be speech (slightly lowered threshold)
            rms_energy = np.sqrt(np.mean(gap_audio ** 2))
            if rms_energy < 0.010:  # Lowered from 0.015 to 0.010 to catch quieter fillers
                return None
            
            # Must have reasonable duration for a filler (slightly more lenient)
            if duration < 0.12 or duration > 1.2:  # Slightly wider range: 0.12-1.2s instead of 0.15-1.0s
                return None
            
            # Extract basic spectral features
            try:
                # Spectral centroid for brightness
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=gap_audio, sr=sample_rate))
                
                # Zero crossing rate for voicing
                zcr = np.mean(librosa.feature.zero_crossing_rate(gap_audio))
                
                # Must sound voice-like (not just noise) - slightly more lenient
                if zcr > 0.35:  # Increased from 0.3 to 0.35 - allow slightly more fricative content
                    return None
                
                if spectral_centroid < 150 or spectral_centroid > 3500:  # Slightly wider speech range
                    return None
                
                # More generous classification for filler detection
                if 700 < spectral_centroid < 1400 and 0.15 < duration < 0.8:
                    print(f"🎯 Potential 'um' detected: centroid={spectral_centroid:.0f}Hz, duration={duration:.2f}s")
                    return 'um'
                elif 500 < spectral_centroid < 1200 and 0.12 < duration < 0.6:
                    print(f"🎯 Potential 'uh' detected: centroid={spectral_centroid:.0f}Hz, duration={duration:.2f}s")
                    return 'uh'
                elif 900 < spectral_centroid < 1700 and 0.15 < duration < 0.8:
                    print(f"🎯 Potential 'err' detected: centroid={spectral_centroid:.0f}Hz, duration={duration:.2f}s")
                    return 'err'
                elif 400 < spectral_centroid < 1000 and 0.12 < duration < 0.5:
                    print(f"🎯 Potential 'ah' detected: centroid={spectral_centroid:.0f}Hz, duration={duration:.2f}s")
                    return 'ah'
                
            except Exception:
                return None
            
            return None  # No confident match
            
        except Exception as e:
            print(f"⚠️ Error analyzing gap: {e}")
            return None

    def detect_fillers_with_forced_alignment(self, audio_data: np.ndarray, samplerate: int, 
                                           whisper_transcript: str, vosk_transcript: str) -> List[FillerWord]:
        """
        NEW: Use forced alignment to detect fillers by comparing expected vs actual phonemes
        This is the most accurate method for catching fillers that speech recognition corrects
        """
        if not FORCEALIGN_AVAILABLE:
            print("⚠️ ForceAlign not available - skipping forced alignment filler detection")
            return []
        
        try:
            print("🔬 Running forced alignment for filler detection...")
            
            # Save audio to temporary file for ForceAlign
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                # Convert numpy array to WAV format
                audio_int16 = (audio_data * 32767).astype(np.int16)
                
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(samplerate)
                    wav_file.writeframes(audio_int16.tobytes())
                
                temp_audio_path = temp_file.name
            
            # Try alignment with Whisper transcript (what user intended to say)
            try:
                align_whisper = ForceAlign(audio_file=temp_audio_path, transcript=whisper_transcript)
                whisper_words = align_whisper.inference()
                
                print(f"🎯 Forced alignment with Whisper transcript successful: {len(whisper_words)} words")
                
                # Try alignment with Vosk transcript (what was actually spoken)
                if vosk_transcript and vosk_transcript.strip() != whisper_transcript.strip():
                    try:
                        align_vosk = ForceAlign(audio_file=temp_audio_path, transcript=vosk_transcript)
                        vosk_words = align_vosk.inference()
                        
                        print(f"🎯 Forced alignment with Vosk transcript successful: {len(vosk_words)} words")
                        
                        # Compare alignments to find fillers
                        fillers = self._compare_forced_alignments(whisper_words, vosk_words, whisper_transcript, vosk_transcript)
                        
                        # Clean up
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass
                        
                        return fillers
                        
                    except Exception as vosk_align_error:
                        print(f"⚠️ Vosk forced alignment failed: {vosk_align_error}")
                        # Fall back to text comparison
                        fillers = self._detect_fillers_by_text_comparison(whisper_transcript, vosk_transcript, len(audio_data) / samplerate)
                        
                        # Clean up
                        try:
                            os.unlink(temp_audio_path)
                        except:
                            pass
                        
                        return fillers
                
                else:
                    print("📝 Whisper and Vosk transcripts are identical - no fillers detected via alignment")
                    
                    # Clean up
                    try:
                        os.unlink(temp_audio_path)
                    except:
                        pass
                    
                    return []
                    
            except Exception as whisper_align_error:
                print(f"⚠️ Whisper forced alignment failed: {whisper_align_error}")
                
                # Clean up
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
                
                # Fall back to text comparison
                return self._detect_fillers_by_text_comparison(whisper_transcript, vosk_transcript, len(audio_data) / samplerate)
        
        except Exception as e:
            print(f"⚠️ Forced alignment filler detection failed: {e}")
            # Fall back to text comparison
            return self._detect_fillers_by_text_comparison(whisper_transcript, vosk_transcript, len(audio_data) / samplerate)
    
    def _compare_forced_alignments(self, whisper_words, vosk_words, whisper_transcript: str, vosk_transcript: str) -> List[FillerWord]:
        """Compare forced alignments to find filler words"""
        fillers = []
        
        print(f"🔍 Comparing alignments:")
        print(f"   Whisper: '{whisper_transcript}' ({len(whisper_words)} words)")
        print(f"   Vosk: '{vosk_transcript}' ({len(vosk_words)} words)")
        
        # Convert to lists for easier comparison
        whisper_word_list = [w.word.lower() for w in whisper_words]
        vosk_word_list = [w.word.lower() for w in vosk_words]
        
        print(f"   Whisper words: {whisper_word_list}")
        print(f"   Vosk words: {vosk_word_list}")
        
        # Find extra words in Vosk that aren't in Whisper (these are likely fillers)
        vosk_idx = 0
        whisper_idx = 0
        
        while vosk_idx < len(vosk_words) and whisper_idx < len(whisper_words):
            vosk_word = vosk_word_list[vosk_idx]
            whisper_word = whisper_word_list[whisper_idx]
            
            if vosk_word == whisper_word:
                # Words match, advance both
                vosk_idx += 1
                whisper_idx += 1
            else:
                # Check if vosk_word is a potential filler
                if self._is_potential_filler(vosk_word, whisper_word_list[whisper_idx:]):
                    # This is likely a filler word
                    vosk_word_obj = vosk_words[vosk_idx]
                    
                    fillers.append(FillerWord(
                        word=vosk_word,
                        start_time=vosk_word_obj.time_start,
                        end_time=vosk_word_obj.time_end,
                        type='forced_alignment_detected'
                    ))
                    
                    print(f"🎯 FILLER DETECTED via forced alignment: '{vosk_word}' at {vosk_word_obj.time_start:.2f}s")
                    vosk_idx += 1
                else:
                    # Not a clear filler, advance whisper to try to find match
                    whisper_idx += 1
        
        # Check remaining vosk words (these might be fillers at the end)
        while vosk_idx < len(vosk_words):
            vosk_word = vosk_word_list[vosk_idx]
            if self._is_clear_filler(vosk_word):
                vosk_word_obj = vosk_words[vosk_idx]
                fillers.append(FillerWord(
                    word=vosk_word,
                    start_time=vosk_word_obj.time_start,
                    end_time=vosk_word_obj.time_end,
                    type='forced_alignment_detected'
                ))
                print(f"🎯 FILLER DETECTED at end: '{vosk_word}' at {vosk_word_obj.time_start:.2f}s")
            vosk_idx += 1
        
        return fillers
    
    def _is_potential_filler(self, word: str, upcoming_whisper_words: List[str]) -> bool:
        """Check if a word is potentially a filler"""
        word = word.lower().strip()
        
        # Clear hesitation fillers
        if word in self.text_fillers:
            return True
        
        # Grammatical fillers (function words in wrong places)
        grammatical_fillers = {'a', 'an', 'the', 'and', 'or', 'but', 'so', 'like', 'you', 'know'}
        if word in grammatical_fillers:
            # Check if this word would make sense in this context
            # If the next whisper word doesn't start with a vowel, "a" is likely a filler
            if word == 'a' and upcoming_whisper_words:
                next_word = upcoming_whisper_words[0].lower()
                if not next_word.startswith(('a', 'e', 'i', 'o', 'u')):
                    return True
            
            # "Like" before a verb is often a filler
            if word == 'like' and upcoming_whisper_words:
                next_word = upcoming_whisper_words[0].lower()
                # Common verbs that shouldn't follow "like" directly
                action_verbs = {'play', 'playing', 'go', 'going', 'do', 'doing', 'run', 'running'}
                if next_word in action_verbs:
                    return True
                    
            return True  # For now, consider all grammatical words as potential fillers
        
        return False
    
    def _is_clear_filler(self, word: str) -> bool:
        """Check if a word is clearly a filler"""
        return word.lower().strip() in self.text_fillers
    
    def _detect_fillers_by_text_comparison(self, whisper_transcript: str, vosk_transcript: str, duration: float) -> List[FillerWord]:
        """Fallback: detect fillers by simple text comparison"""
        if not vosk_transcript or not whisper_transcript:
            return []
        
        print("📝 Using text comparison fallback for filler detection")
        
        whisper_words = whisper_transcript.lower().split()
        vosk_words = vosk_transcript.lower().split()
        
        print(f"   Whisper: {whisper_words}")
        print(f"   Vosk: {vosk_words}")
        
        fillers = []
        
        # Find words in vosk that aren't in whisper
        for i, vosk_word in enumerate(vosk_words):
            if vosk_word not in whisper_words:
                # Check if it's a potential filler
                if self._is_potential_filler(vosk_word, whisper_words):
                    # Estimate timing
                    estimated_start = (i / len(vosk_words)) * duration
                    estimated_end = estimated_start + 0.3
                    
                    fillers.append(FillerWord(
                        word=vosk_word,
                        start_time=estimated_start,
                        end_time=estimated_end,
                        type='text_comparison_detected'
                    ))
                    
                    print(f"📝 FILLER DETECTED via text comparison: '{vosk_word}' at {estimated_start:.2f}s")
        
        return fillers


def enhance_filler_detection(original_fillers: List[FillerWord], 
                           audio_data: np.ndarray, sample_rate: int,
                           vosk_words: List[dict],
                           vosk_transcript: str = None,
                           whisper_transcript: str = None) -> List[FillerWord]:
    """
    Enhanced filler detection using multiple methods including forced alignment
    """
    print("🚀 Running enhanced filler detection with forced alignment...")
    
    # Create detector
    detector = AdvancedFillerDetector()
    
    # STEP 1: Start with original text-based fillers
    all_fillers = list(original_fillers)
    
    # STEP 2: NEW - Use forced alignment to detect fillers (most accurate)
    if FORCEALIGN_AVAILABLE and whisper_transcript and vosk_transcript:
        print("🔬 Attempting forced alignment filler detection...")
        alignment_fillers = detector.detect_fillers_with_forced_alignment(
            audio_data, sample_rate, whisper_transcript, vosk_transcript
        )
        
        # Add non-duplicate fillers
        for filler in alignment_fillers:
            if not any(abs(f.start_time - filler.start_time) < 0.5 for f in all_fillers):
                all_fillers.append(filler)
                print(f"✅ Added alignment-detected filler: '{filler.word}' at {filler.start_time:.2f}s")
    else:
        print("⚠️ Skipping forced alignment - ForceAlign not available or missing transcripts")
    
    # STEP 3: Conservative acoustic detection in gaps (backup method)
    print("🎧 Running acoustic detection - speech recognition may have filtered out real filler words")
    audio_fillers = detector.detect_audio_based_fillers(audio_data, sample_rate, vosk_words)
    
    # Add non-duplicate acoustic fillers
    for filler in audio_fillers:
        if not any(abs(f.start_time - filler.start_time) < 0.5 for f in all_fillers):
            all_fillers.append(filler)
            print(f"✅ Added acoustic-detected filler: '{filler.word}' at {filler.start_time:.2f}s")
    
    print(f"📊 Total fillers found: {len(original_fillers)} original + {len(alignment_fillers) if 'alignment_fillers' in locals() else 0} alignment + {len(audio_fillers)} acoustic = {len(all_fillers)} total")
    
    return all_fillers 