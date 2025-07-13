import numpy as np
import scipy.stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ConfidenceMetrics:
    """Container for comprehensive confidence metrics"""
    acoustic_confidence: float      # 0-1, based on audio quality and clarity
    temporal_confidence: float      # 0-1, based on timing and duration consistency
    lexical_confidence: float       # 0-1, based on word recognition consistency
    phonetic_confidence: float      # 0-1, based on phoneme alignment quality
    overall_confidence: float       # 0-1, weighted combination
    reliability_factors: Dict[str, float]  # Individual factor scores for debugging

class ProductionConfidenceScorer:
    """
    Production-quality confidence scoring using multiple acoustic, temporal, and lexical features
    """
    
    def __init__(self):
        # Weight factors for different confidence components
        self.confidence_weights = {
            'acoustic': 0.3,    # Audio quality, SNR, clarity
            'temporal': 0.2,    # Timing consistency, duration patterns
            'lexical': 0.25,    # Word recognition consistency, dictionary matching
            'phonetic': 0.25    # Phoneme alignment quality, pronunciation accuracy
        }
        
        # Thresholds for confidence assessment
        self.thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
    
    def calculate_comprehensive_confidence(
        self,
        audio_data: np.ndarray,
        samplerate: int,
        vosk_transcription: str,
        vosk_words_data: List[dict],
        whisper_transcription: str = None,
        expected_text: str = None,
        phoneme_alignment_scores: List[float] = None
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence using multiple factors
        """
        
        # 1. Acoustic confidence - audio quality and clarity
        acoustic_confidence = self._calculate_acoustic_confidence(audio_data, samplerate)
        
        # 2. Temporal confidence - timing and duration consistency
        temporal_confidence = self._calculate_temporal_confidence(vosk_words_data, audio_data, samplerate)
        
        # 3. Lexical confidence - word recognition consistency
        lexical_confidence = self._calculate_lexical_confidence(
            vosk_transcription, whisper_transcription, expected_text, vosk_words_data
        )
        
        # 4. Phonetic confidence - pronunciation alignment quality
        phonetic_confidence = self._calculate_phonetic_confidence(
            phoneme_alignment_scores, vosk_words_data
        )
        
        # Calculate weighted overall confidence
        overall_confidence = (
            acoustic_confidence * self.confidence_weights['acoustic'] +
            temporal_confidence * self.confidence_weights['temporal'] +
            lexical_confidence * self.confidence_weights['lexical'] +
            phonetic_confidence * self.confidence_weights['phonetic']
        )
        
        # Collect reliability factors for debugging
        reliability_factors = {
            'audio_snr': self._estimate_snr(audio_data, samplerate),
            'word_count': len(vosk_words_data),
            'avg_word_confidence': np.mean([w.get('conf', 0.5) for w in vosk_words_data]) if vosk_words_data else 0.5,
            'timing_consistency': self._assess_timing_consistency(vosk_words_data),
            'transcription_agreement': self._assess_transcription_agreement(vosk_transcription, whisper_transcription),
            'phoneme_accuracy': np.mean(phoneme_alignment_scores) if phoneme_alignment_scores else 0.5
        }
        
        return ConfidenceMetrics(
            acoustic_confidence=acoustic_confidence,
            temporal_confidence=temporal_confidence,
            lexical_confidence=lexical_confidence,
            phonetic_confidence=phonetic_confidence,
            overall_confidence=overall_confidence,
            reliability_factors=reliability_factors
        )
    
    def _calculate_acoustic_confidence(self, audio_data: np.ndarray, samplerate: int) -> float:
        """
        Calculate confidence based on acoustic features
        """
        try:
            # 1. Signal-to-Noise Ratio
            snr = self._estimate_snr(audio_data, samplerate)
            snr_score = min(1.0, max(0.0, (snr - 5) / 20))  # 5-25 dB range
            
            # 2. Dynamic range assessment
            dynamic_range = np.percentile(np.abs(audio_data), 95) - np.percentile(np.abs(audio_data), 5)
            dynamic_score = min(1.0, dynamic_range * 5)  # 0-0.2 range expected
            
            # 3. Clipping detection
            clipping_ratio = np.sum(np.abs(audio_data) > 0.95) / len(audio_data)
            clipping_score = max(0.0, 1.0 - clipping_ratio * 10)  # Heavy penalty for clipping
            
            # 4. Spectral clarity
            spectral_score = self._assess_spectral_clarity(audio_data, samplerate)
            
            # 5. Voice activity consistency
            voice_activity_score = self._assess_voice_activity_consistency(audio_data, samplerate)
            
            # Weighted combination
            acoustic_confidence = (
                snr_score * 0.3 +
                dynamic_score * 0.2 +
                clipping_score * 0.2 +
                spectral_score * 0.2 +
                voice_activity_score * 0.1
            )
            
            return min(1.0, max(0.0, acoustic_confidence))
            
        except Exception as e:
            print(f"⚠️ Acoustic confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence
    
    def _calculate_temporal_confidence(self, vosk_words_data: List[dict], audio_data: np.ndarray, samplerate: int) -> float:
        """
        Calculate confidence based on temporal patterns
        """
        try:
            if not vosk_words_data:
                return 0.3  # Low confidence for no word detection
            
            # 1. Timing consistency
            timing_consistency = self._assess_timing_consistency(vosk_words_data)
            
            # 2. Word duration reasonableness
            duration_reasonableness = self._assess_word_durations(vosk_words_data)
            
            # 3. Speech rate assessment
            total_duration = len(audio_data) / samplerate
            speech_rate = len(vosk_words_data) / (total_duration / 60)  # words per minute
            
            # Optimal speech rate: 140-180 WPM
            if 120 <= speech_rate <= 200:
                rate_score = 1.0
            elif 100 <= speech_rate <= 250:
                rate_score = 0.8
            else:
                rate_score = 0.5
            
            # 4. Pause pattern assessment
            pause_score = self._assess_pause_patterns(vosk_words_data, total_duration)
            
            # Weighted combination
            temporal_confidence = (
                timing_consistency * 0.3 +
                duration_reasonableness * 0.3 +
                rate_score * 0.2 +
                pause_score * 0.2
            )
            
            return min(1.0, max(0.0, temporal_confidence))
            
        except Exception as e:
            print(f"⚠️ Temporal confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_lexical_confidence(self, vosk_transcription: str, whisper_transcription: str, 
                                    expected_text: str, vosk_words_data: List[dict]) -> float:
        """
        Calculate confidence based on lexical consistency
        """
        try:
            lexical_factors = []
            
            # 1. Vosk word confidence scores
            if vosk_words_data:
                avg_word_conf = np.mean([w.get('conf', 0.5) for w in vosk_words_data])
                lexical_factors.append(avg_word_conf)
            
            # 2. Transcription agreement (Vosk vs Whisper)
            if whisper_transcription:
                agreement_score = self._assess_transcription_agreement(vosk_transcription, whisper_transcription)
                lexical_factors.append(agreement_score)
            
            # 3. Expected text alignment (if available)
            if expected_text:
                alignment_score = self._assess_expected_alignment(vosk_transcription, expected_text)
                lexical_factors.append(alignment_score)
            
            # 4. Vocabulary complexity consistency
            vocab_score = self._assess_vocabulary_consistency(vosk_transcription, vosk_words_data)
            lexical_factors.append(vocab_score)
            
            # 5. Out-of-vocabulary word ratio
            oov_score = self._assess_oov_ratio(vosk_words_data)
            lexical_factors.append(oov_score)
            
            # Average of available factors
            lexical_confidence = np.mean(lexical_factors) if lexical_factors else 0.5
            
            return min(1.0, max(0.0, lexical_confidence))
            
        except Exception as e:
            print(f"⚠️ Lexical confidence calculation failed: {e}")
            return 0.5
    
    def _calculate_phonetic_confidence(self, phoneme_alignment_scores: List[float], 
                                     vosk_words_data: List[dict]) -> float:
        """
        Calculate confidence based on phonetic alignment quality
        """
        try:
            phonetic_factors = []
            
            # 1. Phoneme alignment scores
            if phoneme_alignment_scores:
                avg_phoneme_score = np.mean(phoneme_alignment_scores)
                phonetic_factors.append(avg_phoneme_score)
                
                # Score distribution consistency
                score_std = np.std(phoneme_alignment_scores)
                consistency_score = max(0.0, 1.0 - score_std * 2)  # Lower std = higher consistency
                phonetic_factors.append(consistency_score)
            
            # 2. Word recognition consistency
            if vosk_words_data:
                conf_scores = [w.get('conf', 0.5) for w in vosk_words_data]
                conf_consistency = 1.0 - (np.std(conf_scores) / max(0.1, np.mean(conf_scores)))
                conf_consistency = max(0.0, min(1.0, conf_consistency))
                phonetic_factors.append(conf_consistency)
            
            # 3. Pronunciation difficulty adjustment
            if vosk_words_data:
                difficulty_score = self._assess_pronunciation_difficulty_consistency(vosk_words_data)
                phonetic_factors.append(difficulty_score)
            
            # Average of available factors
            phonetic_confidence = np.mean(phonetic_factors) if phonetic_factors else 0.5
            
            return min(1.0, max(0.0, phonetic_confidence))
            
        except Exception as e:
            print(f"⚠️ Phonetic confidence calculation failed: {e}")
            return 0.5
    
    def _estimate_snr(self, audio_data: np.ndarray, samplerate: int) -> float:
        """Estimate Signal-to-Noise Ratio using voice activity detection"""
        try:
            import librosa
            
            # Use voice activity detection approach
            frame_length = int(0.025 * samplerate)  # 25ms frames
            hop_length = int(0.010 * samplerate)   # 10ms hop
            
            frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
            frame_energy = np.sum(frames**2, axis=0)
            
            # Assume top 30% energy frames are speech, bottom 30% are noise
            energy_sorted = np.sort(frame_energy)
            noise_energy = np.mean(energy_sorted[:len(energy_sorted)//3])
            signal_energy = np.mean(energy_sorted[-len(energy_sorted)//3:])
            
            if noise_energy > 1e-10:
                snr_linear = signal_energy / noise_energy
                snr_db = 10 * np.log10(snr_linear)
                return min(40.0, max(0.0, snr_db))
            else:
                return 30.0  # High SNR if no noise detected
                
        except Exception:
            return 15.0  # Default moderate SNR
    
    def _assess_spectral_clarity(self, audio_data: np.ndarray, samplerate: int) -> float:
        """Assess spectral clarity for speech"""
        try:
            import scipy.signal
            
            # Calculate power spectral density
            freqs, psd = scipy.signal.welch(audio_data, samplerate, nperseg=1024)
            
            # Speech frequency bands
            speech_band = (freqs >= 300) & (freqs <= 3400)
            noise_band = (freqs >= 50) & (freqs <= 200)
            
            speech_power = np.mean(psd[speech_band]) if np.any(speech_band) else 0
            noise_power = np.mean(psd[noise_band]) if np.any(noise_band) else 1e-10
            
            # Speech-to-noise power ratio
            if noise_power > 0:
                clarity_ratio = speech_power / noise_power
                clarity_score = min(1.0, clarity_ratio / 10)  # Normalize
                return clarity_score
            else:
                return 0.8  # Good clarity if no low-frequency noise
                
        except Exception:
            return 0.5
    
    def _assess_voice_activity_consistency(self, audio_data: np.ndarray, samplerate: int) -> float:
        """Assess consistency of voice activity patterns"""
        try:
            import librosa
            
            # Calculate RMS energy
            frame_length = int(0.025 * samplerate)
            hop_length = int(0.010 * samplerate)
            
            rms = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
            
            # Voice activity detection
            threshold = np.percentile(rms, 30)
            voice_activity = rms > threshold
            
            # Assess consistency - fewer transitions = more consistent
            transitions = np.sum(np.diff(voice_activity.astype(int)) != 0)
            total_frames = len(voice_activity)
            
            # Normalize: fewer transitions relative to length = higher consistency
            consistency_score = max(0.0, 1.0 - (transitions / total_frames) * 2)
            
            return min(1.0, consistency_score)
            
        except Exception:
            return 0.5
    
    def _assess_timing_consistency(self, vosk_words_data: List[dict]) -> float:
        """Assess timing consistency of word boundaries"""
        if len(vosk_words_data) < 2:
            return 0.7  # Neutral for single/no words
        
        try:
            # Calculate inter-word gaps
            gaps = []
            for i in range(len(vosk_words_data) - 1):
                end_time = vosk_words_data[i].get('end', 0)
                start_time = vosk_words_data[i + 1].get('start', 0)
                gap = start_time - end_time
                gaps.append(gap)
            
            if not gaps:
                return 0.5
            
            # Assess gap consistency (reasonable gaps: 0-0.5 seconds)
            reasonable_gaps = [g for g in gaps if 0 <= g <= 0.5]
            consistency_ratio = len(reasonable_gaps) / len(gaps)
            
            # Penalize highly variable gaps
            if len(gaps) > 1:
                gap_std = np.std(gaps)
                variability_penalty = min(0.3, gap_std * 2)
                consistency_ratio -= variability_penalty
            
            return max(0.0, min(1.0, consistency_ratio))
            
        except Exception:
            return 0.5
    
    def _assess_word_durations(self, vosk_words_data: List[dict]) -> float:
        """Assess reasonableness of word durations"""
        if not vosk_words_data:
            return 0.3
        
        try:
            durations = []
            for word_data in vosk_words_data:
                start = word_data.get('start', 0)
                end = word_data.get('end', 0)
                duration = end - start
                durations.append(duration)
            
            if not durations:
                return 0.5
            
            # Reasonable word durations: 0.1-2.0 seconds
            reasonable_durations = [d for d in durations if 0.1 <= d <= 2.0]
            reasonableness_ratio = len(reasonable_durations) / len(durations)
            
            return max(0.0, min(1.0, reasonableness_ratio))
            
        except Exception:
            return 0.5
    
    def _assess_pause_patterns(self, vosk_words_data: List[dict], total_duration: float) -> float:
        """Assess naturalness of pause patterns"""
        if len(vosk_words_data) < 2:
            return 0.7
        
        try:
            # Calculate pause durations
            pauses = []
            for i in range(len(vosk_words_data) - 1):
                end_time = vosk_words_data[i].get('end', 0)
                start_time = vosk_words_data[i + 1].get('start', 0)
                pause = start_time - end_time
                if pause > 0:
                    pauses.append(pause)
            
            if not pauses:
                return 0.8  # No pauses detected
            
            # Assess pause reasonableness
            reasonable_pauses = [p for p in pauses if 0.05 <= p <= 3.0]
            pause_ratio = len(reasonable_pauses) / len(pauses)
            
            # Total pause time should be reasonable (not more than 50% of speech)
            total_pause_time = sum(pauses)
            pause_time_ratio = total_pause_time / total_duration
            
            if pause_time_ratio > 0.5:
                pause_ratio *= 0.5  # Heavy penalty for too many pauses
            
            return max(0.0, min(1.0, pause_ratio))
            
        except Exception:
            return 0.5
    
    def _assess_transcription_agreement(self, vosk_text: str, whisper_text: str) -> float:
        """Assess agreement between Vosk and Whisper transcriptions"""
        if not vosk_text or not whisper_text:
            return 0.5
        
        try:
            vosk_words = vosk_text.lower().split()
            whisper_words = whisper_text.lower().split()
            
            if not vosk_words or not whisper_words:
                return 0.3
            
            # Calculate word-level agreement using sequence alignment
            agreement_score = self._calculate_sequence_similarity(vosk_words, whisper_words)
            
            return max(0.0, min(1.0, agreement_score))
            
        except Exception:
            return 0.5
    
    def _assess_expected_alignment(self, transcription: str, expected_text: str) -> float:
        """Assess alignment with expected text"""
        if not transcription or not expected_text:
            return 0.5
        
        try:
            trans_words = transcription.lower().split()
            expected_words = expected_text.lower().split()
            
            if not trans_words or not expected_words:
                return 0.3
            
            # Calculate alignment score
            alignment_score = self._calculate_sequence_similarity(trans_words, expected_words)
            
            return max(0.0, min(1.0, alignment_score))
            
        except Exception:
            return 0.5
    
    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two word sequences using dynamic programming"""
        if not seq1 or not seq2:
            return 0.0
        
        m, n = len(seq1), len(seq2)
        dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1.0
                else:
                    # Partial credit for similar words
                    similarity = self._word_similarity(seq1[i-1], seq2[j-1])
                    dp[i][j] = max(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1] + similarity
                    )
        
        max_possible = max(m, n)
        return dp[m][n] / max_possible if max_possible > 0 else 0.0
    
    def _word_similarity(self, word1: str, word2: str) -> float:
        """Simple word similarity metric"""
        if not word1 or not word2:
            return 0.0
        
        if word1 == word2:
            return 1.0
        
        # Edit distance similarity
        max_len = max(len(word1), len(word2))
        if max_len == 0:
            return 0.0
        
        # Simple character overlap
        chars1 = set(word1)
        chars2 = set(word2)
        overlap = len(chars1.intersection(chars2))
        total = len(chars1.union(chars2))
        
        return overlap / total if total > 0 else 0.0
    
    def _assess_vocabulary_consistency(self, transcription: str, vosk_words_data: List[dict]) -> float:
        """Assess vocabulary complexity consistency"""
        if not transcription or not vosk_words_data:
            return 0.5
        
        try:
            words = transcription.lower().split()
            
            # Simple vocabulary complexity assessment
            complex_words = [w for w in words if len(w) > 6]
            complexity_ratio = len(complex_words) / len(words) if words else 0
            
            # Balanced complexity is good (not too simple, not too complex)
            if 0.1 <= complexity_ratio <= 0.4:
                complexity_score = 1.0
            else:
                complexity_score = 0.7
            
            return complexity_score
            
        except Exception:
            return 0.5
    
    def _assess_oov_ratio(self, vosk_words_data: List[dict]) -> float:
        """Assess out-of-vocabulary word ratio"""
        if not vosk_words_data:
            return 0.5
        
        try:
            # Words with very low confidence might be OOV
            low_conf_words = [w for w in vosk_words_data if w.get('conf', 0.5) < 0.3]
            oov_ratio = len(low_conf_words) / len(vosk_words_data)
            
            # Lower OOV ratio is better
            oov_score = max(0.0, 1.0 - oov_ratio * 2)
            
            return oov_score
            
        except Exception:
            return 0.5
    
    def _assess_pronunciation_difficulty_consistency(self, vosk_words_data: List[dict]) -> float:
        """Assess consistency of pronunciation difficulty with confidence scores"""
        if not vosk_words_data:
            return 0.5
        
        try:
            # Simple heuristic: longer words should generally have lower confidence
            # if pronunciation is challenging
            consistency_scores = []
            
            for word_data in vosk_words_data:
                word = word_data.get('word', '')
                conf = word_data.get('conf', 0.5)
                
                # Expected confidence based on word length
                expected_conf = max(0.3, 1.0 - len(word) * 0.05)
                
                # Consistency: how well actual confidence matches expected
                consistency = 1.0 - abs(conf - expected_conf)
                consistency_scores.append(consistency)
            
            return np.mean(consistency_scores) if consistency_scores else 0.5
            
        except Exception:
            return 0.5
    
    def get_confidence_level_description(self, confidence: float) -> str:
        """Get human-readable confidence level description"""
        if confidence >= self.thresholds['high_confidence']:
            return "High"
        elif confidence >= self.thresholds['medium_confidence']:
            return "Medium"
        elif confidence >= self.thresholds['low_confidence']:
            return "Low"
        else:
            return "Very Low"
    
    def get_reliability_assessment(self, metrics: ConfidenceMetrics) -> Dict[str, str]:
        """Get detailed reliability assessment"""
        assessment = {
            'overall_confidence': self.get_confidence_level_description(metrics.overall_confidence),
            'acoustic_quality': 'Good' if metrics.acoustic_confidence > 0.7 else 'Fair' if metrics.acoustic_confidence > 0.4 else 'Poor',
            'temporal_consistency': 'Consistent' if metrics.temporal_confidence > 0.7 else 'Moderate' if metrics.temporal_confidence > 0.4 else 'Inconsistent',
            'lexical_reliability': 'Reliable' if metrics.lexical_confidence > 0.7 else 'Moderate' if metrics.lexical_confidence > 0.4 else 'Unreliable',
            'phonetic_accuracy': 'High' if metrics.phonetic_confidence > 0.7 else 'Moderate' if metrics.phonetic_confidence > 0.4 else 'Low'
        }
        
        # Add specific recommendations
        recommendations = []
        
        if metrics.acoustic_confidence < 0.5:
            recommendations.append("Consider improving audio quality (reduce background noise, use better microphone)")
        
        if metrics.temporal_confidence < 0.5:
            recommendations.append("Speech timing appears inconsistent (check for interruptions or technical issues)")
        
        if metrics.lexical_confidence < 0.5:
            recommendations.append("Word recognition confidence is low (consider speaking more clearly)")
        
        if metrics.phonetic_confidence < 0.5:
            recommendations.append("Pronunciation assessment may be less reliable (focus on clear articulation)")
        
        assessment['recommendations'] = recommendations
        
        return assessment 