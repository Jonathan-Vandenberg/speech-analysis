import numpy as np
import json
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from .phoneme_engine import get_phoneme_engine

@dataclass
class ValidationResult:
    """Result of pronunciation validation"""
    is_reliable: bool                    # Overall reliability assessment
    correction_likelihood: float         # 0-1, likelihood that ASR corrected the speech
    raw_acoustic_score: float           # Score based on raw acoustic features
    validation_confidence: float        # Confidence in the validation itself
    issues_detected: List[str]          # List of specific issues found
    recommendations: List[str]          # Recommendations for handling

class PronunciationValidator:
    """
    Validates pronunciation analysis results to detect ASR correction artifacts
    and provide more reliable assessment
    """
    
    def __init__(self):
        self.phoneme_engine = get_phoneme_engine()
        
        # Thresholds for detecting correction artifacts
        self.correction_indicators = {
            'confidence_mismatch_threshold': 0.3,    # Confidence vs expected difficulty mismatch
            'phonetic_distance_threshold': 0.7,     # High phonetic distance with high confidence
            'timing_inconsistency_threshold': 0.4,  # Timing patterns that suggest corrections
            'context_mismatch_threshold': 0.5       # Context doesn't match pronunciation
        }
        
        # Common ASR correction patterns
        self.correction_patterns = self._load_correction_patterns()
    
    def _load_correction_patterns(self) -> Dict[str, List[str]]:
        """Load common ASR correction patterns"""
        return {
            # Common mispronunciations that ASR often corrects
            'th_substitutions': {
                'the': ['da', 'de', 'za', 'ze', 'duh'],
                'that': ['dat', 'zat', 'dhat'],
                'this': ['dis', 'zis', 'diss'],
                'think': ['tink', 'sink', 'fink'],
                'three': ['tree', 'free'],
                'there': ['dare', 'dere'],
                'they': ['day', 'dey'],
                'then': ['den', 'ten'],
                'through': ['trough', 'true'],
                'theatre': ['beatre', 'tetre', 'fetre'],  # Added theatre → beatre pattern
                'therapy': ['berapy', 'terapy'],
                'thick': ['bick', 'tick', 'fick'],
                'thin': ['bin', 'tin', 'fin'],
                'throw': ['brow', 'trow', 'frow']
            },
            'r_l_substitutions': {
                'really': ['leally', 'weally'],
                'right': ['light', 'wight'],
                'red': ['led', 'wed'],
                'run': ['lun', 'wun'],
                'rice': ['lice', 'wice']
            },
            'v_w_substitutions': {
                'very': ['wery', 'berry'],
                'voice': ['woice'],
                'wave': ['vave'],
                'vine': ['wine']
            },
            'consonant_clusters': {
                'street': ['stweet', 'steet'],
                'school': ['scool', 'shool'],
                'spring': ['spwing', 'sping'],
                'strong': ['stwong', 'stong']
            }
        }
    
    def validate_pronunciation_result(
        self,
        expected_word: str,
        recognized_word: str,
        vosk_confidence: float,
        phoneme_scores: List[float],
        timing_data: Dict,
        audio_features: Dict
    ) -> ValidationResult:
        """
        Comprehensive validation of pronunciation analysis result
        """
        
        issues_detected = []
        recommendations = []
        
        # 1. Check for confidence-difficulty mismatch
        correction_likelihood = 0.0
        difficulty_mismatch = self._check_difficulty_confidence_mismatch(
            expected_word, vosk_confidence, phoneme_scores
        )
        
        if difficulty_mismatch['is_mismatch']:
            correction_likelihood += 0.3
            issues_detected.append("High confidence for difficult word suggests possible ASR correction")
            recommendations.append("Use acoustic features for validation")
        
        # 2. Check for known correction patterns
        pattern_match = self._check_correction_patterns(expected_word, recognized_word)
        if pattern_match['is_pattern']:
            correction_likelihood += 0.4
            issues_detected.append(f"Detected known correction pattern: {pattern_match['pattern_type']}")
            recommendations.append("Apply raw acoustic analysis")
        
        # 3. Check phonetic distance vs confidence consistency
        phonetic_consistency = self._check_phonetic_consistency(
            expected_word, recognized_word, vosk_confidence
        )
        
        if not phonetic_consistency['is_consistent']:
            correction_likelihood += 0.2
            issues_detected.append("Phonetic distance inconsistent with confidence score")
        
        # 4. Check timing patterns for correction artifacts
        timing_validation = self._validate_timing_patterns(timing_data, expected_word)
        if not timing_validation['is_natural']:
            correction_likelihood += 0.1
            issues_detected.append("Timing patterns suggest processing artifacts")
        
        # 5. Acoustic feature validation
        acoustic_validation = self._validate_acoustic_features(
            audio_features, expected_word, recognized_word
        )
        
        # Calculate raw acoustic score
        raw_acoustic_score = self._calculate_raw_acoustic_score(
            audio_features, expected_word, timing_data
        )
        
        # Overall reliability assessment
        is_reliable = (
            correction_likelihood < 0.5 and
            acoustic_validation['quality_score'] > 0.6 and
            len(issues_detected) < 3
        )
        
        # Validation confidence
        validation_confidence = self._calculate_validation_confidence(
            audio_features, timing_data, phoneme_scores
        )
        
        # Add recommendations based on findings
        if correction_likelihood > 0.6:
            recommendations.append("Use alternative pronunciation assessment method")
        if acoustic_validation['quality_score'] < 0.4:
            recommendations.append("Request audio re-recording for better quality")
        if not is_reliable:
            recommendations.append("Cross-validate with additional ASR systems")
        
        return ValidationResult(
            is_reliable=is_reliable,
            correction_likelihood=correction_likelihood,
            raw_acoustic_score=raw_acoustic_score,
            validation_confidence=validation_confidence,
            issues_detected=issues_detected,
            recommendations=recommendations
        )
    
    def _check_difficulty_confidence_mismatch(
        self, 
        word: str, 
        confidence: float, 
        phoneme_scores: List[float]
    ) -> Dict:
        """Check if confidence is inconsistent with word difficulty"""
        
        # Calculate expected difficulty
        difficulty_factors = {
            'length': min(0.3, len(word) * 0.03),  # Longer words are harder
            'consonant_clusters': self._count_consonant_clusters(word) * 0.1,
            'difficult_phonemes': self._count_difficult_phonemes(word) * 0.15,
            'syllable_complexity': self._assess_syllable_complexity(word) * 0.1
        }
        
        total_difficulty = sum(difficulty_factors.values())
        expected_confidence = max(0.3, 0.9 - total_difficulty)
        
        # Check for mismatch
        confidence_diff = confidence - expected_confidence
        is_mismatch = confidence_diff > self.correction_indicators['confidence_mismatch_threshold']
        
        return {
            'is_mismatch': is_mismatch,
            'expected_confidence': expected_confidence,
            'actual_confidence': confidence,
            'difficulty_score': total_difficulty,
            'confidence_diff': confidence_diff
        }
    
    def _check_correction_patterns(self, expected_word: str, recognized_word: str) -> Dict:
        """Check if word pair matches known correction patterns"""
        
        expected_lower = expected_word.lower()
        recognized_lower = recognized_word.lower()
        
        # Check each correction pattern category
        for pattern_type, patterns in self.correction_patterns.items():
            for correct_word, mispronunciations in patterns.items():
                if (expected_lower == correct_word and 
                    recognized_lower in mispronunciations):
                    
                    return {
                        'is_pattern': True,
                        'pattern_type': pattern_type,
                        'expected_mispronunciation': recognized_lower,
                        'confidence': 0.9
                    }
                
                # Also check reverse (if ASR corrected mispronunciation to correct word)
                if (recognized_lower == correct_word and 
                    expected_lower in mispronunciations):
                    
                    return {
                        'is_pattern': True,
                        'pattern_type': f"{pattern_type}_corrected",
                        'likely_mispronunciation': expected_lower,
                        'confidence': 0.8
                    }
        
        return {'is_pattern': False}
    
    def _check_phonetic_consistency(
        self, 
        expected_word: str, 
        recognized_word: str, 
        confidence: float
    ) -> Dict:
        """Check if phonetic distance is consistent with confidence"""
        
        try:
            # Get phonetic representations
            expected_phonemes = self.phoneme_engine.get_phonemes(expected_word)
            recognized_phonemes = self.phoneme_engine.get_phonemes(recognized_word)
            
            # Calculate phonetic distance
            phonetic_distance = self._calculate_phonetic_distance(
                expected_phonemes, recognized_phonemes
            )
            
            # Expected confidence based on phonetic similarity
            phonetic_similarity = 1.0 - phonetic_distance
            expected_confidence = phonetic_similarity * 0.8 + 0.2  # Base confidence
            
            # Check consistency
            confidence_diff = abs(confidence - expected_confidence)
            is_consistent = confidence_diff < self.correction_indicators['phonetic_distance_threshold']
            
            return {
                'is_consistent': is_consistent,
                'phonetic_distance': phonetic_distance,
                'phonetic_similarity': phonetic_similarity,
                'expected_confidence': expected_confidence,
                'actual_confidence': confidence,
                'confidence_diff': confidence_diff
            }
            
        except Exception as e:
            print(f"⚠️ Phonetic consistency check failed: {e}")
            return {'is_consistent': True}  # Assume consistent if check fails
    
    def _calculate_phonetic_distance(self, phonemes1: List[str], phonemes2: List[str]) -> float:
        """Calculate phonetic distance between two phoneme sequences"""
        
        if not phonemes1 or not phonemes2:
            return 1.0  # Maximum distance for missing sequences
        
        # Use dynamic programming for sequence alignment
        m, n = len(phonemes1), len(phonemes2)
        dp = [[0.0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Initialize with deletion/insertion costs
        for i in range(m + 1):
            dp[i][0] = i * 0.5  # Deletion cost
        for j in range(n + 1):
            dp[0][j] = j * 0.5  # Insertion cost
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if phonemes1[i-1] == phonemes2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No cost for exact match
                else:
                    # Calculate substitution cost based on phonetic similarity
                    substitution_cost = self._phoneme_substitution_cost(
                        phonemes1[i-1], phonemes2[j-1]
                    )
                    
                    dp[i][j] = min(
                        dp[i-1][j] + 0.5,      # Deletion
                        dp[i][j-1] + 0.5,      # Insertion
                        dp[i-1][j-1] + substitution_cost  # Substitution
                    )
        
        # Normalize by maximum possible distance
        max_distance = max(m, n) * 0.5
        return dp[m][n] / max_distance if max_distance > 0 else 0.0
    
    def _phoneme_substitution_cost(self, phoneme1: str, phoneme2: str) -> float:
        """Calculate cost of substituting one phoneme with another"""
        
        # Phonetic feature similarity groups
        feature_groups = {
            'stops': {'p', 'b', 't', 'd', 'k', 'g'},
            'fricatives': {'f', 'v', 'θ', 'ð', 's', 'z', 'ʃ', 'ʒ', 'h'},
            'affricates': {'tʃ', 'dʒ'},
            'nasals': {'m', 'n', 'ŋ'},
            'liquids': {'l', 'r'},
            'glides': {'w', 'j'},
            'vowels_front': {'iː', 'ɪ', 'eɪ', 'ɛ', 'æ'},
            'vowels_central': {'ə', 'ʌ', 'ɜːr'},
            'vowels_back': {'uː', 'ʊ', 'oʊ', 'ɔː', 'ɑː'},
            'diphthongs': {'aɪ', 'aʊ', 'ɔɪ'}
        }
        
        # Find groups for each phoneme
        groups1 = [group for group, phonemes in feature_groups.items() if phoneme1 in phonemes]
        groups2 = [group for group, phonemes in feature_groups.items() if phoneme2 in phonemes]
        
        # Calculate similarity based on shared groups
        if groups1 and groups2:
            shared_groups = set(groups1) & set(groups2)
            if shared_groups:
                return 0.2  # Low cost for same category
            else:
                return 0.8  # High cost for different categories
        else:
            return 1.0  # Maximum cost for unknown phonemes
    
    def _validate_timing_patterns(self, timing_data: Dict, word: str) -> Dict:
        """Validate timing patterns for naturalness"""
        
        word_duration = timing_data.get('end', 0) - timing_data.get('start', 0)
        
        # Expected duration based on word characteristics
        expected_duration = self._estimate_word_duration(word)
        
        # Check if duration is reasonable
        duration_ratio = word_duration / expected_duration if expected_duration > 0 else 1.0
        
        # Natural timing: 0.5x to 3x expected duration
        is_natural = 0.5 <= duration_ratio <= 3.0
        
        return {
            'is_natural': is_natural,
            'actual_duration': word_duration,
            'expected_duration': expected_duration,
            'duration_ratio': duration_ratio
        }
    
    def _estimate_word_duration(self, word: str) -> float:
        """Estimate expected duration for a word"""
        
        # Base duration factors
        base_duration = 0.1  # 100ms base
        
        # Add time for each phoneme (rough estimate)
        try:
            phonemes = self.phoneme_engine.get_phonemes(word)
            phoneme_duration = len(phonemes) * 0.08  # 80ms per phoneme
        except:
            phoneme_duration = len(word) * 0.06  # Fallback: 60ms per character
        
        # Add time for complex features
        complexity_factor = 0.0
        if any(cluster in word.lower() for cluster in ['str', 'spr', 'scr', 'thr']):
            complexity_factor += 0.05
        
        return base_duration + phoneme_duration + complexity_factor
    
    def _validate_acoustic_features(self, audio_features: Dict, expected_word: str, recognized_word: str) -> Dict:
        """Validate acoustic features for consistency"""
        
        quality_factors = []
        
        # Signal quality
        snr = audio_features.get('snr_estimate', 15)
        quality_factors.append(min(1.0, snr / 20))  # 0-20 dB range
        
        # Dynamic range
        dynamic_range = audio_features.get('dynamic_range', 0.1)
        quality_factors.append(min(1.0, dynamic_range * 5))
        
        # Clipping
        clipping_ratio = audio_features.get('clipping_ratio', 0)
        quality_factors.append(max(0.0, 1.0 - clipping_ratio * 5))
        
        # Frequency response
        freq_quality = audio_features.get('frequency_quality', 50)
        quality_factors.append(freq_quality / 100)
        
        overall_quality = np.mean(quality_factors)
        
        return {
            'quality_score': overall_quality,
            'individual_factors': quality_factors,
            'is_high_quality': overall_quality > 0.7
        }
    
    def _calculate_raw_acoustic_score(self, audio_features: Dict, word: str, timing_data: Dict) -> float:
        """Calculate pronunciation score based on raw acoustic features"""
        
        # Start with base score
        base_score = 0.5
        
        # Adjust based on acoustic quality
        quality_score = audio_features.get('overall_quality', 50) / 100
        acoustic_score = base_score + (quality_score - 0.5) * 0.4
        
        # Adjust based on timing reasonableness
        timing_validation = self._validate_timing_patterns(timing_data, word)
        if timing_validation['is_natural']:
            acoustic_score += 0.1
        else:
            acoustic_score -= 0.2
        
        # Adjust based on word difficulty
        difficulty = self._calculate_word_difficulty(word)
        acoustic_score -= difficulty * 0.2
        
        return max(0.0, min(1.0, acoustic_score)) * 100
    
    def _calculate_word_difficulty(self, word: str) -> float:
        """Calculate word pronunciation difficulty (0-1)"""
        
        difficulty_factors = {
            'length': min(0.3, len(word) * 0.02),
            'consonant_clusters': self._count_consonant_clusters(word) * 0.1,
            'difficult_phonemes': self._count_difficult_phonemes(word) * 0.15,
            'syllable_complexity': self._assess_syllable_complexity(word) * 0.1
        }
        
        return min(1.0, sum(difficulty_factors.values()))
    
    def _count_consonant_clusters(self, word: str) -> int:
        """Count consonant clusters in word"""
        clusters = ['str', 'spr', 'scr', 'thr', 'spl', 'sch', 'chr', 'shr']
        count = 0
        word_lower = word.lower()
        for cluster in clusters:
            count += word_lower.count(cluster)
        return count
    
    def _count_difficult_phonemes(self, word: str) -> int:
        """Count difficult phonemes in word"""
        try:
            phonemes = self.phoneme_engine.get_phonemes(word)
            difficult_phonemes = {'θ', 'ð', 'ŋ', 'ʒ', 'ɜːr', 'r'}
            return sum(1 for p in phonemes if p in difficult_phonemes)
        except:
            # Fallback: count difficult letter combinations
            difficult_combinations = ['th', 'ng', 'r']
            count = 0
            word_lower = word.lower()
            for combo in difficult_combinations:
                count += word_lower.count(combo)
            return count
    
    def _assess_syllable_complexity(self, word: str) -> float:
        """Assess syllable complexity (rough estimate)"""
        # Simple vowel counting as syllable estimate
        vowels = 'aeiouAEIOU'
        vowel_count = sum(1 for char in word if char in vowels)
        
        # More vowels = more syllables = more complexity
        return min(1.0, vowel_count * 0.2)
    
    def _calculate_validation_confidence(self, audio_features: Dict, timing_data: Dict, phoneme_scores: List[float]) -> float:
        """Calculate confidence in the validation itself"""
        
        confidence_factors = []
        
        # Audio quality confidence
        audio_quality = audio_features.get('overall_quality', 50) / 100
        confidence_factors.append(audio_quality)
        
        # Timing data availability
        if timing_data.get('start') is not None and timing_data.get('end') is not None:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        # Phoneme score consistency
        if phoneme_scores:
            score_std = np.std(phoneme_scores)
            consistency = max(0.0, 1.0 - score_std * 2)
            confidence_factors.append(consistency)
        else:
            confidence_factors.append(0.5)
        
        return np.mean(confidence_factors)
    
    def get_correction_mitigation_strategy(self, validation_result: ValidationResult) -> Dict[str, any]:
        """Get strategy for handling detected corrections"""
        
        strategy = {
            'use_raw_acoustic': validation_result.correction_likelihood > 0.6,
            'apply_penalty': validation_result.correction_likelihood > 0.4,
            'require_verification': validation_result.correction_likelihood > 0.7,
            'penalty_factor': max(0.5, 1.0 - validation_result.correction_likelihood),
            'alternative_score': validation_result.raw_acoustic_score,
            'confidence_adjustment': max(0.3, 1.0 - validation_result.correction_likelihood * 0.5)
        }
        
        return strategy 