from typing import Optional
from models.responses import IELTSScore, CEFRScore, PronunciationAssessmentResponse, GrammarCorrection, RelevanceAnalysis

class SpeechScorer:
    """
    Calculates IELTS and CEFR scores based on pronunciation, grammar, and relevance analysis
    """
    
    def __init__(self):
        # CEFR level mapping
        self.cefr_levels = ["A1", "A2", "B1", "B2", "C1", "C2"]
    
    def calculate_ielts_score(
        self, 
        pronunciation: PronunciationAssessmentResponse,
        grammar: GrammarCorrection,
        relevance: RelevanceAnalysis,
        fluency_indicators: dict = None
    ) -> IELTSScore:
        """
        Calculate IELTS band score (0-9 scale) - CONSERVATIVE and realistic scoring
        """
        # Extract scores and apply realistic penalties
        pronunciation_score = pronunciation.overall_score / 100 * 9  # Convert to 0-9
        grammar_score = grammar.grammar_score / 100 * 9
        relevance_score = relevance.relevance_score / 100 * 9
        
        # Estimate fluency from speech characteristics
        fluency_score = self._estimate_fluency_score(pronunciation, fluency_indicators)
        
        # IMPORTANT: Apply heavy penalties for simple speech with errors
        # Simple sentences should not score above 6.0 if they have fillers/errors
        sentence_complexity = self._assess_sentence_complexity(pronunciation, grammar, relevance)
        
        # CONSERVATIVE scoring adjustments
        if sentence_complexity == 'very_simple':
            # Very simple sentences (1-8 words) with fillers should score max 5.5
            max_possible_score = 5.5
            if grammar.grammar_score < 90:  # Any grammar issues
                max_possible_score = 5.0
            if fluency_score < 80:  # Fillers or disfluencies
                max_possible_score = 4.5
        elif sentence_complexity == 'simple':
            # Simple sentences (9-15 words) max 6.5 
            max_possible_score = 6.5
            if grammar.grammar_score < 85:
                max_possible_score = 6.0
            if fluency_score < 75:
                max_possible_score = 5.5
        else:
            # Complex sentences can score higher
            max_possible_score = 9.0
        
        # Calculate individual band scores with realistic caps
        fluency_coherence = min(max_possible_score, round((fluency_score + relevance_score) / 2, 1))
        lexical_resource = min(max_possible_score, min(8.0, relevance_score + 0.5))  # Vocabulary usage
        grammatical_range = min(max_possible_score, grammar_score)
        pronunciation_band = min(max_possible_score, pronunciation_score)
        
        # Overall band calculation (weighted average) - much more conservative
        overall_band = (
            fluency_coherence * 0.25 +
            lexical_resource * 0.25 +
            grammatical_range * 0.25 +
            pronunciation_band * 0.25
        )
        
        # Apply maximum cap based on complexity
        overall_band = min(overall_band, max_possible_score)
        
        # Round to nearest 0.5
        overall_band = round(overall_band * 2) / 2
        overall_band = max(1.0, min(9.0, overall_band))
        
        return IELTSScore(
            overall_band=overall_band,
            fluency_coherence=max(1.0, min(9.0, fluency_coherence)),
            lexical_resource=max(1.0, min(9.0, lexical_resource)),
            grammatical_range=max(1.0, min(9.0, grammatical_range)),
            pronunciation=max(1.0, min(9.0, pronunciation_band)),
            explanation=self._generate_ielts_explanation(overall_band, sentence_complexity)
        )
    
    def calculate_cefr_score(
        self,
        pronunciation: PronunciationAssessmentResponse,
        grammar: GrammarCorrection,
        relevance: RelevanceAnalysis
    ) -> CEFRScore:
        """
        Calculate CEFR level (A1-C2)
        """
        # Convert scores to CEFR scale
        scores = [
            pronunciation.overall_score,
            grammar.grammar_score,
            relevance.relevance_score
        ]
        
        average_score = sum(scores) / len(scores)
        
        # Map to CEFR levels
        if average_score >= 90:
            level = "C2"
            confidence = 95
        elif average_score >= 80:
            level = "C1"
            confidence = 90
        elif average_score >= 70:
            level = "B2"
            confidence = 85
        elif average_score >= 60:
            level = "B1"
            confidence = 80
        elif average_score >= 45:
            level = "A2"
            confidence = 75
        else:
            level = "A1"
            confidence = 70
        
        # Adjust confidence based on score consistency
        score_variance = max(scores) - min(scores)
        if score_variance > 20:
            confidence -= 10  # Lower confidence for inconsistent scores
        
        breakdown = {
            "pronunciation": self._score_to_cefr_level(pronunciation.overall_score),
            "grammar": self._score_to_cefr_level(grammar.grammar_score),
            "content_relevance": self._score_to_cefr_level(relevance.relevance_score)
        }
        
        return CEFRScore(
            overall_level=level,
            confidence=max(50, min(100, confidence)),
            breakdown=breakdown,
            explanation=self._generate_cefr_explanation(level, average_score)
        )
    
    def _estimate_fluency_score(self, pronunciation: PronunciationAssessmentResponse, indicators: dict = None) -> float:
        """
        Estimate fluency based on available indicators
        """
        if not indicators:
            # Basic estimation from pronunciation data
            if pronunciation.processing_time_ms and pronunciation.words:
                # Estimate speaking rate
                total_duration = sum((w.end_time or 0) - (w.start_time or 0) for w in pronunciation.words)
                if total_duration > 0:
                    words_per_second = len(pronunciation.words) / total_duration
                    # Normal speaking rate is around 2-3 words per second
                    if 1.5 <= words_per_second <= 4.0:
                        fluency_estimate = 7.0
                    elif words_per_second < 1.5:
                        fluency_estimate = 5.0  # Too slow
                    else:
                        fluency_estimate = 6.0  # Too fast
                else:
                    fluency_estimate = 6.0
            else:
                fluency_estimate = 6.0
        else:
            # Use provided indicators (pauses, hesitations, etc.)
            fluency_estimate = indicators.get('fluency_score', 6.0)
        
        return min(9.0, max(1.0, fluency_estimate))
    
    def _score_to_cefr_level(self, score: float) -> str:
        """Convert percentage score to CEFR level"""
        if score >= 90:
            return "C2"
        elif score >= 80:
            return "C1"
        elif score >= 70:
            return "B2"
        elif score >= 60:
            return "B1"
        elif score >= 45:
            return "A2"
        else:
            return "A1"
    
    def _generate_ielts_explanation(self, band: float, sentence_complexity: str) -> str:
        """Generate explanation for IELTS band score"""
        base_explanation = ""
        if band >= 8.5:
            base_explanation = "Excellent command of English with rare minor errors"
        elif band >= 7.5:
            base_explanation = "Very good command with occasional errors that don't impede communication"
        elif band >= 6.5:
            base_explanation = "Good command with some errors but meaning is clear"
        elif band >= 5.5:
            base_explanation = "Modest command with noticeable errors but basic communication is effective"
        elif band >= 4.5:
            base_explanation = "Limited command with frequent errors affecting communication"
        else:
            base_explanation = "Basic command with significant limitations in communication"
        
        # Add complexity-specific context
        if sentence_complexity == 'very_simple' and band < 6.0:
            base_explanation += " (Simple sentence with fillers/errors limits higher scoring)"
        elif sentence_complexity == 'simple' and band < 7.0:
            base_explanation += " (Basic sentence structure with some disfluencies)"
        
        return base_explanation
    
    def _generate_cefr_explanation(self, level: str, score: float) -> str:
        """Generate explanation for CEFR level"""
        explanations = {
            "C2": "Mastery level - Can communicate with near-native fluency and precision",
            "C1": "Advanced level - Can communicate effectively in complex situations",
            "B2": "Upper-intermediate - Can handle complex topics with good fluency",
            "B1": "Intermediate - Can handle routine topics and familiar situations",
            "A2": "Elementary - Can communicate in simple, routine tasks",
            "A1": "Beginner - Can communicate basic information in familiar contexts"
        }
        return explanations.get(level, "Assessment completed")
    
    def _assess_sentence_complexity(self, pronunciation, grammar, relevance) -> str:
        """Assess the complexity of the spoken content"""
        word_count = len(pronunciation.words) if pronunciation.words else 0
        
        if word_count <= 8:
            return 'very_simple'
        elif word_count <= 15:
            return 'simple'
        else:
            return 'complex' 