import numpy as np
import time
import asyncio
from typing import Optional
from .pronunciation import PronunciationAnalyzer
from .relevance import RelevanceAnalyzer
from .grammar import GrammarCorrector
from .scoring import SpeechScorer
from models.responses import FreestyleSpeechResponse

class FreestyleSpeechAnalyzer:
    """
    Main orchestrator for freestyle speech analysis combining pronunciation, relevance, grammar, and scoring
    """
    
    def __init__(
        self,
        pronunciation_analyzer: PronunciationAnalyzer,
        relevance_analyzer: RelevanceAnalyzer,
        grammar_corrector: GrammarCorrector,
        speech_scorer: SpeechScorer
    ):
        self.pronunciation_analyzer = pronunciation_analyzer
        self.relevance_analyzer = relevance_analyzer
        self.grammar_corrector = grammar_corrector
        self.speech_scorer = speech_scorer
    
    async def analyze_freestyle_speech(
        self,
        audio_data: np.ndarray,
        samplerate: int,
        question: str,
        scoring_criteria: str = "ielts",
        expected_language_level: str = "intermediate"
    ) -> FreestyleSpeechResponse:
        """
        Perform comprehensive freestyle speech analysis using optimized parallel processing
        """
        start_time = time.time()
        
        # Step 1: Get transcription and pronunciation analysis in one unified step
        print("🎤 Transcribing speech and analyzing pronunciation...")
        transcribed_text, pronunciation_result = await self.pronunciation_analyzer.analyze_pronunciation_freestyle(
            audio_data, samplerate
        )
        
        if not transcribed_text or len(transcribed_text.strip()) < 3:
            return self._create_minimal_response(
                "Speech could not be transcribed clearly",
                question,
                int((time.time() - start_time) * 1000)
            )
        
        print(f"📝 Transcribed: '{transcribed_text}'")
        print(f"🗣️ Pronunciation score: {pronunciation_result.overall_score}%")
        
        # Step 1.5: Create transcription with fillers for grammar analysis
        transcription_with_fillers = self._create_transcription_with_fillers(
            transcribed_text, pronunciation_result
        )
        print(f"📝 With fillers: '{transcription_with_fillers}'")
        
        # Step 2 & 3: Run grammar and relevance analysis in parallel for efficiency
        print("📝 Correcting grammar and analyzing relevance in parallel...")
        grammar_task = self.grammar_corrector.correct_grammar(transcription_with_fillers, question)
        relevance_task = self.relevance_analyzer.analyze_relevance(question, transcribed_text)
        
        # Wait for both tasks to complete
        grammar_result, relevance_result = await asyncio.gather(grammar_task, relevance_task)
        print(f"✅ Grammar and relevance analysis completed")
        print(f"✅ Grammar result type: {type(grammar_result)}")
        print(f"✅ Grammar corrected text: '{grammar_result.corrected_text[:100] if grammar_result.corrected_text else 'None'}...'")
        print(f"✅ Relevance score: {relevance_result.relevance_score}")
        
        # Step 4: Calculate scores
        print("📊 Calculating scores...")
        ielts_score = None
        cefr_score = None
        
        if scoring_criteria in ["ielts", "both"]:
            ielts_score = self.speech_scorer.calculate_ielts_score(
                pronunciation_result, grammar_result, relevance_result
            )
        
        if scoring_criteria in ["cefr", "both"]:
            cefr_score = self.speech_scorer.calculate_cefr_score(
                pronunciation_result, grammar_result, relevance_result
            )
        
        # Calculate overall confidence
        confidence_level = self._calculate_confidence(
            pronunciation_result, relevance_result, grammar_result
        )
        
        end_time = time.time()
        processing_time_ms = int((end_time - start_time) * 1000)
        
        print(f"⚡ Total freestyle analysis took {processing_time_ms} ms")
        
        return FreestyleSpeechResponse(
            transcribed_text=transcribed_text,
            pronunciation=pronunciation_result,
            relevance=relevance_result,
            grammar=grammar_result,
            ielts_score=ielts_score,
            cefr_score=cefr_score,
            processing_time_ms=processing_time_ms,
            confidence_level=confidence_level
        )
    
    def _calculate_confidence(
        self,
        pronunciation_result,
        relevance_result,
        grammar_result
    ) -> float:
        """
        Calculate overall confidence in the analysis
        """
        # Base confidence on score consistency and speech quality
        scores = [
            pronunciation_result.overall_score,
            relevance_result.relevance_score,
            grammar_result.grammar_score
        ]
        
        # Calculate variance - lower variance = higher confidence
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = max(60, min(95, 90 - variance * 0.3))
        
        # Adjust based on transcription quality (length and content)
        if pronunciation_result.words and len(pronunciation_result.words) >= 3:
            confidence += 5  # Bonus for substantial speech
        
        return round(confidence, 1)
    
    def _create_minimal_response(
        self, transcribed_text: str, question: str, processing_time_ms: int
    ) -> FreestyleSpeechResponse:
        """
        Create a minimal response when analysis fails
        """
        from models.responses import (
            PronunciationAssessmentResponse, RelevanceAnalysis, 
            GrammarCorrection, IELTSScore, CEFRScore
        )
        
        # Create minimal responses
        minimal_pronunciation = PronunciationAssessmentResponse(
            overall_score=30.0,
            words=[],
            processing_time_ms=100
        )
        
        minimal_relevance = RelevanceAnalysis(
            relevance_score=25.0,
            explanation="Could not analyze due to unclear speech",
            key_points_covered=[],
            missing_points=["Clear speech required for analysis"]
        )
        
        minimal_grammar = GrammarCorrection(
            original_text=transcribed_text,
            corrected_text=transcribed_text,
            differences=[],
            tagged_text=transcribed_text,
            lexical_analysis="Analysis unavailable due to unclear speech",
            strengths=[],
            improvements=["Speak more clearly for detailed analysis"],
            lexical_band_score=3.0,
            model_answers={
                "band4": "Basic response needed",
                "band5": "Simple but clear response",
                "band6": "Adequate response with variety",
                "band7": "Good response with development", 
                "band8": "Very good response with sophistication",
                "band9": "Excellent response with precision"
            },
            grammar_score=30.0
        )
        
        return FreestyleSpeechResponse(
            transcribed_text=transcribed_text,
            pronunciation=minimal_pronunciation,
            relevance=minimal_relevance,
            grammar=minimal_grammar,
            ielts_score=None,
            cefr_score=None,
            processing_time_ms=processing_time_ms,
            confidence_level=20.0
        )

    def _create_transcription_with_fillers(
        self,
        transcribed_text: str,
        pronunciation_result
    ) -> str:
        """
        Create a transcription with fillers and repetitions inserted at their detected positions for grammar analysis
        """
        # Check if we have fluency metrics
        if not pronunciation_result.fluency_metrics:
            return transcribed_text
        
        # Get detected fillers and repetitions
        fillers = pronunciation_result.fluency_metrics.filler_words or []
        repetitions = pronunciation_result.fluency_metrics.repetitions or []
        
        # If no fillers or repetitions, return original text
        if not fillers and not repetitions:
            return transcribed_text
        
        print(f"🔤 Creating transcription with {len(fillers)} fillers and {len(repetitions)} repetitions")
        
        # Split transcription into words with positions
        words = transcribed_text.split()
        
        # Create a list of all speech elements (words + fillers + repetitions) with their start times
        speech_elements = []
        
        # Add original words with estimated timing
        word_duration = 0.8  # Estimate 0.8 seconds per word average
        for i, word in enumerate(words):
            estimated_time = i * word_duration
            speech_elements.append({
                'type': 'word',
                'text': word,
                'start_time': estimated_time,
                'priority': 1  # Original words have priority
            })
        
        # Add detected fillers
        for filler in fillers:
            speech_elements.append({
                'type': 'filler',
                'text': filler.word,
                'start_time': filler.start_time,
                'priority': 2  # Fillers have lower priority
            })
            print(f"📝 Adding filler '{filler.word}' at {filler.start_time:.2f}s")
        
        # Add repetitions by inserting repeated words
        for repetition in repetitions:
            if repetition.occurrences and len(repetition.occurrences) > 1:
                # Add additional occurrences (first is usually already in main text)
                for i, occurrence in enumerate(repetition.occurrences[1:], 1):
                    speech_elements.append({
                        'type': 'repetition',
                        'text': occurrence['text'],
                        'start_time': occurrence['start_time'],
                        'priority': 2  # Repetitions have lower priority
                    })
                    print(f"📝 Adding repetition '{occurrence['text']}' at {occurrence['start_time']:.2f}s")
        
        # Sort all elements by start time, then by priority
        speech_elements.sort(key=lambda x: (x['start_time'], x['priority']))
        
        # Reconstruct text with fillers and repetitions included
        result_words = []
        for element in speech_elements:
            result_words.append(element['text'])
        
        result_text = ' '.join(result_words)
        
        print(f"🔤 Original: '{transcribed_text}'")
        print(f"🔤 With fillers: '{result_text}'")
        print(f"🔤 Added {len(fillers)} fillers and {sum(max(0, len(rep.occurrences) - 1) for rep in repetitions)} repetitions")
        
        return result_text 