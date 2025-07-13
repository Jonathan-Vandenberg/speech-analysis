from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class PhonemeScore(BaseModel):
    ipa_label: str
    phoneme_score: float

class WordScore(BaseModel):
    word_text: str
    word_score: float
    phonemes: List[PhonemeScore] = []
    start_time: Optional[float] = None
    end_time: Optional[float] = None

class FillerWord(BaseModel):
    word: str
    start_time: float
    end_time: float
    type: str  # 'hesitation', 'discourse_marker', 'repetition'

class Pause(BaseModel):
    start_time: float
    end_time: float
    duration: float  # in seconds

class Repetition(BaseModel):
    repeated_text: str
    occurrences: List[dict]  # [{'start_time': float, 'end_time': float, 'text': str}]
    count: int

class FluencyMetrics(BaseModel):
    overall_fluency_score: float  # 0-100
    filler_words: List[FillerWord]
    long_pauses: List[Pause]
    repetitions: List[Repetition]
    speech_rate: float  # words per minute (average)
    speech_rate_over_time: List[dict]  # [{'time': float, 'rate': float, 'segment_text': str}]
    total_filler_count: int
    total_pause_time: float  # seconds
    total_repetition_count: int

class PronunciationAssessmentResponse(BaseModel):
    overall_score: float
    words: List[WordScore] = []
    fluency_metrics: Optional[FluencyMetrics] = None
    processing_time_ms: Optional[int] = None

class GrammarDifference(BaseModel):
    type: str  # 'addition', 'deletion', 'substitution'
    original: Optional[str] = None
    corrected: Optional[str] = None
    position: int

class GrammarCorrection(BaseModel):
    original_text: str
    corrected_text: str
    differences: List[GrammarDifference] = []
    tagged_text: str  # Original text with <grammar-mistake> tags
    lexical_analysis: str  # Detailed lexical resource analysis
    strengths: List[str]  # Good points in the response
    improvements: List[str]  # Areas for improvement
    lexical_band_score: float  # 0-9 for lexical resource
    model_answers: Dict[str, str] = {}  # band4 through band9 examples
    grammar_score: float  # 0-100

class RelevanceAnalysis(BaseModel):
    relevance_score: float  # 0-100
    explanation: str
    key_points_covered: List[str]
    missing_points: List[str]

class IELTSScore(BaseModel):
    overall_band: float  # 0-9
    fluency_coherence: float
    lexical_resource: float  
    grammatical_range: float
    pronunciation: float
    explanation: str

class CEFRScore(BaseModel):
    overall_level: str  # A1, A2, B1, B2, C1, C2
    confidence: float  # 0-100
    breakdown: Dict[str, str]  # Different skill areas
    explanation: str

class FreestyleSpeechResponse(BaseModel):
    # Transcription
    transcribed_text: str
    
    # Pronunciation Analysis
    pronunciation: PronunciationAssessmentResponse
    
    # Relevance Analysis
    relevance: RelevanceAnalysis
    
    # Grammar Correction
    grammar: GrammarCorrection
    
    # Scoring
    ielts_score: Optional[IELTSScore] = None
    cefr_score: Optional[CEFRScore] = None
    
    # Overall
    processing_time_ms: int
    confidence_level: float  # Overall confidence in analysis 