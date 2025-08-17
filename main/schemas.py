from typing import List, Optional, Dict, Any

from pydantic import BaseModel


class PhonemeScore(BaseModel):
    ipa_label: str
    phoneme_score: float


class WordPronunciation(BaseModel):
    word_text: str
    phonemes: List[PhonemeScore]
    word_score: float


class PronunciationResult(BaseModel):
    words: List[WordPronunciation]
    overall_score: float


class PauseDetail(BaseModel):
    start_index: int
    end_index: int
    duration: float


class DiscourseMarker(BaseModel):
    text: str
    start_index: int
    end_index: int
    description: str


class FillerWordDetail(BaseModel):
    text: str
    start_index: int
    end_index: int
    phonemes: str


class Repetition(BaseModel):
    text: str
    start_index: int
    end_index: int
    repeat_count: int


class SpeechMetrics(BaseModel):
    speech_rate: float
    speech_rate_over_time: List[float]
    pauses: int
    filler_words: int
    discourse_markers: List[DiscourseMarker]
    filler_words_per_min: float
    pause_details: List[PauseDetail]
    repetitions: List[Repetition]
    filler_words_details: List[FillerWordDetail]


class GrammarDifference(BaseModel):
    type: str  # 'addition', 'deletion', 'substitution'
    original: Optional[str]
    corrected: Optional[str]
    position: int


class GrammarCorrection(BaseModel):
    original_text: str
    corrected_text: str
    differences: List[GrammarDifference]
    taggedText: str
    lexical_analysis: str
    strengths: List[str]
    improvements: List[str]
    lexical_band_score: float
    modelAnswers: Dict[str, Dict[str, str]]
    grammar_score: float


class RelevanceAnalysis(BaseModel):
    relevance_score: float
    explanation: str
    key_points_covered: List[str]
    missing_points: List[str]


class IELTSScore(BaseModel):
    overall_band: float
    fluency_coherence: float
    lexical_resource: float
    grammatical_range: float
    pronunciation: float
    explanation: str


class AnalyzeResponse(BaseModel):
    pronunciation: PronunciationResult
    predicted_text: str
    metrics: Optional[SpeechMetrics] = None
    grammar: Optional[GrammarCorrection] = None
    relevance: Optional[RelevanceAnalysis] = None
    ielts_score: Optional[IELTSScore] = None


