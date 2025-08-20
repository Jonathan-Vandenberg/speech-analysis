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


# Admin API Models
class APIKeyInfo(BaseModel):
    """API Key information for admin interface"""
    id: str
    description: str
    is_active: bool
    usage_count: int
    minute_usage: int
    daily_usage: int
    monthly_usage: int
    minute_limit: int
    daily_limit: int
    monthly_limit: int
    last_used_at: Optional[str] = None
    created_at: str


class APIKeyCreateRequest(BaseModel):
    """Request model for creating new API keys"""
    description: str
    minute_limit: int = 10
    daily_limit: int = 1000
    monthly_limit: int = 10000


class APIKeyCreateResponse(BaseModel):
    """Response model for API key creation"""
    api_key: str
    key_id: str
    description: str
    minute_limit: int
    daily_limit: int
    monthly_limit: int


class APIKeysListResponse(BaseModel):
    """Response model for listing API keys"""
    api_keys: List[APIKeyInfo]


class APIKeyUpdateRequest(BaseModel):
    """Request model for updating API keys"""
    description: Optional[str] = None
    is_active: Optional[bool] = None
    minute_limit: Optional[int] = None
    daily_limit: Optional[int] = None
    monthly_limit: Optional[int] = None


class UsageAnalyticsResponse(BaseModel):
    """Response model for usage analytics"""
    api_keys: List[Dict[str, Any]]
    recent_logs: List[Dict[str, Any]]


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    database: str
    version: str


class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str


