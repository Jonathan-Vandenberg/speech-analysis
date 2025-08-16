from typing import List, Optional

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


class AnalyzeResponse(BaseModel):
    pronunciation: PronunciationResult
    predicted_text: str
    metrics: Optional[SpeechMetrics] = None


