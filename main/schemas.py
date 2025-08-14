from typing import List

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


class AnalyzeResponse(BaseModel):
    pronunciation: PronunciationResult
    predicted_text: str


