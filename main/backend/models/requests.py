from pydantic import BaseModel, Field
from typing import Optional

class PronunciationAssessmentRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data.")
    audio_format: str = Field(..., min_length=2, max_length=4, description="Format of the audio (e.g., 'wav', 'mp3', 'webm').")
    expected_text: str = Field(..., min_length=1, description="The text the user was expected to say.")

class FreestyleSpeechRequest(BaseModel):
    audio_base64: str = Field(..., description="Base64 encoded audio data.")
    audio_format: str = Field(..., min_length=2, max_length=4, description="Format of the audio (e.g., 'wav', 'mp3', 'webm').")
    question: str = Field(..., min_length=1, description="The question that was asked to the user.")
    expected_language_level: Optional[str] = Field("intermediate", description="Expected CEFR level (A1, A2, B1, B2, C1, C2) for calibrated scoring.")
    scoring_criteria: Optional[str] = Field("ielts", description="Scoring system to use (ielts, cefr, or both).") 