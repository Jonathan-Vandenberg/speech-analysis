"""
Transcription endpoint for standalone Whisper transcription
"""
import logging
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from .middleware import api_key_bearer, APIKeyInfo, request_tracker, generate_request_id
from .routes_unscripted import transcribe_faster_whisper, load_audio_to_mono16k
from .schemas import ErrorResponse

logger = logging.getLogger("speech_analyzer")
router = APIRouter()

@router.post(
    "/transcribe",
    summary="Audio Transcription",
    description="""
    Transcribe audio using Whisper AI model.
    
    **Parameters:**
    - `file`: Audio file (required) - Supports WAV, MP3, M4A, WEBM, OGG
    - `language`: Language code (optional, default: 'en') 
    - `response_format`: Response format (optional, default: 'json')
    
    **Returns:**
    - `text`: Transcribed text from the audio
    """,
    tags=["Transcription"],
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "example": {
                        "text": "Hello, this is the transcribed text from your audio."
                    }
                }
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid audio file"},
        500: {"model": ErrorResponse, "description": "Transcription failed"}
    }
)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    language: Optional[str] = Form("en", description="Language code (e.g., 'en', 'es', 'fr')"),
    response_format: Optional[str] = Form("json", description="Response format (json, text)"),
    api_key_info: Optional[APIKeyInfo] = Depends(api_key_bearer),
):
    """Standalone Whisper transcription endpoint."""
    
    # Generate request ID and start tracking
    request_id = generate_request_id()
    request_tracker.start_request(request_id, api_key_info, "transcribe", request)
    
    try:
        # Validate content type
        if not file.content_type or not (
            file.content_type.startswith("audio/") or 
            file.filename.lower().endswith((".wav", ".mp3", ".m4a", ".webm", ".ogg"))
        ):
            logger.warning(f"Invalid content type: {file.content_type}")
            raise HTTPException(
                status_code=400, 
                detail="Please upload an audio file (WAV, MP3, M4A, WEBM, or OGG format)."
            )
        
        logger.info(f"🎤 [TRANSCRIBE] Processing audio file: {file.filename} ({file.size} bytes)")
        
        # Load and process audio
        audio_data = await file.read()
        audio = load_audio_to_mono16k(audio_data)
        
        logger.info(f"🎤 [TRANSCRIBE] Audio loaded successfully, duration: {len(audio)/16000:.2f}s")
        
        # Transcribe using Whisper
        transcript = transcribe_faster_whisper(audio)
        
        if not transcript or not transcript.strip():
            logger.error("Whisper returned empty transcript")
            raise HTTPException(
                status_code=500, 
                detail="Could not transcribe audio. Please check audio quality and ensure it contains speech."
            )
        
        logger.info(f"🎤 [TRANSCRIBE] Transcription successful: '{transcript[:50]}{'...' if len(transcript) > 50 else ''}'")
        
        # Track successful completion
        form_data = {
            "language": language,
            "response_format": response_format
        }
        response_data = {
            "text": transcript.strip(),
            "transcript_length": len(transcript),
            "audio_duration_ms": int(len(audio) / 16000 * 1000)
        }
        await request_tracker.finish_request(request_id, response_data, form_data, audio_data)
        
        # Return response
        return {"text": transcript.strip()}
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
        
    except Exception as e:
        logger.error(f"🎤 [TRANSCRIBE] Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal transcription error: {str(e)}"
        )
