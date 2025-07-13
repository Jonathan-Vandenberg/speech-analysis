import os
import base64
import time
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from supabase import create_client, Client
import hashlib

# Import modular components
from analyzers import PronunciationAnalyzer, FreestyleSpeechAnalyzer, RelevanceAnalyzer, GrammarCorrector, SpeechScorer
from models.requests import PronunciationAssessmentRequest, FreestyleSpeechRequest
from models.responses import PronunciationAssessmentResponse, FreestyleSpeechResponse
from utils import calculate_audio_hash, convert_audio_to_wav, prepare_audio_data

# Load environment variables
load_dotenv()

# Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PRONUNCIATION_ANALYSIS_APP_ID = os.getenv("PRONUNCIATION_ANALYSIS_APP_ID")

# Initialize Supabase client
supabase_client: Client | None = None

if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✅ Supabase client initialized successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not initialize Supabase client: {e}")
else:
    print("❌ CRITICAL ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY. Supabase client not initialized.")

if not OPENAI_API_KEY:
    print("⚠️ WARNING: OPENAI_API_KEY is not set. AI-powered analysis features will be disabled.")

# Initialize analyzers with professional forced alignment
print("🚀 Initializing Professional Speech Analysis System...")
pronunciation_analyzer = PronunciationAnalyzer()  # No longer needs Vosk model path

# Initialize other analyzers
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
relevance_analyzer = RelevanceAnalyzer(OPENAI_API_KEY, OPENAI_URL)
grammar_corrector = GrammarCorrector(OPENAI_API_KEY, OPENAI_URL)
speech_scorer = SpeechScorer()

freestyle_analyzer = FreestyleSpeechAnalyzer(
    pronunciation_analyzer=pronunciation_analyzer,
    relevance_analyzer=relevance_analyzer,
    grammar_corrector=grammar_corrector,
    speech_scorer=speech_scorer
)

# FastAPI app
app = FastAPI(
    title="Professional Audio Analysis API",
    description="Advanced pronunciation assessment using TorchAudio forced alignment, AI-powered speech analysis, and multilingual support",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Frontend development server
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:3001",  # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# --- Authentication ---
async def verify_api_key_and_get_id(api_key_from_client: str) -> str:
    """Verify API key against Supabase database and return the API key ID"""
    
    # Temporary bypass for testing
    if api_key_from_client == "test_client_key_123":
        print("✅ Using test API key - bypassing database verification")
        return "test_key_id"
    
    if not supabase_client:
        print("❌ ERROR: Supabase client not available for API key verification.")
        raise HTTPException(status_code=503, detail="API key verification service unavailable.")
    
    if not api_key_from_client:
        raise HTTPException(status_code=401, detail="X-API-Key header missing.")

    hashed_incoming_key = hashlib.sha256(api_key_from_client.encode('utf-8')).hexdigest()

    try:
        query_result = supabase_client.table("api_keys")\
            .select("id, usage_count")\
            .eq("hashed_api_key", hashed_incoming_key)\
            .eq("is_active", True)\
            .single()\
            .execute()
    except Exception as e:
        print(f"❌ Error during API key verification: {e}")
        raise HTTPException(status_code=503, detail="Could not verify API key at this time.")

    if not query_result.data:
        print(f"⚠️ Invalid API key attempt: {hashed_incoming_key[:10]}...")
        raise HTTPException(status_code=403, detail="Invalid or inactive API key.")
    
    api_key_db_id = query_result.data['id']
    
    # Update usage count and last used timestamp
    try:
        current_usage = query_result.data['usage_count']
        supabase_client.table("api_keys").update({
            "usage_count": current_usage + 1, 
            "last_used_at": "now()"
        }).eq("id", api_key_db_id).execute()
        print(f"✅ API key verified: {api_key_db_id} (usage: {current_usage + 1})")
    except Exception as e:
        print(f"⚠️ Warning: Failed to update API key usage count for {api_key_db_id}: {e}")
    
    return api_key_db_id

# --- API Endpoints ---

@app.post("/pronunciation-analysis/assess/us", response_model=PronunciationAssessmentResponse)
async def assess_pronunciation_us(
    request_data: PronunciationAssessmentRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Assess pronunciation quality using Vosk speech recognition and G2P phoneme analysis.
    This endpoint uses unlimited English word support via G2P (Grapheme-to-Phoneme) conversion.
    """
    start_time = time.time()
    
    # Authentication
    await verify_api_key_and_get_id(x_api_key)
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request_data.audio_base64)
        
        # Convert audio to WAV format
        wav_bytes = convert_audio_to_wav(audio_bytes, request_data.audio_format)
        
        # Prepare audio data for analysis
        audio_data, samplerate = prepare_audio_data(wav_bytes)
        
        # Perform pronunciation analysis
        result = pronunciation_analyzer.analyze_pronunciation(
            audio_data=audio_data,
            samplerate=samplerate,
            expected_text=request_data.expected_text
        )
        
        # Log analysis results
        audio_hash = calculate_audio_hash(audio_bytes)
        print(f"📊 Analysis completed - Audio hash: {audio_hash[:10]}... | Score: {result.overall_score}% | Words: {len(result.words)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in pronunciation analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/freestyle-speech/analyze", response_model=FreestyleSpeechResponse)
async def analyze_freestyle_speech(
    request_data: FreestyleSpeechRequest,
    x_api_key: str = Header(..., alias="X-API-Key")
):
    """
    Comprehensive freestyle speech analysis including pronunciation, relevance, grammar, and scoring.
    Uses AI-powered analysis for answer relevance and grammar correction, plus IELTS/CEFR scoring.
    """
    start_time = time.time()
    
    # Authentication
    await verify_api_key_and_get_id(x_api_key)
    
    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request_data.audio_base64)
        
        # Convert audio to WAV format
        wav_bytes = convert_audio_to_wav(audio_bytes, request_data.audio_format)
        
        # Prepare audio data for analysis
        audio_data, samplerate = prepare_audio_data(wav_bytes)
        
        # Perform comprehensive freestyle speech analysis
        result = await freestyle_analyzer.analyze_freestyle_speech(
            audio_data=audio_data,
            samplerate=samplerate,
            question=request_data.question,
            expected_language_level=request_data.expected_language_level,
            scoring_criteria=request_data.scoring_criteria
        )
        
        # Log analysis results
        audio_hash = calculate_audio_hash(audio_bytes)
        print(f"🎯 Freestyle analysis completed - Audio hash: {audio_hash[:10]}... | Confidence: {result.confidence_level}% | Processing: {result.processing_time_ms}ms")
        
        return result
        
    except Exception as e:
        print(f"❌ Error in freestyle speech analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify system status
    """
    status = {
        "status": "healthy",
        "version": "3.0.0",
        "model_version": "torchaudio_mms_forced_alignment_v3.0",
        "features": {
            "forced_alignment": pronunciation_analyzer.model is not None,
            "professional_phoneme_analysis": True,
            "supabase_connection": supabase_client is not None,
            "ai_analysis": OPENAI_API_KEY is not None,
            "multilingual_support": True,  # TorchAudio MMS supports 1000+ languages
            "no_vocabulary_limits": True   # Forced alignment works with any text
        },
        "endpoints": [
            "/pronunciation-analysis/assess/us",
            "/freestyle-speech/analyze",
            "/health"
        ]
    }
    
    # Check if any critical components are missing
    if not pronunciation_analyzer.model:
        status["warnings"] = status.get("warnings", [])
        status["warnings"].append("TorchAudio forced alignment model not loaded - using fallback analysis")
    
    if not supabase_client:
        status["warnings"] = status.get("warnings", [])
        status["warnings"].append("Supabase database connection not available")
    
    if not OPENAI_API_KEY:
        status["warnings"] = status.get("warnings", [])
        status["warnings"].append("OpenAI API key not configured - AI analysis disabled")
    
    return status

@app.get("/test-pronunciation-system")
async def test_pronunciation_system():
    """
    Test endpoint to demonstrate the professional forced alignment system
    """
    test_cases = [
        {
            "text": "hello world",
            "description": "Basic pronunciation test with common words"
        },
        {
            "text": "pronunciation assessment", 
            "description": "Testing complex words with forced alignment"
        },
        {
            "text": "beautiful day outside",
            "description": "Testing multi-word phrases"
        }
    ]
    
    results = []
    for test_case in test_cases:
        # Use a simple mock audio array for testing
        import numpy as np
        mock_audio = np.random.random(16000).astype(np.float32) * 0.1  # 1 second of quiet noise
        
        try:
            result = pronunciation_analyzer.analyze_pronunciation(
                audio_data=mock_audio,
                samplerate=16000,
                expected_text=test_case["text"]
            )
            
            results.append({
                "test_case": test_case,
                "result": {
                    "overall_score": result.overall_score,
                    "word_count": len(result.words),
                    "processing_time_ms": result.processing_time_ms,
                    "words": [
                        {
                            "word": ws.word_text,
                            "score": ws.word_score,
                            "phoneme_count": len(ws.phonemes)
                        } for ws in result.words
                    ]
                }
            })
        except Exception as e:
            results.append({
                "test_case": test_case,
                "error": str(e)
            })
    
    return {
        "message": "Professional forced alignment system test results",
        "system_info": {
            "analyzer_type": "TorchAudio MMS Forced Alignment",
            "model_info": "Pre-trained multilingual model (1000+ languages)",
            "phoneme_system": "Direct forced alignment (no complex word similarity logic)",
            "advantages": [
                "Industry standard approach",
                "Direct phoneme-level confidence scores", 
                "No vocabulary limitations",
                "Multilingual support",
                "No complex alignment algorithms needed"
            ]
        },
        "test_results": results
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Professional Audio Analysis API with Forced Alignment...")
    print("🔬 System uses TorchAudio MMS Forced Alignment (1000+ languages)")
    print("🎯 Features: Professional Pronunciation Analysis + Freestyle Speech Analysis")
    print("🔗 Endpoints: /pronunciation-analysis/assess/us, /freestyle-speech/analyze")
    uvicorn.run(app, host="0.0.0.0", port=8000) 