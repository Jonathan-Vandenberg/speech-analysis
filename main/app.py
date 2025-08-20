# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
load_dotenv()

import os
import logging
from fastapi import FastAPI, HTTPException, Depends, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .routes_scripted import router as scripted_router
from .routes_unscripted import router as unscripted_router
from .database import db_manager
from .middleware import api_key_bearer, APIKeyInfo
from .schemas import (
    APIKeyCreateResponse, APIKeysListResponse, APIKeyUpdateRequest,
    UsageAnalyticsResponse, HealthCheckResponse, ErrorResponse
)


FRONTEND_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()  # Back to INFO now that Panphon is fixed


# Create FastAPI app with enhanced metadata
app = FastAPI(
    title="Speech Analysis API",
    description="""
🎯 **Advanced Speech & Pronunciation Analysis API**

This API provides comprehensive speech analysis capabilities including:

* 🗣️ **Pronunciation Analysis** - Detailed phoneme-level scoring
* 📝 **Scripted Speech Analysis** - Compare speech against expected text
* 🎤 **Unscripted Speech Analysis** - Open-ended speech evaluation with IELTS scoring
* 📊 **Usage Analytics** - Track API usage and performance metrics
* 🔐 **API Key Management** - Secure authentication with rate limiting

## Authentication

All analysis endpoints require a Bearer token API key:
```
Authorization: Bearer sk-your-api-key-here
```

## Rate Limits

API keys have configurable limits:
- **Per-minute limit**: Default 10 requests/minute
- **Daily limit**: Default 1,000 requests/day  
- **Monthly limit**: Default 10,000 requests/month

## Response Format

All analysis endpoints return comprehensive results including:
- Pronunciation scoring (phoneme-level accuracy)
- Speech transcription
- Grammar analysis and corrections
- IELTS band scoring (for unscripted analysis)
- Speech metrics (rate, pauses, filler words)
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Speech Analysis API Support",
        "url": "https://github.com/your-username/audio-analysis",
        "email": "support@yourapi.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.yourapi.com",
            "description": "Production server"
        }
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers with API key authentication
app.include_router(scripted_router, prefix="/analyze", dependencies=[Depends(api_key_bearer)])
app.include_router(unscripted_router, prefix="/analyze", dependencies=[Depends(api_key_bearer)])


logger = logging.getLogger("speech_analyzer")
if not logger.handlers:
    logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))


@app.get(
    "/healthz", 
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check the health status of the API and database connection",
    tags=["Health"],
    responses={
        200: {
            "description": "API is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "ok",
                        "database": "connected",
                        "version": "1.0.0"
                    }
                }
            }
        }
    }
)
async def healthz():
    """Health check endpoint - no authentication required."""
    db_status = "connected" if db_manager.is_available() else "disconnected"
    return {
        "status": "ok",
        "database": db_status,
        "version": "1.0.0"
    }

@app.get(
    "/api/admin/keys",
    response_model=APIKeysListResponse,
    summary="List API Keys",
    description="Retrieve all API keys with usage statistics and limits",
    tags=["Admin - API Keys"],
    responses={
        200: {
            "description": "List of API keys retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "api_keys": [
                            {
                                "id": "123e4567-e89b-12d3-a456-426614174000",
                                "description": "Production API Key",
                                "is_active": True,
                                "usage_count": 150,
                                "minute_usage": 2,
                                "daily_usage": 45,
                                "monthly_usage": 150,
                                "minute_limit": 60,
                                "daily_limit": 1000,
                                "monthly_limit": 10000,
                                "last_used_at": "2024-01-15T10:30:00Z",
                                "created_at": "2024-01-01T09:00:00Z"
                            }
                        ]
                    }
                }
            }
        },
        503: {"description": "Database unavailable", "model": ErrorResponse}
    }
)
async def get_api_keys():
    """Get all API keys for admin interface."""
    if not db_manager.is_available():
        raise HTTPException(status_code=503, detail="Database not available")
    
    keys = await db_manager.get_api_keys()
    return {"api_keys": keys}

@app.post(
    "/api/admin/keys",
    response_model=APIKeyCreateResponse,
    summary="Create API Key",
    description="""
    Create a new API key with custom rate limits.
    
    **⚠️ Important**: The API key is only shown once for security reasons. 
    Make sure to copy and store it securely.
    """,
    tags=["Admin - API Keys"],
    responses={
        200: {
            "description": "API key created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "api_key": "sk-1234567890abcdef1234567890abcdef1234567890abcdef",
                        "key_id": "123e4567-e89b-12d3-a456-426614174000",
                        "description": "Production API Key",
                        "minute_limit": 60,
                        "daily_limit": 1000,
                        "monthly_limit": 10000
                    }
                }
            }
        },
        400: {"description": "Invalid input parameters", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse},
        503: {"description": "Database unavailable", "model": ErrorResponse}
    }
)
async def create_api_key(
    description: str = Form(..., description="Description of the API key purpose"),
    minute_limit: int = Form(10, description="Maximum requests per minute", ge=1, le=1000),
    daily_limit: int = Form(1000, description="Maximum requests per day", ge=1, le=100000),
    monthly_limit: int = Form(10000, description="Maximum requests per month", ge=1, le=1000000)
):
    """Create a new API key with specified rate limits."""
    if not db_manager.is_available():
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        api_key, key_id = await db_manager.create_api_key(description, minute_limit, daily_limit, monthly_limit)
        return {
            "api_key": api_key,
            "key_id": key_id,
            "description": description,
            "minute_limit": minute_limit,
            "daily_limit": daily_limit,
            "monthly_limit": monthly_limit
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put(
    "/api/admin/keys/{key_id}",
    summary="Update API Key",
    description="Update API key settings such as rate limits, description, or active status",
    tags=["Admin - API Keys"],
    responses={
        200: {
            "description": "API key updated successfully",
            "content": {
                "application/json": {
                    "example": {"success": True}
                }
            }
        },
        404: {"description": "API key not found", "model": ErrorResponse},
        500: {"description": "Failed to update API key", "model": ErrorResponse},
        503: {"description": "Database unavailable", "model": ErrorResponse}
    }
)
async def update_api_key(
    key_id: str,
    updates: APIKeyUpdateRequest = Body(
        ...,
        description="Fields to update",
        example={
            "description": "Updated Production Key",
            "is_active": True,
            "minute_limit": 120,
            "daily_limit": 2000
        }
    )
):
    """Update an API key's settings."""
    if not db_manager.is_available():
        raise HTTPException(status_code=503, detail="Database not available")
    
    success = await db_manager.update_api_key(key_id, updates.dict(exclude_unset=True))
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update API key")
    
    return {"success": True}

@app.get(
    "/api/admin/analytics",
    response_model=UsageAnalyticsResponse,
    summary="Usage Analytics",
    description="Get comprehensive usage analytics and statistics for all API keys",
    tags=["Admin - Analytics"],
    responses={
        200: {
            "description": "Analytics data retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "api_keys": [
                            {
                                "id": "123e4567-e89b-12d3-a456-426614174000",
                                "total_requests": 1500,
                                "last_request_at": "2024-01-15T10:30:00Z",
                                "first_request_at": "2024-01-01T09:00:00Z"
                            }
                        ],
                        "recent_logs": [
                            {
                                "endpoint": "/analyze/pronunciation",
                                "deep_analysis": True,
                                "use_audio": True,
                                "created_at": "2024-01-15T10:30:00Z"
                            }
                        ]
                    }
                }
            }
        },
        503: {"description": "Database unavailable", "model": ErrorResponse}
    }
)
async def get_usage_analytics(
    days: int = 30
):
    """Get usage analytics for admin dashboard."""
    if not db_manager.is_available():
        raise HTTPException(status_code=503, detail="Database not available")
    
    analytics = await db_manager.get_usage_analytics(days)
    return analytics

 
