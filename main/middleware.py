"""
API Key authentication middleware for the speech analysis API.
"""
import time
import hashlib
import logging
from typing import Optional
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .database import db_manager, APIKeyInfo, UsageLogData
import os
import jwt

logger = logging.getLogger("speech_analyzer")

class APIKeyBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)

    async def __call__(self, request: Request) -> Optional[APIKeyInfo]:
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if credentials:
            if not credentials.scheme == "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme."
                )
            api_key_info = await self.verify_api_key(credentials.credentials)
            if not api_key_info:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or inactive API key."
                )
            
            # Check rate limits
            can_proceed, error_msg = await db_manager.check_rate_limits(api_key_info)
            if not can_proceed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=error_msg
                )
            
            return api_key_info
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid authorization code."
            )

    async def verify_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """Verify API key against database."""
        if not db_manager.is_available():
            # If database is not available, allow requests (for development)
            logger.warning("Database not available - allowing request without API key validation")
            return None
        
        try:
            # Accept either Bearer sk-... API keys or JWT tenant tokens
            if api_key.count('.') == 2:
                try:
                    payload = jwt.decode(api_key, os.getenv('JWT_SIGNING_SECRET', 'dev-secret-change-me'), algorithms=["HS256"])
                    tenant_id = payload.get('sub')
                    # Return a minimal APIKeyInfo-like object for downstream logging
                    return APIKeyInfo(
                        id="jwt-tenant",
                        description="JWT tenant token",
                        is_active=True,
                        usage_count=0,
                        minute_usage=0,
                        daily_usage=0,
                        monthly_usage=0,
                        minute_limit=10**9,
                        daily_limit=10**9,
                        monthly_limit=10**9,
                        last_used_at=None,
                        created_at=None,  # type: ignore
                        tenant_id=tenant_id,
                    )
                except Exception as e:
                    logger.error(f"Invalid JWT token: {e}")
                    # fall through to API key validation
            api_key_info = await db_manager.validate_api_key(api_key)
            return api_key_info
        except Exception as e:
            logger.error(f"Error verifying API key: {e}")
            return None

# Global API key validator
api_key_bearer = APIKeyBearer()

class RequestTracker:
    """Tracks requests for usage logging."""
    
    def __init__(self):
        self.request_data = {}
    
    def start_request(self, request_id: str, api_key_info: Optional[APIKeyInfo], endpoint: str, request: Request):
        """Start tracking a request."""
        self.request_data[request_id] = {
            "api_key_info": api_key_info,
            "endpoint": endpoint,
            "start_time": time.time(),
            "request": request,
            "ip_address": self.get_client_ip(request),
            "user_agent": request.headers.get("user-agent", "")
        }
    
    def get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded IP first (in case of proxy/load balancer)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fallback to request client host
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    async def finish_request(self, 
                           request_id: str, 
                           response_data: dict, 
                           form_data: dict = None,
                           audio_data: bytes = None):
        """Finish tracking a request and log usage."""
        if request_id not in self.request_data:
            return
        
        request_info = self.request_data[request_id]
        processing_time_ms = int((time.time() - request_info["start_time"]) * 1000)
        
        # Extract parameters from form data
        deep_analysis = form_data.get("deep_analysis", "false").lower() == "true" if form_data else False
        use_audio = form_data.get("use_audio", "false").lower() == "true" if form_data else False
        expected_text = form_data.get("expected_text", "") if form_data else ""
        
        # Calculate audio hash if audio provided
        audio_hash = None
        audio_duration_ms = None
        if audio_data:
            audio_hash = hashlib.md5(audio_data).hexdigest()
            # Rough estimate - would need proper audio analysis for exact duration
            audio_duration_ms = len(audio_data) // 32  # Rough estimate
        
        # Extract response information
        overall_score = None
        predicted_text = ""
        if "pronunciation" in response_data:
            overall_score = response_data["pronunciation"].get("overall_score")
        if "predicted_text" in response_data:
            predicted_text = response_data["predicted_text"]
        
        # Create usage log entry
        if request_info["api_key_info"]:
            usage_data = UsageLogData(
                api_key_id=request_info["api_key_info"].id,
                endpoint=request_info["endpoint"],
                deep_analysis=deep_analysis,
                use_audio=use_audio,
                tenant_id=getattr(request_info["api_key_info"], "tenant_id", None),
                audio_hash=audio_hash,
                audio_duration_ms=audio_duration_ms,
                text_length=len(expected_text) if expected_text else None,
                expected_text=expected_text[:1000] if expected_text else None,  # Truncate for storage
                predicted_text=predicted_text[:1000] if predicted_text else None,  # Truncate for storage
                overall_score=overall_score,
                processing_time_ms=processing_time_ms,
                model_version="v1.0",
                response_data=response_data,
                ip_address=request_info["ip_address"],
                user_agent=request_info["user_agent"]
            )
            
            await db_manager.log_api_usage(usage_data)
            
            # Also log detailed pronunciation analysis if this was an analysis endpoint
            if request_info["endpoint"] in ["pronunciation", "scripted", "unscripted"]:
                analysis_data = {
                    "audio_hash": audio_hash,
                    "expected_text": expected_text,
                    "predicted_text": predicted_text,
                    "overall_score": overall_score,
                    "words": response_data.get("pronunciation", {}).get("words", []),
                    "processing_time_ms": processing_time_ms,
                    "endpoint_used": request_info["endpoint"],
                    "deep_analysis_used": deep_analysis,
                    "use_audio_used": use_audio,
                    "speech_metrics": response_data.get("metrics"),
                    "grammar_analysis": response_data.get("grammar"),
                    "relevance_analysis": response_data.get("relevance"),
                    "ielts_score": response_data.get("ielts_score")
                }
                
                await db_manager.log_pronunciation_analysis(request_info["api_key_info"].id, analysis_data)
        
        # Clean up
        del self.request_data[request_id]

# Global request tracker
request_tracker = RequestTracker()

def generate_request_id() -> str:
    """Generate a unique request ID."""
    import uuid
    return str(uuid.uuid4())

async def track_ai_interaction(api_key_info: Optional[APIKeyInfo], 
                             interaction_type: str,
                             model_used: str,
                             input_text: str,
                             output_text: str,
                             processing_time_ms: int,
                             tokens_data: dict = None):
    """Track AI model interactions for cost analysis."""
    if not api_key_info or not db_manager.is_available():
        return
    
    # Estimate costs based on model and tokens
    estimated_cost = 0.0
    if tokens_data and "total_tokens" in tokens_data:
        # Rough cost estimates (update with actual pricing)
        if "gpt-4" in model_used.lower():
            estimated_cost = tokens_data["total_tokens"] * 0.00003  # $0.03 per 1K tokens
        elif "gpt-3.5" in model_used.lower():
            estimated_cost = tokens_data["total_tokens"] * 0.000002  # $0.002 per 1K tokens
    
    interaction_data = {
        "interaction_type": interaction_type,
        "model_used": model_used,
        "input_tokens": tokens_data.get("input_tokens") if tokens_data else None,
        "output_tokens": tokens_data.get("output_tokens") if tokens_data else None,
        "total_tokens": tokens_data.get("total_tokens") if tokens_data else None,
        "estimated_cost_usd": estimated_cost,
        "input_text_length": len(input_text),
        "output_text_length": len(output_text),
        "processing_time_ms": processing_time_ms,
        "request_data": {"input_preview": input_text[:500]},  # Store preview only
        "response_data": {"output_preview": output_text[:500]}  # Store preview only
    }
    
    await db_manager.log_ai_interaction(api_key_info.id, interaction_data, getattr(api_key_info, 'tenant_id', None))
