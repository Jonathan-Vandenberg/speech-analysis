"""
Database module for Supabase integration and API key management.
"""
import hashlib
import secrets
import os
import json
import logging
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from pydantic import BaseModel

logger = logging.getLogger("speech_analyzer")

class DatabaseConfig(BaseModel):
    supabase_url: str
    supabase_key: str
    
class APIKeyInfo(BaseModel):
    id: str
    description: Optional[str]
    is_active: bool
    usage_count: int
    minute_usage: int
    daily_usage: int
    monthly_usage: int
    minute_limit: int
    daily_limit: int
    monthly_limit: int
    last_used_at: Optional[datetime]
    created_at: datetime
    # Multi-tenant context (optional)
    tenant_id: Optional[str] = None

class UsageLogData(BaseModel):
    api_key_id: str
    endpoint: str
    method: str = "POST"
    deep_analysis: bool = False
    use_audio: bool = False
    tenant_id: Optional[str] = None
    audio_hash: Optional[str] = None
    audio_duration_ms: Optional[int] = None
    text_length: Optional[int] = None
    expected_text: Optional[str] = None
    predicted_text: Optional[str] = None
    overall_score: Optional[float] = None
    processing_time_ms: Optional[int] = None
    model_version: str = "v1.0"
    response_data: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None

class DatabaseManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_ANON_KEY") 
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Supabase credentials not found. Database features will be disabled.")
            self.client = None
        else:
            try:
                self.client: Client = create_client(self.supabase_url, self.supabase_key)
                logger.info("✅ Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
                self.client = None
    
    def is_available(self) -> bool:
        """Check if database is available."""
        return self.client is not None
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def generate_api_key(self) -> str:
        """Generate a new API key."""
        return f"sk-{secrets.token_urlsafe(32)}"
    
    async def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """Validate an API key and return key info if valid."""
        if not self.client:
            logger.warning("Database not available for API key validation")
            return None
        
        try:
            hashed_key = self.hash_api_key(api_key)
            
            # First, reset usage counters if needed
            await self.reset_usage_counters()
            
            result = self.client.table("api_keys").select("*").eq("hashed_api_key", hashed_key).eq("is_active", True).execute()
            
            if not result.data:
                return None
            
            key_data = result.data[0]
            # Lookup tenant mapping if exists
            try:
                tenant_link = self.client.table("api_keys_tenants").select("tenant_id").eq("api_key_id", key_data["id"]).limit(1).execute()
                tenant_id = tenant_link.data[0]["tenant_id"] if tenant_link.data else None
            except Exception:
                tenant_id = None
            return APIKeyInfo(**key_data, tenant_id=tenant_id)
            
        except Exception as e:
            logger.error(f"Error validating API key: {e}")
            return None
    
    async def check_rate_limits(self, api_key_info: APIKeyInfo) -> tuple[bool, str]:
        """Check if API key has exceeded rate limits."""
        if api_key_info.minute_usage >= api_key_info.minute_limit:
            return False, f"Per-minute limit exceeded ({api_key_info.minute_limit} requests/minute)"
        
        if api_key_info.daily_usage >= api_key_info.daily_limit:
            return False, f"Daily limit exceeded ({api_key_info.daily_limit} requests/day)"
        
        if api_key_info.monthly_usage >= api_key_info.monthly_limit:
            return False, f"Monthly limit exceeded ({api_key_info.monthly_limit} requests/month)"
        
        return True, ""
    
    async def log_api_usage(self, usage_data: UsageLogData) -> bool:
        """Log API usage to the database."""
        if not self.client:
            return True  # Silently continue if database not available
        
        try:
            # Insert usage log
            log_data = usage_data.model_dump()
            log_data['created_at'] = datetime.utcnow().isoformat()
            
            self.client.table("api_usage_logs").insert(log_data).execute()
            
            # Update API key usage counters
            self.client.table("api_keys").update({
                "usage_count": self.client.table("api_keys").select("usage_count").eq("id", usage_data.api_key_id).execute().data[0]["usage_count"] + 1,
                "minute_usage": self.client.table("api_keys").select("minute_usage").eq("id", usage_data.api_key_id).execute().data[0]["minute_usage"] + 1,
                "daily_usage": self.client.table("api_keys").select("daily_usage").eq("id", usage_data.api_key_id).execute().data[0]["daily_usage"] + 1,
                "monthly_usage": self.client.table("api_keys").select("monthly_usage").eq("id", usage_data.api_key_id).execute().data[0]["monthly_usage"] + 1,
                "last_used_at": datetime.utcnow().isoformat()
            }).eq("id", usage_data.api_key_id).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging API usage: {e}")
            return False
    
    async def log_pronunciation_analysis(self, api_key_id: str, analysis_data: Dict[str, Any]) -> bool:
        """Log detailed pronunciation analysis to the database."""
        if not self.client:
            return True  # Silently continue if database not available
        
        try:
            # Prepare pronunciation analysis data
            log_data = {
                "api_key_id": api_key_id,
                "audio_hash": analysis_data.get("audio_hash"),
                "expected_text": analysis_data.get("expected_text"),
                "predicted_text": analysis_data.get("predicted_text"),
                "target_accent": analysis_data.get("target_accent", "us"),
                "overall_score": analysis_data.get("overall_score"),
                "words": json.dumps(analysis_data.get("words", [])),
                "problematic_sounds": json.dumps(analysis_data.get("problematic_sounds", [])),
                "processing_time_ms": analysis_data.get("processing_time_ms"),
                "model_version": analysis_data.get("model_version", "v1.0"),
                "warnings": json.dumps(analysis_data.get("warnings", [])),
                "endpoint_used": analysis_data.get("endpoint_used"),
                "deep_analysis_used": analysis_data.get("deep_analysis_used", False),
                "use_audio_used": analysis_data.get("use_audio_used", False),
                "speech_metrics": json.dumps(analysis_data.get("speech_metrics")),
                "grammar_analysis": json.dumps(analysis_data.get("grammar_analysis")),
                "relevance_analysis": json.dumps(analysis_data.get("relevance_analysis")),
                "ielts_score": json.dumps(analysis_data.get("ielts_score")),
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.client.table("pronunciation_analyses").insert(log_data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Error logging pronunciation analysis: {e}")
            return False
    
    async def log_ai_interaction(self, api_key_id: str, interaction_data: Dict[str, Any], tenant_id: Optional[str] = None) -> bool:
        """Log AI model interactions for cost tracking."""
        if not self.client:
            return True  # Silently continue if database not available
        
        try:
            log_data = {
                "api_key_id": api_key_id,
                "interaction_type": interaction_data.get("interaction_type"),
                "model_used": interaction_data.get("model_used"),
                "input_tokens": interaction_data.get("input_tokens"),
                "output_tokens": interaction_data.get("output_tokens"),
                "total_tokens": interaction_data.get("total_tokens"),
                "estimated_cost_usd": interaction_data.get("estimated_cost_usd"),
                "input_text_length": interaction_data.get("input_text_length"),
                "output_text_length": interaction_data.get("output_text_length"),
                "processing_time_ms": interaction_data.get("processing_time_ms"),
                "request_data": json.dumps(interaction_data.get("request_data", {})),
                "response_data": json.dumps(interaction_data.get("response_data", {})),
                "tenant_id": tenant_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            self.client.table("ai_interactions").insert(log_data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Error logging AI interaction: {e}")
            return False
    
    async def reset_usage_counters(self):
        """Reset daily and monthly usage counters as needed."""
        if not self.client:
            return
        
        try:
            # The database function handles this automatically via SQL
            self.client.rpc("reset_usage_counters").execute()
        except Exception as e:
            logger.error(f"Error resetting usage counters: {e}")
    
    async def create_api_key(self, description: str, minute_limit: int = 10, daily_limit: int = 1000, monthly_limit: int = 10000) -> tuple[str, str]:
        """Create a new API key and return (api_key, key_id)."""
        if not self.client:
            raise Exception("Database not available")
        
        try:
            # Generate new API key
            api_key = self.generate_api_key()
            hashed_key = self.hash_api_key(api_key)
            
            # Insert into database
            result = self.client.table("api_keys").insert({
                "hashed_api_key": hashed_key,
                "description": description,
                "minute_limit": minute_limit,
                "daily_limit": daily_limit,
                "monthly_limit": monthly_limit,
                "is_active": True,
                "usage_count": 0,
                "minute_usage": 0,
                "daily_usage": 0,
                "monthly_usage": 0,
                "last_minute_reset": datetime.utcnow().isoformat(),
                "last_daily_reset": date.today().isoformat(),
                "last_monthly_reset": date.today().replace(day=1).isoformat(),
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            if result.data:
                return api_key, result.data[0]["id"]
            else:
                raise Exception("Failed to create API key")
                
        except Exception as e:
            logger.error(f"Error creating API key: {e}")
            raise
    
    async def get_api_keys(self) -> List[Dict[str, Any]]:
        """Get all API keys for admin interface."""
        if not self.client:
            return []
        
        try:
            result = self.client.table("api_keys").select("id, description, is_active, usage_count, minute_usage, daily_usage, monthly_usage, minute_limit, daily_limit, monthly_limit, last_used_at, created_at").order("created_at", desc=True).execute()
            return result.data
        except Exception as e:
            logger.error(f"Error fetching API keys: {e}")
            return []
    
    async def update_api_key(self, key_id: str, updates: Dict[str, Any]) -> bool:
        """Update an API key."""
        if not self.client:
            return False
        
        try:
            self.client.table("api_keys").update(updates).eq("id", key_id).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating API key: {e}")
            return False
    
    async def get_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get usage analytics for the admin dashboard."""
        if not self.client:
            return {}
        try:
            # Get usage data from the view
            result = self.client.table("api_usage_analytics").select("*").execute()
            # Get recent usage logs for trends
            logs_result = self.client.table("api_usage_logs").select(
                "endpoint, deep_analysis, use_audio, created_at"
            ).gte("created_at", f"now() - interval '{days} days'").execute()
            return {
                "api_keys": result.data,
                "recent_logs": logs_result.data,
            }
        except Exception as e:
            logger.error(f"Error fetching usage analytics: {e}")
            return {"api_keys": [], "recent_logs": []}

    async def get_tenant_id_by_subdomain(self, subdomain: Optional[str]) -> Optional[str]:
        if not self.client or not subdomain:
            return None
        try:
            res = self.client.table("tenants").select("id").eq("subdomain", subdomain).limit(1).execute()
            if not res.data:
                return None
            return res.data[0]["id"]
        except Exception as e:
            logger.error(f"Error resolving tenant id for subdomain {subdomain}: {e}")
            return None

    async def get_tenant_config(self, *, subdomain: str | None = None, host: str | None = None) -> Optional[Dict[str, Any]]:
        """Fetch tenant public config by subdomain or host.

        If host is supplied like 'school.speechanalyser.com', the subdomain is
        parsed as the first label.
        """
        if not self.client:
            return None
        try:
            target_sub = subdomain
            if not target_sub and host:
                target_sub = host.split(":")[0].split(".")[0]
            if not target_sub:
                return None

            t = self.client.table("tenants").select(
                "id, subdomain, display_name, status"
            ).eq("subdomain", target_sub).limit(1).execute()
            if not t.data:
                return None

            tenant = t.data[0]
            b = self.client.table("tenant_branding").select(
                "logo_url, primary_hex, secondary_hex, accent_hex, dark_mode"
            ).eq("tenant_id", tenant["id"]).limit(1).execute()
            branding = b.data[0] if b.data else {}
            return {**tenant, "branding": branding}
        except Exception as e:
            logger.error(f"Error fetching tenant config: {e}")
            return None

    async def create_tenant(self, subdomain: str, display_name: str, status: str = "active", branding: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a tenant and optional branding; return tenant_id."""
        if not self.client:
            return None
        try:
            res = self.client.table("tenants").insert({
                "subdomain": subdomain,
                "display_name": display_name,
                "status": status,
            }).execute()
            if not res.data:
                return None
            tenant_id = res.data[0]["id"]
            if branding:
                self.client.table("tenant_branding").upsert({
                    "tenant_id": tenant_id,
                    **{k: v for k, v in branding.items() if v is not None},
                }).execute()
            return tenant_id
        except Exception as e:
            logger.error(f"Error creating tenant: {e}")
            return None

    async def store_tenant_creds(self, tenant_id: str, supabase_url: str, anon_key: str, service_role_key_encrypted: str, region: Optional[str] = None, rotation_at: Optional[str] = None) -> bool:
        if not self.client:
            return False
        try:
            self.client.table("tenant_supabase_creds").upsert({
                "tenant_id": tenant_id,
                "supabase_url": supabase_url,
                "anon_key": anon_key,
                "service_role_key_encrypted": service_role_key_encrypted,
                "region": region,
                "rotation_at": rotation_at,
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error storing tenant creds: {e}")
            return False

    async def link_key_to_tenant(self, key_id: str, tenant_id: Optional[str] = None, subdomain: Optional[str] = None) -> bool:
        if not self.client:
            return False
        try:
            resolved_tenant_id = tenant_id
            if not resolved_tenant_id and subdomain:
                tr = self.client.table("tenants").select("id").eq("subdomain", subdomain).limit(1).execute()
                if not tr.data:
                    return False
                resolved_tenant_id = tr.data[0]["id"]
            if not resolved_tenant_id:
                return False
            self.client.table("api_keys_tenants").upsert({
                "api_key_id": key_id,
                "tenant_id": resolved_tenant_id,
            }).execute()
            return True
        except Exception as e:
            logger.error(f"Error linking key to tenant: {e}")
            return False

    async def list_tenants(self) -> List[Dict[str, Any]]:
        """List tenants with basic branding fields."""
        if not self.client:
            return []
        try:
            tenants = self.client.table("tenants").select("id, subdomain, display_name, status").execute().data
            result: List[Dict[str, Any]] = []
            for t in tenants:
                b = self.client.table("tenant_branding").select("logo_url, primary_hex, secondary_hex, accent_hex, dark_mode").eq("tenant_id", t["id"]).limit(1).execute()
                branding = b.data[0] if b.data else {}
                result.append({**t, "branding": branding})
            return result
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            return []

    async def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        if not self.client:
            return False
        try:
            # Split branding vs core fields
            branding = updates.pop("branding", None)
            if updates:
                self.client.table("tenants").update(updates).eq("id", tenant_id).execute()
            if branding:
                self.client.table("tenant_branding").upsert({
                    "tenant_id": tenant_id,
                    **branding,
                }).execute()
            return True
        except Exception as e:
            logger.error(f"Error updating tenant: {e}")
            return False

# Global database manager instance
db_manager = DatabaseManager()
