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
from .routes_transcribe import router as transcribe_router
from .database import db_manager
from .middleware import api_key_bearer, APIKeyInfo
from .schemas import (
    APIKeyCreateResponse, APIKeysListResponse, APIKeyUpdateRequest,
    UsageAnalyticsResponse, HealthCheckResponse, ErrorResponse,
    TenantConfigResponse, TenantCreateRequest, TenantCreateResponse, TenantCredsRequest, LinkKeyRequest, TenantUpdateRequest, TenantProvisionRequest, TenantMigrationResponse
)
def _decrypt(ciphertext: str) -> str:
    from .crypto import decrypt_string
    return decrypt_string(ciphertext)

import os
import time
import jwt


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
        "url": "https://github.com/Jonathan-Vandenberg/speech-analysis",
        "email": "support@speechanalyser.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    servers=[
        {
            "url": "https://api.speechanalyser.com",
            "description": "Production server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Development server"
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
app.include_router(transcribe_router, prefix="/analyze", dependencies=[Depends(api_key_bearer)])


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
    "/tenants/config",
    response_model=TenantConfigResponse | None,
    summary="Public tenant config",
    description="Resolve tenant config by host or subdomain; returns branding and display info.",
    tags=["Tenants"],
)
async def get_tenant_config(host: str | None = None, subdomain: str | None = None):
    cfg = await db_manager.get_tenant_config(subdomain=subdomain, host=host)
    if not cfg:
        # Return 200 with null to let frontend show fallback
        return None
    return cfg


@app.get(
    "/api/admin/tenant-id",
    summary="Resolve tenant id by subdomain",
    tags=["Admin - Tenants"],
)
async def admin_resolve_tenant_id(subdomain: str):
    tid = await db_manager.get_tenant_id_by_subdomain(subdomain)
    return {"tenant_id": tid}


@app.post(
    "/tenants/branding",
    summary="Update tenant branding",
    tags=["Tenants"],
)
async def update_tenant_branding(request: dict):
    """Update tenant branding configuration."""
    try:
        host = request.get("host")
        branding = request.get("branding", {})
        
        if not host:
            raise HTTPException(status_code=400, detail="Host is required")
        
        # Parse subdomain from host
        subdomain = host.split(":")[0].split(".")[0]
        if not subdomain:
            raise HTTPException(status_code=400, detail="Unable to determine subdomain from host")
        
        # Get tenant ID
        tenant_id = await db_manager.get_tenant_id_by_subdomain(subdomain)
        if not tenant_id:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        # Update tenant branding
        success = await db_manager.update_tenant_branding(tenant_id, branding)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update branding")
        
        return {"success": True, "message": "Branding updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating tenant branding: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SIGNING_SECRET")
    if not secret:
        # Dev fallback; should be set in production
        secret = "dev-secret-change-me"
    return secret


@app.post(
    "/api/admin/tenants/{tenant_id}/token",
    summary="Issue short-lived tenant access token",
    tags=["Admin - Tenants"],
)
async def admin_issue_tenant_token(tenant_id: str, minutes: int = 10):
    now = int(time.time())
    exp = now + max(60, minutes * 60)
    payload = {
        "sub": tenant_id,
        "scope": "tenant",
        "exp": exp,
        "iat": now,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return {"token": token, "expires_at": exp}


@app.post(
    "/api/admin/tenants/{tenant_id}/provision",
    summary="Trigger CI workflow to provision Supabase project",
    tags=["Admin - Tenants"],
)
async def admin_trigger_provision(tenant_id: str, payload: TenantProvisionRequest):
    import requests
    gh_token = os.getenv("GITHUB_TOKEN_FOR_PROVISION") or os.getenv("GH_PROVISION_TOKEN")
    repo = os.getenv("GITHUB_REPO", "owner/repo")
    workflow = os.getenv("GITHUB_PROVISION_WORKFLOW", "provision-tenant.yml")
    if not gh_token or repo == "owner/repo":
        raise HTTPException(status_code=500, detail="Provisioning not configured")
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    headers = {"Authorization": f"Bearer {gh_token}", "Accept": "application/vnd.github+json"}
    payload = {
        "ref": os.getenv("GITHUB_REF", "main"),
        "inputs": {
            "tenant_id": tenant_id,
            "subdomain": payload.subdomain,
            "display_name": payload.display_name,
            "region": payload.region or "us-east-1",
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code not in (200, 201, 204):
        raise HTTPException(status_code=500, detail=f"Failed to trigger workflow: {resp.status_code} {resp.text}")
    return {"success": True}


# Admin endpoints for control plane (protect at gateway or add auth later)
@app.post(
    "/api/admin/tenants",
    response_model=TenantCreateResponse,
    summary="Create tenant",
    tags=["Admin - Tenants"],
)
async def admin_create_tenant(payload: TenantCreateRequest):
    tenant_id = await db_manager.create_tenant(
        subdomain=payload.subdomain,
        display_name=payload.display_name,
        status=payload.status,
        branding=payload.branding.model_dump() if payload.branding else None,
    )
    if not tenant_id:
        raise HTTPException(status_code=500, detail="Failed to create tenant")
    # Optionally issue an API key and link it to the tenant
    api_key_plain = None
    key_id = None
    if payload.issue_api_key:
        try:
            api_key_plain, key_id = await db_manager.create_api_key(
                payload.key_description or f"Key for {payload.subdomain}",
                payload.minute_limit or 10,
                payload.daily_limit or 1000,
                payload.monthly_limit or 10000,
            )
            await db_manager.link_key_to_tenant(key_id, tenant_id=tenant_id)
        except Exception as e:
            # Don't fail tenant creation if key creation fails; just omit key
            key_id = None
            api_key_plain = None
    return {
        "id": tenant_id,
        "subdomain": payload.subdomain,
        "display_name": payload.display_name,
        "status": payload.status,
        "api_key": api_key_plain,
        "key_id": key_id,
    }


@app.post(
    "/api/admin/tenants/{tenant_id}/db",
    summary="Store tenant Supabase credentials (encrypted)",
    tags=["Admin - Tenants"],
)
async def admin_store_tenant_creds(tenant_id: str, creds: TenantCredsRequest):
    from .crypto import encrypt_string
    encrypted = encrypt_string(creds.service_role_key)
    # Optionally encrypt and store db password if provided
    db_pw_enc = encrypt_string(creds.db_password) if getattr(creds, "db_password", None) else None
    ok = await db_manager.store_tenant_creds(
        tenant_id,
        creds.supabase_url,
        creds.anon_key,
        encrypted,
        creds.region,
        creds.rotation_at,
        db_password_encrypted=db_pw_enc,
    )
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to store credentials")
    return {"success": True}


@app.post(
    "/api/admin/tenants/{tenant_id}/migrate",
    response_model=TenantMigrationResponse,
    summary="Run school schema migration on tenant database",
    tags=["Admin - Tenants"],
)
async def admin_migrate_tenant(tenant_id: str):
    """Apply SQL migration to tenant DB using stored credentials.
    For now, uses a bundled SQL file path exposed by SCHOOL_SQL_PATH env.
    """
    import requests
    sql_path = os.getenv("SCHOOL_SQL_PATH")
    if not sql_path or not os.path.exists(sql_path):
        raise HTTPException(status_code=500, detail="SCHOOL_SQL_PATH not configured on server")
    # Load tenant creds
    if not db_manager.client:
        raise HTTPException(status_code=503, detail="Control plane DB unavailable")
    try:
        row = db_manager.client.table("tenant_supabase_creds").select("supabase_url, anon_key, service_role_key_encrypted").eq("tenant_id", tenant_id).limit(1).execute()
        if not row.data:
            raise HTTPException(status_code=404, detail="Tenant credentials not found")
        supabase_url = row.data[0]["supabase_url"].rstrip("/")
        service_role = _decrypt(row.data[0]["service_role_key_encrypted"])
        applied = 0
        with open(sql_path, "r", encoding="utf-8") as f:
            sql = f.read()
        # Prefer Supabase Postgres HTTP (pg-meta) endpoint to avoid direct DB connectivity issues (IPv6)
        headers = {
            "Authorization": f"Bearer {service_role}",
            "apikey": service_role,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Try official Postgres HTTP path first, then fallback to pg-meta legacy path(s)
        urls_to_try = [
            f"{supabase_url}/postgres/v1/query",
            f"{supabase_url}/pg-meta/query",
        ]
        last_error = None
        resp = None
        for url in urls_to_try:
            try:
                # Some deployments expect 'q', others expect 'query'
                resp = requests.post(url, headers=headers, json={"q": sql, "params": []}, timeout=180)
                if resp.status_code == 200:
                    break
                # Retry same URL with alternate payload shape
                resp = requests.post(url, headers=headers, json={"query": sql}, timeout=180)
                if resp.status_code == 200:
                    break
                last_error = f"{resp.status_code} {resp.text[:500]}"
            except Exception as e:
                last_error = str(e)
        if not resp or resp.status_code != 200:
            # Fallback: connect via Supavisor (IPv4 pooler) using DB password
            region = None
            try:
                # Try to fetch region if stored alongside creds
                cr = db_manager.client.table("tenant_supabase_creds").select("region").eq("tenant_id", tenant_id).limit(1).execute()
                if cr.data:
                    region = cr.data[0].get("region")
            except Exception:
                region = None
            region = region or "us-east-1"
            project_ref = supabase_url.split("//")[-1].split(".")[0]
            pooler_host = f"aws-0-{region}.pooler.supabase.com"
            db_password = os.getenv("DEFAULT_TENANT_DB_PASSWORD")
            if not db_password:
                raise HTTPException(status_code=500, detail=f"pg-meta sql failed: {last_error}; fallback requires DEFAULT_TENANT_DB_PASSWORD env")
            import psycopg
            last_exc: Exception | None = None
            # Try variant 1: username includes project ref
            try:
                conn = psycopg.connect(
                    host=pooler_host,
                    dbname="postgres",
                    user=f"postgres.{project_ref}",
                    password=db_password,
                    port=6543,
                    sslmode="require",
                    connect_timeout=30,
                )
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(sql)
                    conn.commit()
                return TenantMigrationResponse(success=True, applied_statements=1)
            except Exception as e1:
                last_exc = e1
            # Try variant 2: options=project=<ref> with plain username
            try:
                conn = psycopg.connect(
                    host=pooler_host,
                    dbname="postgres",
                    user="postgres",
                    password=db_password,
                    options=f"project={project_ref}",
                    port=6543,
                    sslmode="require",
                    connect_timeout=30,
                )
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(sql)
                    conn.commit()
                return TenantMigrationResponse(success=True, applied_statements=1)
            except Exception as e2:
                last_exc = e2
            raise HTTPException(status_code=500, detail=f"pg-meta sql failed and supavisor fallback failed: {last_exc}")
        try:
            data = resp.json()
        except Exception:
            data = None
        # If pg-meta returns an error field
        if isinstance(data, dict) and data.get("error"):
            raise HTTPException(status_code=500, detail=f"pg-meta error: {data['error']}")
        applied = 1
        return TenantMigrationResponse(success=True, applied_statements=applied)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/api/admin/keys/link",
    summary="Link API key to tenant",
    tags=["Admin - API Keys"],
)
async def admin_link_key(payload: LinkKeyRequest):
    ok = await db_manager.link_key_to_tenant(payload.key_id, tenant_id=payload.tenant_id, subdomain=payload.subdomain)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to link key to tenant")
    return {"success": True}


@app.get(
    "/api/admin/tenants",
    summary="List tenants",
    tags=["Admin - Tenants"],
)
async def admin_list_tenants():
    tenants = await db_manager.list_tenants()
    return {"tenants": tenants}


@app.put(
    "/api/admin/tenants/{tenant_id}",
    summary="Update tenant (display/status/branding)",
    tags=["Admin - Tenants"],
)
async def admin_update_tenant(tenant_id: str, payload: TenantUpdateRequest):
    ok = await db_manager.update_tenant(tenant_id, payload.model_dump(exclude_unset=True))
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to update tenant")
    return {"success": True}

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
    """Get usage analytics for admin dashboard.
    Returns empty analytics object if DB is unavailable or errors occur.
    """
    try:
        analytics = await db_manager.get_usage_analytics(days)
        # Ensure required keys exist
        return analytics or {"api_keys": [], "recent_logs": []}
    except Exception:
        return {"api_keys": [], "recent_logs": []}

 
