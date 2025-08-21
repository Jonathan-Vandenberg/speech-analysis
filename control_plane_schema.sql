-- Control Plane schema for multi-tenant support
-- This lives in the audio-analysis Supabase project and manages tenants
-- Each school has its own Supabase project; this control plane stores
-- metadata, branding, and encrypted credentials. API keys are linked to
-- tenants via a join table.

-- Required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Tenants (schools) registry
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    subdomain TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', -- active | disabled | pending
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Per-tenant branding config
CREATE TABLE IF NOT EXISTS tenant_branding (
    tenant_id UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    logo_url TEXT,
    primary_hex TEXT,
    secondary_hex TEXT,
    accent_hex TEXT,
    dark_mode BOOLEAN DEFAULT false
);

-- Per-tenant Supabase credentials
-- NOTE: service_role_key must be stored encrypted at rest.
CREATE TABLE IF NOT EXISTS tenant_supabase_creds (
    tenant_id UUID PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    supabase_url TEXT NOT NULL,
    anon_key TEXT NOT NULL,
    service_role_key_encrypted TEXT NOT NULL,
    region TEXT,
    rotation_at TIMESTAMPTZ
);

-- Link API keys to tenants (many keys can map to one tenant)
CREATE TABLE IF NOT EXISTS api_keys_tenants (
    api_key_id UUID REFERENCES api_keys(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    PRIMARY KEY (api_key_id, tenant_id)
);

-- Ensure api_usage_logs has tenant_id, and index it
ALTER TABLE IF EXISTS api_usage_logs
    ADD COLUMN IF NOT EXISTS tenant_id UUID;

CREATE INDEX IF NOT EXISTS idx_usage_logs_tenant_id
ON api_usage_logs(tenant_id);

-- Ensure ai_interactions carries tenant context for cost analytics
ALTER TABLE IF EXISTS ai_interactions
    ADD COLUMN IF NOT EXISTS tenant_id UUID;
CREATE INDEX IF NOT EXISTS idx_ai_interactions_tenant ON ai_interactions(tenant_id);

-- Replace analytics view to include tenant context when available
-- Drop first to allow column additions/reordering safely
DROP VIEW IF EXISTS api_usage_analytics;
CREATE VIEW api_usage_analytics AS
SELECT 
    ak.id as api_key_id,
    ak.description as api_key_description,
    akt.tenant_id as tenant_id,
    t.subdomain as tenant_subdomain,
    t.display_name as tenant_display_name,
    COUNT(aul.id) as total_requests,
    COUNT(DISTINCT DATE(aul.created_at)) as active_days,
    AVG(aul.processing_time_ms) as avg_processing_time,
    SUM(CASE WHEN aul.endpoint = 'pronunciation' THEN 1 ELSE 0 END) as pronunciation_requests,
    SUM(CASE WHEN aul.endpoint = 'scripted' THEN 1 ELSE 0 END) as scripted_requests,
    SUM(CASE WHEN aul.endpoint = 'unscripted' THEN 1 ELSE 0 END) as unscripted_requests,
    SUM(CASE WHEN aul.deep_analysis THEN 1 ELSE 0 END) as deep_analysis_requests,
    SUM(CASE WHEN aul.use_audio THEN 1 ELSE 0 END) as audio_requests,
    MAX(aul.created_at) as last_request_at,
    MIN(aul.created_at) as first_request_at
FROM api_keys ak
LEFT JOIN api_keys_tenants akt ON ak.id = akt.api_key_id
LEFT JOIN tenants t ON akt.tenant_id = t.id
LEFT JOIN api_usage_logs aul ON ak.id = aul.api_key_id
WHERE ak.is_active = true
GROUP BY ak.id, ak.description, akt.tenant_id, t.subdomain, t.display_name;

-- Seed a demo tenant for local/testing use and link the default admin key if present
WITH upsert_tenant AS (
    INSERT INTO tenants (subdomain, display_name, status)
    VALUES ('demo', 'Demo School', 'active')
    ON CONFLICT (subdomain) DO UPDATE SET display_name = EXCLUDED.display_name
    RETURNING id
), branding AS (
    INSERT INTO tenant_branding (tenant_id, logo_url, primary_hex, secondary_hex, accent_hex, dark_mode)
    SELECT id, NULL, '#4f46e5', '#0ea5e9', '#22c55e', false FROM upsert_tenant
    ON CONFLICT (tenant_id) DO NOTHING
)
INSERT INTO api_keys_tenants (api_key_id, tenant_id)
SELECT ak.id, t.id
FROM api_keys ak
JOIN upsert_tenant t ON TRUE
WHERE ak.description = 'Admin Development Key - CHANGE THIS!'
ON CONFLICT DO NOTHING;


