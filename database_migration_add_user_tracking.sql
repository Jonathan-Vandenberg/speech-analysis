-- Migration: Add user tracking to api_usage_logs
-- This allows tracking individual users within tenants

-- Add user_id column to api_usage_logs
ALTER TABLE api_usage_logs 
ADD COLUMN IF NOT EXISTS user_id VARCHAR(100);

-- Add index for better performance on user queries
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_user_id ON api_usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_logs_tenant_user ON api_usage_logs(tenant_id, user_id);

-- Create view for tenant user analytics
CREATE OR REPLACE VIEW tenant_user_analytics AS
SELECT 
    aul.tenant_id,
    t.display_name as tenant_name,
    aul.user_id,
    COUNT(*) as total_requests,
    COUNT(DISTINCT aul.endpoint) as unique_endpoints_used,
    SUM(CASE WHEN aul.endpoint = 'transcribe' THEN 1 ELSE 0 END) as transcribe_requests,
    SUM(CASE WHEN aul.endpoint = 'scripted' THEN 1 ELSE 0 END) as scripted_requests,
    SUM(CASE WHEN aul.endpoint = 'pronunciation' THEN 1 ELSE 0 END) as pronunciation_requests,
    SUM(CASE WHEN aul.endpoint = 'unscripted' THEN 1 ELSE 0 END) as unscripted_requests,
    AVG(aul.processing_time_ms) as avg_processing_time,
    SUM(aul.audio_duration_ms) as total_audio_duration_ms,
    COUNT(DISTINCT DATE(aul.created_at)) as active_days,
    MAX(aul.created_at) as last_activity,
    MIN(aul.created_at) as first_activity
FROM api_usage_logs aul
LEFT JOIN api_keys ak ON aul.api_key_id = ak.id
LEFT JOIN api_keys_tenants akt ON ak.id = akt.api_key_id
LEFT JOIN tenants t ON akt.tenant_id = t.id
WHERE ak.is_active = true
    AND aul.tenant_id IS NOT NULL
    AND aul.user_id IS NOT NULL
GROUP BY aul.tenant_id, t.display_name, aul.user_id
ORDER BY total_requests DESC;

-- Create view for tenant summary analytics (with user counts)
CREATE OR REPLACE VIEW tenant_summary_analytics AS
SELECT 
    t.id as tenant_id,
    t.display_name as tenant_name,
    t.subdomain,
    t.status,
    COUNT(DISTINCT aul.user_id) as unique_users,
    COUNT(*) as total_requests,
    SUM(CASE WHEN aul.endpoint = 'transcribe' THEN 1 ELSE 0 END) as transcribe_requests,
    SUM(CASE WHEN aul.endpoint = 'scripted' THEN 1 ELSE 0 END) as scripted_requests,
    SUM(CASE WHEN aul.endpoint = 'pronunciation' THEN 1 ELSE 0 END) as pronunciation_requests,
    SUM(CASE WHEN aul.endpoint = 'unscripted' THEN 1 ELSE 0 END) as unscripted_requests,
    AVG(aul.processing_time_ms) as avg_processing_time,
    SUM(aul.audio_duration_ms) as total_audio_duration_ms,
    COUNT(DISTINCT DATE(aul.created_at)) as active_days,
    MAX(aul.created_at) as last_activity,
    t.created_at as tenant_created_at
FROM tenants t
LEFT JOIN api_keys_tenants akt ON t.id = akt.tenant_id
LEFT JOIN api_keys ak ON akt.api_key_id = ak.id
LEFT JOIN api_usage_logs aul ON ak.id = aul.api_key_id
WHERE t.status = 'active'
GROUP BY t.id, t.display_name, t.subdomain, t.status, t.created_at
ORDER BY total_requests DESC NULLS LAST;
