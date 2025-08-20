-- Enhanced Supabase schema for API key management and usage tracking
-- This builds upon the existing api_keys and pronunciation_analyses tables

-- Create API keys table (enhanced version)
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    hashed_api_key TEXT NOT NULL UNIQUE,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    -- Rate limiting fields
    minute_limit INTEGER DEFAULT 10,
    daily_limit INTEGER DEFAULT 1000,
    monthly_limit INTEGER DEFAULT 10000,
    minute_usage INTEGER DEFAULT 0,
    daily_usage INTEGER DEFAULT 0,
    monthly_usage INTEGER DEFAULT 0,
    last_minute_reset TIMESTAMPTZ DEFAULT DATE_TRUNC('minute', NOW()),
    last_daily_reset DATE DEFAULT CURRENT_DATE,
    last_monthly_reset DATE DEFAULT DATE_TRUNC('month', CURRENT_DATE)
);

-- Create comprehensive usage tracking table for all endpoints
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE CASCADE,
    endpoint VARCHAR(50) NOT NULL, -- 'pronunciation', 'scripted', 'unscripted'
    method VARCHAR(10) NOT NULL DEFAULT 'POST',
    
    -- Request parameters
    deep_analysis BOOLEAN DEFAULT false,
    use_audio BOOLEAN DEFAULT false,
    
    -- Audio/text metadata
    audio_hash TEXT, -- Hash of audio file for deduplication
    audio_duration_ms INTEGER,
    text_length INTEGER,
    expected_text TEXT,
    predicted_text TEXT,
    
    -- Results metadata
    overall_score REAL,
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    
    -- Response data (JSONB for flexible storage)
    response_data JSONB,
    
    -- Request metadata
    ip_address INET,
    user_agent TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enhanced pronunciation analyses table (keeping existing structure, adding new fields)
CREATE TABLE IF NOT EXISTS pronunciation_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE CASCADE,
    
    -- Audio and text data
    audio_hash TEXT,
    expected_text TEXT,
    predicted_text TEXT,
    target_accent VARCHAR(10) DEFAULT 'us',
    
    -- Scores and results
    overall_score REAL,
    words JSONB, -- WordPronunciation array
    problematic_sounds JSONB,
    
    -- Analysis metadata
    processing_time_ms INTEGER,
    model_version VARCHAR(50) DEFAULT 'v1.0',
    warnings JSONB,
    
    -- Enhanced fields for new functionality
    endpoint_used VARCHAR(50), -- 'pronunciation', 'scripted', 'unscripted'
    deep_analysis_used BOOLEAN DEFAULT false,
    use_audio_used BOOLEAN DEFAULT false,
    
    -- Speech metrics (when deep_analysis is used)
    speech_metrics JSONB,
    grammar_analysis JSONB,
    relevance_analysis JSONB,
    ielts_score JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create AI interactions tracking table
CREATE TABLE IF NOT EXISTS ai_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID REFERENCES api_keys(id) ON DELETE CASCADE,
    
    -- Interaction type
    interaction_type VARCHAR(50) NOT NULL, -- 'grammar_analysis', 'ielts_scoring', 'transcription'
    model_used VARCHAR(100), -- 'gpt-4o', 'whisper-small', etc.
    
    -- Token usage (for OpenAI)
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    
    -- Cost tracking
    estimated_cost_usd DECIMAL(10, 6),
    
    -- Request/response metadata
    input_text_length INTEGER,
    output_text_length INTEGER,
    processing_time_ms INTEGER,
    
    -- Full request/response data
    request_data JSONB,
    response_data JSONB,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_api_keys_hashed_key ON api_keys(hashed_api_key);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active);
CREATE INDEX IF NOT EXISTS idx_usage_logs_api_key ON api_usage_logs(api_key_id);
CREATE INDEX IF NOT EXISTS idx_usage_logs_endpoint ON api_usage_logs(endpoint);
CREATE INDEX IF NOT EXISTS idx_usage_logs_created_at ON api_usage_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_pronunciation_api_key ON pronunciation_analyses(api_key_id);
CREATE INDEX IF NOT EXISTS idx_pronunciation_created_at ON pronunciation_analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_api_key ON ai_interactions(api_key_id);
CREATE INDEX IF NOT EXISTS idx_ai_interactions_type ON ai_interactions(interaction_type);

-- Create a function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for api_keys updated_at
DROP TRIGGER IF EXISTS update_api_keys_updated_at ON api_keys;
CREATE TRIGGER update_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a function to reset minute/daily/monthly usage counters
CREATE OR REPLACE FUNCTION reset_usage_counters()
RETURNS void AS $$
BEGIN
    -- Reset minute counters
    UPDATE api_keys 
    SET minute_usage = 0, last_minute_reset = DATE_TRUNC('minute', NOW())
    WHERE last_minute_reset < DATE_TRUNC('minute', NOW());
    
    -- Reset daily counters
    UPDATE api_keys 
    SET daily_usage = 0, last_daily_reset = CURRENT_DATE
    WHERE last_daily_reset < CURRENT_DATE;
    
    -- Reset monthly counters
    UPDATE api_keys 
    SET monthly_usage = 0, last_monthly_reset = DATE_TRUNC('month', CURRENT_DATE)
    WHERE last_monthly_reset < DATE_TRUNC('month', CURRENT_DATE);
END;
$$ LANGUAGE plpgsql;

-- Create views for easy analytics
CREATE OR REPLACE VIEW api_usage_analytics AS
SELECT 
    ak.id as api_key_id,
    ak.description as api_key_description,
    COUNT(*) as total_requests,
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
LEFT JOIN api_usage_logs aul ON ak.id = aul.api_key_id
WHERE ak.is_active = true
GROUP BY ak.id, ak.description;

-- Create view for AI costs analytics
CREATE OR REPLACE VIEW ai_costs_analytics AS
SELECT 
    ak.id as api_key_id,
    ak.description as api_key_description,
    ai.interaction_type,
    ai.model_used,
    COUNT(*) as interaction_count,
    SUM(ai.total_tokens) as total_tokens,
    SUM(ai.estimated_cost_usd) as total_cost_usd,
    AVG(ai.processing_time_ms) as avg_processing_time,
    DATE_TRUNC('day', ai.created_at) as date
FROM api_keys ak
JOIN ai_interactions ai ON ak.id = ai.api_key_id
WHERE ak.is_active = true
GROUP BY ak.id, ak.description, ai.interaction_type, ai.model_used, DATE_TRUNC('day', ai.created_at)
ORDER BY date DESC;

-- Insert a default admin API key (you should change this!)
-- Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
INSERT INTO api_keys (hashed_api_key, description, minute_limit, daily_limit, monthly_limit)
VALUES (
    'sk-admin-' || encode(digest('admin-development-key-change-me', 'sha256'), 'hex'),
    'Admin Development Key - CHANGE THIS!',
    100,
    10000,
    100000
) ON CONFLICT (hashed_api_key) DO NOTHING;

-- Row Level Security (RLS) - Optional but recommended
-- ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE api_usage_logs ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE pronunciation_analyses ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE ai_interactions ENABLE ROW LEVEL SECURITY;

-- Grant permissions for service role (adjust as needed)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
