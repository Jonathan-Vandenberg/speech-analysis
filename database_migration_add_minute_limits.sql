-- Migration script to add minute-based rate limiting to existing api_keys table
-- Run this if you already have an api_keys table without minute limit fields

-- Add minute limit columns to existing api_keys table
ALTER TABLE api_keys 
ADD COLUMN IF NOT EXISTS minute_limit INTEGER DEFAULT 10,
ADD COLUMN IF NOT EXISTS minute_usage INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS last_minute_reset TIMESTAMPTZ DEFAULT DATE_TRUNC('minute', NOW());

-- Update the reset function to handle minute resets
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

-- Recreate the analytics view to include minute limit data
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
