-- Audio Analysis API Database Schema
-- Execute this in your Supabase SQL editor

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- API Keys table for authentication
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hashed_api_key TEXT UNIQUE NOT NULL,
    name TEXT,
    is_active BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    last_used_at TIMESTAMP WITH TIME ZONE,
    rate_limit_per_hour INTEGER DEFAULT 1000,
    notes TEXT
);

-- Users table (optional, for progress tracking)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Pronunciation analyses table
CREATE TABLE pronunciation_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    api_key_id UUID REFERENCES api_keys(id),
    user_id UUID REFERENCES users(id) NULL,
    expected_text TEXT NOT NULL,
    target_accent TEXT NOT NULL, -- 'us', 'uk', 'general'
    overall_score DECIMAL(5,2),
    words JSONB, -- Store word-level scores and phoneme data
    processing_time_ms INTEGER,
    model_version TEXT,
    audio_format TEXT,
    audio_quality_score DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    metadata JSONB -- Additional analysis metadata
);

-- User progress tracking (aggregated scores over time)
CREATE TABLE user_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    phoneme TEXT, -- IPA symbol
    average_score DECIMAL(5,2),
    total_attempts INTEGER DEFAULT 1,
    last_practiced TIMESTAMP WITH TIME ZONE DEFAULT now(),
    improvement_trend DECIMAL(5,2), -- Positive for improving, negative for declining
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(user_id, phoneme)
);

-- Indexes for performance
CREATE INDEX idx_api_keys_hashed ON api_keys(hashed_api_key);
CREATE INDEX idx_api_keys_active ON api_keys(is_active);
CREATE INDEX idx_pronunciation_analyses_api_key ON pronunciation_analyses(api_key_id);
CREATE INDEX idx_pronunciation_analyses_created_at ON pronunciation_analyses(created_at);
CREATE INDEX idx_pronunciation_analyses_target_accent ON pronunciation_analyses(target_accent);
CREATE INDEX idx_user_progress_user_id ON user_progress(user_id);
CREATE INDEX idx_user_progress_phoneme ON user_progress(phoneme);

-- Row Level Security (RLS) policies
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE pronunciation_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_progress ENABLE ROW LEVEL SECURITY;

-- RLS policies (adjust based on your authentication strategy)
-- For now, allowing service role to access everything
CREATE POLICY "Service role access" ON api_keys
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role access" ON users
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role access" ON pronunciation_analyses
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role access" ON user_progress
    FOR ALL USING (auth.role() = 'service_role');

-- Sample API key for testing (hashed version of 'test_client_key_123')
INSERT INTO api_keys (hashed_api_key, name, notes) VALUES 
('45e03624e260becdf4098251797d50d6ecb532425f4293e0d4133c5d848b40b4', 'Test API Key', 'Hash of test_client_key_123 for local development');

-- Functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for auto-updating timestamps
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_progress_updated_at BEFORE UPDATE ON user_progress
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 