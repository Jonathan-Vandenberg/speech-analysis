# Speech Analysis API Setup

This document describes how to set up the speech analysis API with API key management and usage tracking.

## Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Supabase Configuration (Required for API key management and usage tracking)
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# OpenAI Configuration (Required for grammar analysis and IELTS scoring)
OPENAI_API_KEY=your_openai_api_key

# CORS Configuration (Include admin frontend URL)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Logging Configuration
LOG_LEVEL=INFO

# Whisper Model Configuration (Optional)
WHISPER_MODEL=small
WHISPER_COMPUTE_TYPE=auto
WHISPER_BEAM_SIZE=1
```

## Database Setup

1. **Create Supabase Project**:
   - Go to [supabase.com](https://supabase.com) and create a new project
   - Get your project URL and anon key from the project settings

2. **Run Database Schema**:
   - Execute the SQL in `database_schema.sql` in your Supabase SQL editor
   - This creates all necessary tables, indexes, and functions

3. **Verify Tables**:
   The schema creates these tables:
   - `api_keys` - API key management
   - `api_usage_logs` - Request tracking
   - `pronunciation_analyses` - Detailed analysis data
   - `ai_interactions` - AI model usage tracking

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the API**:
```bash
uvicorn main.app:app --host 0.0.0.0 --port 8000 --reload
```

## API Key Authentication

All analysis endpoints now require API key authentication:

### Request Format
```bash
curl -X POST "http://localhost:8000/analyze/pronunciation" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "expected_text=Hello world"
```

### API Key Management
Use the admin interface at `http://localhost:3001` to:
- Create new API keys
- Set rate limits (daily/monthly)
- Monitor usage
- Activate/deactivate keys

## Endpoints

### Analysis Endpoints (Require API Key)
- `POST /analyze/pronunciation` - Audio pronunciation analysis
- `POST /analyze/scripted` - Text-based pronunciation analysis
- `POST /analyze/unscripted` - Speech analysis with optional AI features

### Admin Endpoints (No Auth Currently)
- `GET /healthz` - Health check
- `GET /api/admin/keys` - List API keys
- `POST /api/admin/keys` - Create API key
- `PUT /api/admin/keys/{id}` - Update API key
- `GET /api/admin/analytics` - Usage analytics

## Usage Tracking

The system automatically tracks:
- **Request counts** by endpoint and API key
- **Parameter usage** (deep_analysis, use_audio)
- **Processing times** and performance metrics
- **AI interactions** (OpenAI usage and costs)
- **Rate limiting** (daily/monthly limits)

## Features

### Endpoints
- **Pronunciation Analysis**: Audio-to-phoneme analysis
- **Scripted Analysis**: Text-based pronunciation scoring
- **Unscripted Analysis**: Speech analysis with grammar and IELTS scoring

### Advanced Features (when `deep_analysis=true`)
- Speech fluency metrics (rate, pauses, fillers)
- Grammar analysis and corrections
- IELTS band scoring
- Relevance analysis

### Audio Processing (when `use_audio=true`)
- Whisper transcription
- Phoneme extraction
- Audio quality analysis

## Rate Limiting

Each API key has:
- **Daily limit**: Default 1,000 requests/day
- **Monthly limit**: Default 10,000 requests/month
- **Automatic reset**: Counters reset daily/monthly
- **Enforcement**: Requests blocked when limits exceeded

## Development

For development without database:
- API will work but skip usage tracking
- Health check will show "database: disconnected"
- Admin endpoints will return 503 errors

## Production Considerations

1. **Security**: Add authentication to admin endpoints
2. **Rate Limiting**: Configure appropriate limits for your use case
3. **Monitoring**: Use the analytics dashboard to monitor usage
4. **Scaling**: Consider using connection pooling for high traffic
5. **Backup**: Regular database backups for usage data
