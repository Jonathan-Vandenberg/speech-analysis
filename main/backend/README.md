# Audio Analysis API

A pronunciation assessment API that provides detailed phonetic analysis and scoring for language learners.

## 🎯 Current Status

**Phase 1 MVP: ~80% Complete** ✅

### ✅ Completed Features
- ✅ Supabase database integration with API key authentication
- ✅ POST `/pronunciation-analysis/assess/us` endpoint
- ✅ Base64 audio processing (WAV format)
- ✅ Mock pronunciation scoring with phoneme analysis
- ✅ Problematic sounds identification
- ✅ Database storage of analysis results
- ✅ Request validation and error handling
- ✅ Test client for validation

### 🔄 In Progress
- Real phoneme recognition model integration (currently using mock data)

### ⏳ Next Priorities
1. Replace mock scoring with actual phoneme recognition model
2. Add MP3 audio format support  
3. Implement UK English endpoint
4. Set up CI/CD pipeline
5. Deploy to staging environment

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Supabase account and project
- Virtual environment

### Setup

1. **Clone and navigate to the backend directory:**
   ```bash
   cd audio-analysis/main/backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env file with your Supabase credentials
   ```

5. **Set up database schema:**
   - Go to your Supabase SQL editor
   - Execute the schema from `database_schema.sql`
   - Or use the existing schema if you already have it set up

6. **Add test API key to database:**
   ```bash
   python setup_test_api_key.py
   ```

7. **Start the server:**
   ```bash
   python main.py
   ```

### Testing

Run the test client to verify everything is working:
```bash
python test_client.py
```

Visit the API documentation at: http://localhost:8000/docs

## 📊 Database Schema

The API uses the following Supabase tables:

### `api_keys`
- `id` (UUID, Primary Key)
- `hashed_api_key` (TEXT, Unique)
- `description` (TEXT)
- `is_active` (BOOLEAN)
- `usage_count` (INTEGER)
- `last_used_at` (TIMESTAMP)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### `pronunciation_analyses`
- `id` (UUID, Primary Key)
- `api_key_id` (UUID, Foreign Key)
- `audio_hash` (TEXT)
- `expected_text` (TEXT)
- `target_accent` (VARCHAR)
- `overall_score` (REAL)
- `words` (JSONB)
- `problematic_sounds` (JSONB)
- `processing_time_ms` (INTEGER)
- `model_version` (VARCHAR)
- `warnings` (JSONB)
- `created_at` (TIMESTAMP)

## 🔌 API Usage

### Authentication
All requests require an `X-API-Key` header:
```bash
curl -H "X-API-Key: your_api_key_here" ...
```

### Pronunciation Assessment

**Endpoint:** `POST /pronunciation-analysis/assess/us`

**Request:**
```json
{
  "audio_base64": "UklGRkTnBABXQ...wEA//8AAAAAAAAAA",
  "audio_format": "wav",
  "expected_text": "hello world"
}
```

**Response:**
```json
{
  "overall_score": 85.2,
  "words": [
    {
      "word_text": "hello",
      "word_score": 81.3,
      "phonemes": [
        {
          "ipa_label": "h",
          "phoneme_score": 0.9
        },
        {
          "ipa_label": "ɛ", 
          "phoneme_score": 0.7
        }
      ],
      "start_time": 0.0,
      "end_time": 0.7
    }
  ],
  "processing_time_ms": 245
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "supabase_connected": true,
  "version": "1.0.0"
}
```

## 🛠️ Development

### Project Structure
```
audio-analysis/main/backend/
├── main.py                 # FastAPI application
├── test_client.py         # Test client for validation
├── setup_test_api_key.py  # Database setup script
├── requirements.txt       # Python dependencies
├── env.example           # Environment variables template
├── database_schema.sql   # Supabase schema (for reference)
└── README.md            # This file
```

### Adding New Features

1. **New Endpoints:** Add new FastAPI route handlers in `main.py`
2. **Database Changes:** Update schema and migration scripts
3. **Models:** Add new Pydantic models for request/response validation
4. **Tests:** Update `test_client.py` with new test cases

### Environment Variables

Create a `.env` file based on `env.example`:

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# OpenRouter API (for future features)
OPEN_ROUTER_API_KEY=your_openrouter_key

# Application Configuration
PRONUNCIATION_ANALYSIS_APP_ID=pronunciation_analysis_app_v1
```

## 🔍 Next Steps

### Immediate (Phase 1 completion)
1. **Real Phoneme Recognition:** Replace mock scoring with actual ASR/phoneme model
   - Consider: Vosk, Allosaurus, or Wav2Vec2
   - Implement proper phoneme alignment and scoring
2. **MP3 Support:** Add MP3 audio format processing
3. **Error Handling:** Improve audio quality detection and warnings

### Phase 2 Features
1. **UK English Support:** New endpoint with UK phoneme models
2. **Prosody Analysis:** Add stress, intonation, and rhythm scoring
3. **User Progress Tracking:** Implement user accounts and progress analytics
4. **Advanced Feedback:** More detailed pronunciation improvement suggestions

### Production Readiness
1. **CI/CD Pipeline:** GitHub Actions for automated testing and deployment
2. **Rate Limiting:** Implement proper API rate limiting
3. **Monitoring:** Add logging, metrics, and error tracking
4. **Documentation:** Generate OpenAPI specs and user guides

## 🐛 Troubleshooting

### Common Issues

**"Supabase client not initialized"**
- Check your `.env` file has correct `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`
- Verify the Supabase project is active

**"Invalid or inactive API key"**
- Run `python setup_test_api_key.py` to add the test key
- Check the API key exists in your Supabase `api_keys` table

**Audio processing errors**
- Ensure audio is in WAV format
- Check base64 encoding is correct
- Verify audio file is not corrupted

### Debug Mode
Set `DEBUG=true` in your `.env` file for verbose logging.

## 📝 Contributing

1. Follow the existing code style and patterns
2. Add tests for new features
3. Update documentation
4. Check the CHECKLIST.md for task priorities

## 📞 Support

- Check the FastAPI docs at `/docs` endpoint
- Review the CHECKLIST.md for development roadmap
- Test with `test_client.py` for validation 