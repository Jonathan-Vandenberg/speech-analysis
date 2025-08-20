#!/usr/bin/env python3
"""
Generate OpenAPI specification file for the Speech Analysis API
"""

import json
import os
from main.app import app

def generate_openapi_spec():
    """Generate and save OpenAPI specification to JSON file"""
    
    # Get the OpenAPI schema
    openapi_schema = app.openapi()
    
    # Add security scheme for API key authentication
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "API Key",
            "description": "API key authentication using Bearer token. Format: `Bearer sk-your-api-key-here`"
        }
    }
    
    # Add security requirement for protected endpoints
    protected_paths = ["/analyze/pronunciation", "/analyze/scripted", "/analyze/unscripted"]
    
    for path, methods in openapi_schema["paths"].items():
        if any(path.startswith(protected) for protected in protected_paths):
            for method_info in methods.values():
                if isinstance(method_info, dict):
                    method_info["security"] = [{"BearerAuth": []}]
    
    # Enhance the main description with more details
    openapi_schema["info"]["description"] = """
# 🎯 Speech Analysis API

A comprehensive speech and pronunciation analysis API designed for language learning applications.

## 🚀 Features

### Core Analysis Capabilities
- **🗣️ Pronunciation Analysis**: Detailed phoneme-level scoring with IPA transcription
- **📝 Scripted Speech Analysis**: Compare recorded speech against expected text
- **🎤 Unscripted Speech Analysis**: Open-ended speech evaluation with IELTS scoring
- **🧠 AI-Powered Feedback**: Grammar correction, fluency analysis, and constructive feedback

### Management & Analytics
- **🔐 API Key Management**: Secure authentication with configurable rate limits
- **📊 Usage Analytics**: Comprehensive tracking of API usage and performance
- **⚡ Rate Limiting**: Per-minute, daily, and monthly usage controls
- **🏥 Health Monitoring**: Real-time status checks and database connectivity

## 🔐 Authentication

All analysis endpoints require API key authentication:

```bash
curl -X POST "https://api.example.com/analyze/pronunciation" \\
  -H "Authorization: Bearer sk-your-api-key-here" \\
  -F "expected_text=Hello world" \\
  -F "file=@audio.wav"
```

## 📈 Rate Limits

API keys have configurable limits:
- **Per-minute**: Default 10 requests/minute (configurable 1-1000)
- **Daily**: Default 1,000 requests/day (configurable 1-100,000)
- **Monthly**: Default 10,000 requests/month (configurable 1-1,000,000)

## 📋 Supported Audio Formats

- **WAV** (recommended)
- **MP3**
- **WEBM**
- **OGG**
- **FLAC**

Audio is automatically converted to 16kHz mono for processing.

## 🎯 Use Cases

### Language Learning Apps
- Pronunciation training with phoneme-level feedback
- Speaking practice with real-time scoring
- IELTS/TOEFL preparation with band scoring

### Educational Platforms
- Automated speech assessment
- Progress tracking and analytics
- Adaptive learning based on performance

### Voice Training Applications
- Professional voice coaching
- Accent reduction training
- Public speaking improvement

## 📞 Support

- **Documentation**: [GitHub Pages](https://your-username.github.io/audio-analysis)
- **Issues**: [GitHub Issues](https://github.com/your-username/audio-analysis/issues)
- **Email**: support@yourapi.com

---

*Built with FastAPI, advanced phonetic analysis, and AI-powered speech processing.*
"""
    
    # Add examples to the main info
    openapi_schema["info"]["x-logo"] = {
        "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png",
        "altText": "Speech Analysis API"
    }
    
    # Save to file
    output_file = "openapi.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2, ensure_ascii=False)
    
    print(f"✅ OpenAPI specification generated: {output_file}")
    print(f"📄 Total endpoints: {len(openapi_schema['paths'])}")
    print(f"🏷️  Tags: {[tag['name'] for tag in openapi_schema.get('tags', [])]}")
    
    return output_file

if __name__ == "__main__":
    generate_openapi_spec()
