#!/usr/bin/env python3
"""
Test client for the Audio Analysis API
Generates test audio and sends requests to verify the API is working correctly.
"""

import base64
import json
import requests
import numpy as np
import soundfile as sf
import io
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "test_client_key_123"  # This matches the hash in the backend

def generate_test_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """Generate a simple test audio file (sine wave)"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    # Generate a simple tone (440 Hz - A note)
    frequency = 440
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Convert to bytes using soundfile
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer.read()

def test_pronunciation_assessment():
    """Test the main pronunciation assessment endpoint"""
    print("🧪 Testing pronunciation assessment endpoint...")
    
    # Generate test audio
    audio_bytes = generate_test_audio()
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Prepare request
    request_data = {
        "audio_base64": audio_base64,
        "audio_format": "wav",
        "expected_text": "hello world testing"
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        # Send request
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/pronunciation-analysis/assess/us",
            json=request_data,
            headers=headers,
            timeout=30
        )
        end_time = time.time()
        
        # Check response
        print(f"⏱️  Request took {(end_time - start_time)*1000:.0f}ms")
        print(f"📡 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"📊 Overall Score: {result.get('overall_score', 'N/A')}")
            print(f"🔤 Number of words analyzed: {len(result.get('words', []))}")
            print(f"⚡ Processing time: {result.get('processing_time_ms', 'N/A')}ms")
            
            # Print word details
            for i, word in enumerate(result.get('words', [])[:3]):  # Show first 3 words
                print(f"   Word {i+1}: '{word.get('word_text')}' - Score: {word.get('word_score')}")
                for phoneme in word.get('phonemes', [])[:2]:  # Show first 2 phonemes
                    print(f"      Phoneme: {phoneme.get('ipa_label')} - Score: {phoneme.get('phoneme_score')}")
            
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"🚨 Request error: {e}")
        return False
    except Exception as e:
        print(f"🚨 Unexpected error: {e}")
        return False

def test_invalid_api_key():
    """Test API key validation"""
    print("\n🔐 Testing API key validation...")
    
    audio_bytes = generate_test_audio(1.0)
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    request_data = {
        "audio_base64": audio_base64,
        "audio_format": "wav",
        "expected_text": "test"
    }
    
    headers = {
        "X-API-Key": "invalid_key_123",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/pronunciation-analysis/assess/us",
            json=request_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 403:
            print("✅ API key validation working correctly (403 for invalid key)")
            return True
        else:
            print(f"❌ Expected 403, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"🚨 Error testing invalid API key: {e}")
        return False

def test_invalid_audio_format():
    """Test unsupported audio format handling"""
    print("\n🎵 Testing unsupported audio format...")
    
    request_data = {
        "audio_base64": "dGVzdA==",  # base64 of "test"
        "audio_format": "mp3",  # Unsupported in current implementation
        "expected_text": "test"
    }
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/pronunciation-analysis/assess/us",
            json=request_data,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 400:
            print("✅ Audio format validation working correctly (400 for unsupported format)")
            return True
        else:
            print(f"❌ Expected 400, got {response.status_code}")
            print(f"📝 Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"🚨 Error testing invalid audio format: {e}")
        return False

def check_server_health():
    """Check if the server is running"""
    print("🏥 Checking server health...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running (FastAPI docs accessible)")
            return True
        else:
            print(f"⚠️  Server responded with {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"🚨 Error checking server health: {e}")
        return False

def main():
    print("🚀 Audio Analysis API Test Client")
    print("=" * 50)
    
    # Check if server is running
    if not check_server_health():
        print("\n💡 To start the server, run:")
        print("   cd audio-analysis/main/backend")
        print("   python main.py")
        return
    
    # Run tests
    tests = [
        ("Basic pronunciation assessment", test_pronunciation_assessment),
        ("API key validation", test_invalid_api_key),
        ("Audio format validation", test_invalid_audio_format),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        if test_func():
            passed += 1
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the server logs.")

if __name__ == "__main__":
    main() 