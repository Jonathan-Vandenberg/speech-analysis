#!/usr/bin/env python3
"""
Test script for pronunciation analysis with consonants audio file.
Tests the /pronunciation endpoint with "Peter Piper picked peppers. Three thick trees. She sells sea shells."
"""
import requests
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"

# Get API key from environment or use provided key
API_KEY = os.getenv("API_KEY", "sk-6bAjDoJWOc0Ugn9GTnznWKwp1kJurWmEqCUiX4_YIes")

def test_pronunciation_consonants():
    """Test pronunciation analysis with consonants audio file."""
    print("🎤 Testing pronunciation analysis with consonants audio...")
    
    # Expected text
    expected_text = "Peter Piper picked peppers. Three thick trees. She sells sea shells."
    
    # Path to audio file
    audio_file_path = Path(__file__).parent / "test_consonants.aiff"
    
    if not audio_file_path.exists():
        print(f"❌ Audio file not found: {audio_file_path}")
        return False
    
    print(f"📁 Audio file: {audio_file_path}")
    print(f"📝 Expected text: {expected_text}")
    
    # Prepare request
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Open audio file
    with open(audio_file_path, "rb") as audio_file:
        files = {
            "file": ("test_consonants.aiff", audio_file, "audio/aiff")
        }
        data = {
            "expected_text": expected_text
        }
        
        try:
            print("\n🔄 Sending request to /analyze/pronunciation...")
            response = requests.post(
                f"{API_BASE_URL}/analyze/pronunciation",
                headers=headers,
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n✅ Pronunciation analysis successful!")
                print(f"\n📊 Results:")
                print(f"   Overall Score: {result.get('pronunciation', {}).get('overall_score', 0):.2f}%")
                print(f"   Predicted Text: {result.get('predicted_text', 'N/A')}")
                
                print(f"\n📝 Word-by-word analysis:")
                words = result.get('pronunciation', {}).get('words', [])
                for word_data in words:
                    word_text = word_data.get('word_text', '')
                    word_score = word_data.get('word_score', 0)
                    phonemes = word_data.get('phonemes', [])
                    
                    print(f"\n   Word: '{word_text}' (Score: {word_score:.2f}%)")
                    for phoneme in phonemes:
                        ipa = phoneme.get('ipa_label', '')
                        score = phoneme.get('phoneme_score', 0)
                        print(f"      {ipa}: {score:.2f}%")
                
                # Save full results to JSON file
                output_file = Path(__file__).parent / "test_consonants_results.json"
                with open(output_file, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"\n💾 Full results saved to: {output_file}")
                
                return True
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out (60s)")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("=" * 60)
    print("Pronunciation Analysis Test - Consonants")
    print("=" * 60)
    
    success = test_pronunciation_consonants()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    print("=" * 60)

