#!/usr/bin/env python3
"""
Test script for API key creation, validation, and usage tracking.
Run this after setting up the database and starting the API server.
"""
import requests
import json
import time
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
ADMIN_BASE_URL = "http://localhost:3001"

def test_health_check():
    """Test API health and database connection."""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/healthz")
        data = response.json()
        print(f"✅ Health check: {data}")
        
        if data.get("database") == "connected":
            print("✅ Database is connected")
        else:
            print("❌ Database is not connected - some tests will fail")
            
        return data.get("database") == "connected"
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_create_api_key():
    """Test API key creation via admin endpoint."""
    print("\n🔑 Testing API key creation...")
    try:
        # Create API key using form data (as expected by backend)
        data = {
            "description": "Test API Key",
            "minute_limit": "5",
            "daily_limit": "100",
            "monthly_limit": "1000"
        }
        
        response = requests.post(f"{API_BASE_URL}/api/admin/keys", data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API key created: {result['api_key'][:20]}...")
            return result['api_key']
        else:
            print(f"❌ Failed to create API key: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ API key creation failed: {e}")
        return None

def test_api_key_authentication(api_key):
    """Test API key authentication."""
    print(f"\n🔐 Testing API key authentication...")
    
    if not api_key:
        print("❌ No API key to test with")
        return False
    
    # Test with valid API key
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Create a simple test request to scripted endpoint
        data = {
            "expected_text": "hello world",
            "browser_transcript": "hello world"
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze/scripted", headers=headers, data=data)
        
        if response.status_code == 200:
            print("✅ Valid API key accepted")
            return True
        else:
            print(f"❌ Request with valid API key failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        return False

def test_invalid_api_key():
    """Test that invalid API keys are rejected."""
    print(f"\n🚫 Testing invalid API key rejection...")
    
    # Test with invalid API key
    headers = {"Authorization": "Bearer invalid-key-12345"}
    
    try:
        data = {
            "expected_text": "hello world",
            "browser_transcript": "hello world"
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze/scripted", headers=headers, data=data)
        
        if response.status_code == 403:
            print("✅ Invalid API key correctly rejected")
            return True
        else:
            print(f"❌ Invalid API key not rejected: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Invalid key test failed: {e}")
        return False

def test_no_api_key():
    """Test that requests without API keys are rejected."""
    print(f"\n❌ Testing request without API key...")
    
    try:
        data = {
            "expected_text": "hello world",
            "browser_transcript": "hello world"
        }
        
        response = requests.post(f"{API_BASE_URL}/analyze/scripted", data=data)
        
        if response.status_code == 403:
            print("✅ Request without API key correctly rejected")
            return True
        else:
            print(f"❌ Request without API key not rejected: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ No API key test failed: {e}")
        return False

def test_usage_tracking(api_key):
    """Test that usage is tracked in analytics."""
    print(f"\n📊 Testing usage tracking...")
    
    if not api_key:
        print("❌ No API key to test with")
        return False
    
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        # Make a few requests
        for i in range(3):
            data = {
                "expected_text": f"test request {i+1}",
                "browser_transcript": f"test request {i+1}"
            }
            
            response = requests.post(f"{API_BASE_URL}/analyze/scripted", headers=headers, data=data)
            if response.status_code != 200:
                print(f"❌ Request {i+1} failed: {response.status_code}")
                return False
            
        print("✅ Made 3 test requests")
        
        # Wait a moment for tracking to complete
        time.sleep(2)
        
        # Check analytics
        analytics_response = requests.get(f"{API_BASE_URL}/api/admin/analytics")
        if analytics_response.status_code == 200:
            analytics = analytics_response.json()
            print(f"✅ Analytics endpoint working")
            
            # Look for our API key in the results
            if "api_keys" in analytics and analytics["api_keys"]:
                print(f"✅ Found usage data for {len(analytics['api_keys'])} API keys")
                return True
            else:
                print("❌ No usage data found in analytics")
                return False
        else:
            print(f"❌ Analytics request failed: {analytics_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Usage tracking test failed: {e}")
        return False

def test_admin_endpoints():
    """Test admin endpoints."""
    print(f"\n⚙️ Testing admin endpoints...")
    
    try:
        # Test getting API keys
        response = requests.get(f"{API_BASE_URL}/api/admin/keys")
        if response.status_code == 200:
            keys = response.json()
            print(f"✅ Retrieved {len(keys.get('api_keys', []))} API keys")
            return True
        else:
            print(f"❌ Failed to get API keys: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Admin endpoints test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting API integration tests...\n")
    
    # Track test results
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health_check()
    
    if not results['health']:
        print("\n❌ Database not connected. Stopping tests.")
        print("Make sure to:")
        print("1. Set SUPABASE_URL and SUPABASE_ANON_KEY environment variables")
        print("2. Run the database_schema.sql script in Supabase")
        print("3. Start the API server with: uvicorn main.app:app --reload")
        return
    
    # Test 2: Admin endpoints
    results['admin'] = test_admin_endpoints()
    
    # Test 3: API key creation
    api_key = test_create_api_key()
    results['create_key'] = api_key is not None
    
    # Test 4: Authentication
    results['auth_valid'] = test_api_key_authentication(api_key)
    results['auth_invalid'] = test_invalid_api_key()
    results['auth_none'] = test_no_api_key()
    
    # Test 5: Usage tracking
    results['tracking'] = test_usage_tracking(api_key)
    
    # Summary
    print(f"\n📋 Test Results Summary:")
    print(f"{'='*50}")
    
    passed = 0
    total = 0
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.ljust(20)}: {status}")
        if result:
            passed += 1
        total += 1
    
    print(f"{'='*50}")
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API key system is working correctly.")
        print("\nNext steps:")
        print("1. Start the admin frontend: cd ../school-ai-admin && npm run dev")
        print("2. Visit http://localhost:3001 to manage API keys")
        print("3. Use the created API keys to access the speech analysis endpoints")
    else:
        print(f"\n❌ {total - passed} tests failed. Please check the setup and try again.")

if __name__ == "__main__":
    main()
