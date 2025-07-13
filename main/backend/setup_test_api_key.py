#!/usr/bin/env python3
"""
Script to set up a test API key in Supabase database
Run this once to add the test API key that matches the test client
"""

import os
import hashlib
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def setup_test_api_key():
    """Add the test API key to Supabase database"""
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        print("❌ ERROR: Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env file")
        print("Please make sure you have a .env file with your Supabase credentials")
        return False
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✅ Connected to Supabase")
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False
    
    # The test API key that the test client uses
    test_api_key = "test_client_key_123"
    hashed_key = hashlib.sha256(test_api_key.encode('utf-8')).hexdigest()
    
    try:
        # Check if the test key already exists
        existing = supabase.table("api_keys").select("id").eq("hashed_api_key", hashed_key).execute()
        
        if existing.data:
            print(f"✅ Test API key already exists with ID: {existing.data[0]['id']}")
            return True
        
        # Insert the test API key
        result = supabase.table("api_keys").insert({
            "hashed_api_key": hashed_key,
            "description": "Test API Key for local development",
            "is_active": True,
            "usage_count": 0
        }).execute()
        
        if result.data:
            print(f"✅ Test API key created with ID: {result.data[0]['id']}")
            print(f"🔑 Test API key: {test_api_key}")
            print(f"🔗 Hash: {hashed_key[:16]}...")
            return True
        else:
            print("❌ Failed to create test API key")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up test API key: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Setting up test API key in Supabase...")
    print("=" * 50)
    
    success = setup_test_api_key()
    
    if success:
        print("\n🎉 Setup complete! You can now test the API with:")
        print("   python test_client.py")
    else:
        print("\n❌ Setup failed. Please check your .env file and Supabase configuration.") 