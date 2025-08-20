#!/usr/bin/env python3
"""
Test the health endpoint directly
"""
import asyncio
import sys
import os
from dotenv import load_dotenv

async def test_health():
    """Test the health check function directly"""
    print("Testing health check function...")
    
    try:
        # Import the app module
        from main.app import db_manager
        
        print("✅ Successfully imported db_manager")
        print(f"db_manager.is_available(): {db_manager.is_available()}")
        
        # Test the actual health check logic
        db_status = "connected" if db_manager.is_available() else "disconnected"
        result = {
            "status": "ok",
            "database": db_status,
            "version": "1.0.0"
        }
        
        print(f"✅ Health check result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error in health check: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    load_dotenv()
    result = asyncio.run(test_health())
    if result:
        print("✅ Health check completed successfully")
    else:
        print("❌ Health check failed")
        sys.exit(1)
