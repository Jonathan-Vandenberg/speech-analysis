#!/usr/bin/env python3
"""
Quick test to check what's happening with the health endpoint
"""
import os
from dotenv import load_dotenv
load_dotenv()

print("Testing database manager initialization...")

try:
    from main.database import DatabaseManager
    print("Importing DatabaseManager...")
    
    db = DatabaseManager()
    print(f"DatabaseManager created. Client exists: {db.client is not None}")
    print(f"is_available(): {db.is_available()}")
    
    # Test a simple query with timeout
    if db.client:
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError('Query timed out')
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        try:
            # This is what might be hanging
            result = db.client.table('api_keys').select('id').limit(1).execute()
            signal.alarm(0)
            print(f"✅ Query successful: {len(result.data)} rows")
        except Exception as e:
            signal.alarm(0)
            print(f"❌ Query failed: {e}")
    
except Exception as e:
    print(f"❌ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
