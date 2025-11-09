#!/usr/bin/env python3
"""
Test script to identify exact line causing NoneType error
"""
import traceback
import sys
import os
import logging
sys.path.append('.')

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

from nano_graphrag import GraphRAG, QueryParam

def test_with_detailed_traceback():
    """Test with detailed error tracking"""
    try:
        # Test racines_ciel_gary which is failing on Railway
        book_id = "racines_ciel_gary"
        working_dir = f"./book_data/{book_id}"

        print(f"📁 Testing book: {book_id}")
        print(f"📂 Working directory: {working_dir}")

        if not os.path.exists(working_dir):
            print(f"❌ Directory {working_dir} doesn't exist")
            return False

        # Initialize GraphRAG
        rag = GraphRAG(working_dir=working_dir, enable_llm_cache=True)

        # Test the exact query failing on Railway
        query = "qui est le personnage principal ?"
        param = QueryParam(mode="local")  # Test local mode first

        print(f"🔍 Testing LOCAL mode query: '{query}'")
        result = rag.query(query, param=param)
        print(f"✅ LOCAL mode successful")

        # Now test global mode which is causing the error
        param = QueryParam(mode="global")
        print(f"🔍 Testing GLOBAL mode query: '{query}'")
        result = rag.query(query, param=param)
        print(f"✅ GLOBAL mode successful")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        print("📍 Full traceback:")
        traceback.print_exc()

        # Print more detailed info about the error
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("\n🔍 Detailed traceback:")
        import traceback
        for line in traceback.format_tb(exc_traceback):
            print(line.strip())

        return False

if __name__ == "__main__":
    success = test_with_detailed_traceback()
    if not success:
        sys.exit(1)
    print("✅ All tests passed")