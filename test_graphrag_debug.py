#!/usr/bin/env python3
"""
Test script to debug GraphRAG NoneType error locally
"""
import traceback
import sys
import os
sys.path.append('.')

from nano_graphrag import GraphRAG, QueryParam

def test_graphrag_query():
    """Test GraphRAG with the same query that's failing on Railway"""
    try:
        print("🔍 Testing GraphRAG locally...")

        # Test with racines_ciel_gary like in the Railway error
        book_id = "racines_ciel_gary"
        working_dir = f"./nano_graphrag_cache_{book_id}"

        if not os.path.exists(working_dir):
            print(f"❌ Working directory {working_dir} does not exist")
            return False

        print(f"📁 Using working directory: {working_dir}")

        # Initialize GraphRAG
        rag = GraphRAG(working_dir=working_dir, enable_llm_cache=True)

        # Test the same query that's failing
        query = "qui est le personnage principal ?"
        print(f"❓ Testing query: '{query}'")

        # Test with local mode like Railway
        param = QueryParam(mode="local")
        print(f"🔧 Using mode: local")

        # Execute the query and catch the exact error
        print("🚀 Executing query...")
        result = rag.query(query, param=param)
        print(f"✅ Query successful: {len(result)} characters returned")
        print(f"📄 First 200 chars: {result[:200]}...")

        return True

    except Exception as e:
        print(f"❌ GraphRAG query failed with error: {e}")
        print(f"📍 Error type: {type(e).__name__}")
        print("🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_graphrag_query()
    if not success:
        sys.exit(1)
    print("✅ Local test completed successfully")