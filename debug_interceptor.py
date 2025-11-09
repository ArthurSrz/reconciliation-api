#!/usr/bin/env python3
"""
Debug script to test GraphRAG with interceptor exactly like production
"""
import os
import logging
from pathlib import Path
from nano_graphrag import GraphRAG, QueryParam
from graphrag_interceptor import graphrag_interceptor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_interceptor_init(book_path):
    """Test GraphRAG initialization with interceptor exactly like production"""
    try:
        logger.info(f"📥 Ensuring GraphRAG data is available for book: a_rebours_huysmans")

        if not Path(book_path).exists():
            logger.error(f"❌ Book path does not exist: {book_path}")
            return None

        # Check required files
        required_files = ['vdb_entities.json']
        for file in required_files:
            if not (Path(book_path) / file).exists():
                logger.error(f"❌ Required file missing: {file}")
                return None

        logger.info(f"✅ All required files found in: {book_path}")

        # Initialize like in production
        from nano_graphrag._llm import gpt_4o_mini_complete

        # Créer l'intercepteur LLM comme dans test_query_analysis.py
        logger.info("🔧 Creating intercepted LLM function...")
        intercepted_llm = graphrag_interceptor.intercept_query_processing(gpt_4o_mini_complete)
        logger.info("✅ LLM interceptor created")

        # Intercepter aussi la fonction _build_local_query_context pour capturer les vraies entités
        try:
            logger.info("🔧 Intercepting _build_local_query_context function...")
            from nano_graphrag._op import _build_local_query_context
            original_build_context = _build_local_query_context
            intercepted_build_context = graphrag_interceptor.intercept_build_local_query_context(original_build_context)

            # Remplacer temporairement la fonction dans le module
            import nano_graphrag._op
            nano_graphrag._op._build_local_query_context = intercepted_build_context
            logger.info("✅ Successfully intercepted _build_local_query_context function")
        except Exception as e:
            logger.warning(f"⚠️ Could not intercept _build_local_query_context: {e}")

        logger.info("🔧 Creating GraphRAG instance...")
        local_graphrag = GraphRAG(
            working_dir=book_path,
            best_model_func=intercepted_llm,
            cheap_model_func=intercepted_llm,
            embedding_func_max_async=4,
            best_model_max_async=2,
            cheap_model_max_async=4,
            embedding_batch_num=16,
            graph_cluster_algorithm="leiden"
        )

        logger.info(f"✅ Local GraphRAG initialized with GDRIVE data from: {book_path}")
        return local_graphrag

    except Exception as e:
        logger.error(f"❌ Failed to initialize local GraphRAG with GDrive data: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    book_path = "/Users/arthursarazin/Documents/nano-graphrag/reconciliation-api/book_data/a_rebours_huysmans"

    logger.info("=== Testing GraphRAG with Interceptor (Production Style) ===")
    graphrag = test_interceptor_init(book_path)

    if graphrag:
        logger.info("=== Testing query with interceptor ===")
        try:
            result = graphrag.query("What is the main theme?", param=QueryParam(mode="local"))
            logger.info(f"✅ Query successful: {result[:100]}...")

            # Check interceptor data
            debug_info = graphrag_interceptor.get_real_debug_info()
            logger.info(f"🔍 Interceptor captured: {len(debug_info.get('processing_phases', {}).get('entity_selection', {}).get('entities', []))} entities")
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            import traceback
            logger.error(f"❌ Query traceback: {traceback.format_exc()}")
    else:
        logger.error("❌ Cannot test query - GraphRAG initialization failed")