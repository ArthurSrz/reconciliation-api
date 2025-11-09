#!/usr/bin/env python3
"""
Debug script to identify GraphRAG initialization issues
"""
import os
import json
import logging
from pathlib import Path
from nano_graphrag import GraphRAG, QueryParam

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_json_files(book_path):
    """Test if all JSON files are valid"""
    json_files = ['vdb_entities.json', 'kv_store_community_reports.json',
                  'kv_store_full_docs.json', 'kv_store_llm_response_cache.json',
                  'kv_store_text_chunks.json']

    for filename in json_files:
        filepath = Path(book_path) / filename
        try:
            if filepath.exists():
                logger.info(f"Testing {filename}...")
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.info(f"✅ {filename} - Valid JSON with {type(data)} containing {len(data) if hasattr(data, '__len__') else 'N/A'} items")
            else:
                logger.warning(f"❌ {filename} - File not found")
        except json.JSONDecodeError as e:
            logger.error(f"❌ {filename} - JSON Error: {e}")
        except Exception as e:
            logger.error(f"❌ {filename} - Other Error: {e}")

def test_graphrag_init(book_path):
    """Test GraphRAG initialization step by step"""
    try:
        logger.info(f"Testing GraphRAG initialization with path: {book_path}")

        # Test minimal GraphRAG init
        from nano_graphrag._llm import gpt_4o_mini_complete

        graphrag = GraphRAG(
            working_dir=book_path,
            best_model_func=gpt_4o_mini_complete,
            cheap_model_func=gpt_4o_mini_complete
        )

        logger.info("✅ GraphRAG initialized successfully")
        return graphrag

    except Exception as e:
        logger.error(f"❌ GraphRAG initialization failed: {e}")
        return None

if __name__ == "__main__":
    book_path = "/Users/arthursarazin/Documents/nano-graphrag/reconciliation-api/book_data/a_rebours_huysmans"

    logger.info("=== Testing JSON files ===")
    test_json_files(book_path)

    logger.info("=== Testing GraphRAG initialization ===")
    graphrag = test_graphrag_init(book_path)

    if graphrag:
        logger.info("=== Testing query ===")
        try:
            result = graphrag.query("What is the main character?", param=QueryParam(mode="local"))
            logger.info(f"✅ Query successful: {result[:100]}...")
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")