#!/usr/bin/env python3
"""
Startup script for Railway deployment
Downloads book data and starts the reconciliation API
"""
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_book_data():
    """Ensure book data is available"""
    book_data_dir = Path("book_data")

    # Check if data already exists
    if book_data_dir.exists():
        books = [item.name for item in book_data_dir.iterdir()
                if item.is_dir() and (item / "vdb_entities.json").exists()]
        if books:
            logger.info(f"✅ Book data already available: {books}")
            return

    # Download book data
    logger.info("📥 Book data not found, downloading...")
    try:
        from download_book_data import download_book_data
        download_book_data()
    except Exception as e:
        logger.error(f"❌ Failed to download book data: {e}")
        # Continue anyway - app might work with fallback

def start_app():
    """Start the reconciliation API"""
    logger.info("🚀 Starting Reconciliation API...")

    # Ensure book data is available
    ensure_book_data()

    # Import and run the main app
    from reconciliation_api import app

    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    start_app()