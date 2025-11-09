#!/usr/bin/env python3
"""
Download book data during Railway build
"""
import os
import zipfile
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_book_data():
    """Download book data from Google Drive during build"""

    # File ID from your GDrive
    file_id = "1NTgs97rvlVHYozTfodNo5kKsambOpXr1"

    # Install gdown
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown'], check=True)

    import gdown

    # Download zip file
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = "book_data.zip"

    logger.info("📥 Downloading book data from Google Drive...")
    gdown.download(url, output_path, quiet=False)

    # Extract to book_data/
    logger.info("📦 Extracting book data...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("book_data")

    # Clean up zip
    os.remove(output_path)

    logger.info("✅ Book data downloaded and extracted successfully!")

    # List available books
    book_data_dir = Path("book_data")
    if book_data_dir.exists():
        books = [item.name for item in book_data_dir.iterdir()
                if item.is_dir() and (item / "vdb_entities.json").exists()]
        logger.info(f"📚 Available books: {books}")

if __name__ == "__main__":
    download_book_data()