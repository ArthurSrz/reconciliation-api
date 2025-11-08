"""
Google Drive Data Manager for Reconciliation API
Automatically downloads and manages GraphRAG data from Google Drive
"""

import os
import zipfile
import requests
import logging
import shutil
from pathlib import Path
from typing import Optional
import time

logger = logging.getLogger(__name__)

class GDriveDataManager:
    """
    Manages downloading and extracting GraphRAG data from Google Drive
    """

    def __init__(self, file_id: str = "1NTgs97rvlVHYozTfodNo5kKsambOpXr1"):
        self.file_id = file_id
        # Use direct download URL for Google Drive
        self.download_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
        self.data_dir = Path("./gdrive_data")
        self.zip_path = self.data_dir / "book_data.zip"
        self.extracted_dir = self.data_dir / "borges_graph"

    def ensure_data_available(self) -> str:
        """
        Ensure GraphRAG data is available locally
        Returns path to the data directory
        """

        # Check if data already exists
        if self._is_data_valid():
            logger.info(f"✅ GraphRAG data already available at {self.extracted_dir}")
            return str(self.extracted_dir)

        # Download and extract data
        logger.info("📥 Downloading GraphRAG data from Google Drive...")
        if self._download_data():
            if self._extract_data():
                logger.info(f"✅ GraphRAG data ready at {self.extracted_dir}")
                return str(self.extracted_dir)

        raise Exception("Failed to download or extract GraphRAG data")

    def _is_data_valid(self) -> bool:
        """Check if extracted data exists and is valid"""
        if not self.extracted_dir.exists():
            return False

        # Check for expected subdirectories (book folders)
        expected_books = ["a_rebours_huysmans", "chien_blanc_gary", "peau_bison_frison"]
        for book in expected_books:
            book_path = self.extracted_dir / book
            if not book_path.exists():
                return False

            # Check for essential GraphRAG files
            essential_files = [
                "vdb_entities.json",
                "kv_store_community_reports.json",
                "kv_store_full_docs.json"
            ]

            for file in essential_files:
                if not (book_path / file).exists():
                    return False

        return True

    def _download_data(self) -> bool:
        """Download data from Google Drive using gdown for better handling"""
        try:
            # Try using requests first for small files
            import subprocess
            import sys

            # Create data directory
            self.data_dir.mkdir(exist_ok=True)

            # Try using gdown if available, otherwise use requests
            try:
                # Try gdown first (better for Google Drive)
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 'gdown', '--quiet'
                ], capture_output=True)

                import gdown
                logger.info("📥 Using gdown for Google Drive download...")

                output_path = str(self.zip_path)
                url = f"https://drive.google.com/uc?id={self.file_id}"

                # Download using gdown
                gdown.download(url, output_path, quiet=False)

                if self.zip_path.exists() and self.zip_path.stat().st_size > 1000:
                    logger.info(f"✅ Downloaded with gdown: {self.zip_path} ({self.zip_path.stat().st_size} bytes)")
                    return True

            except Exception as e:
                logger.warning(f"⚠️ gdown failed: {e}, trying requests...")

            # Fallback to requests with improved handling
            session = requests.Session()
            session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})

            # Try direct download
            url = f"https://drive.google.com/uc?export=download&id={self.file_id}"
            response = session.get(url, stream=True, allow_redirects=True)

            if response.status_code == 200:
                # Check if this is the actual file or a confirmation page
                content_type = response.headers.get('content-type', '')

                if 'text/html' in content_type:
                    # This is likely a confirmation page, try to get the download link
                    logger.info("📄 Got HTML response, looking for download link...")

                    # Look for the download link in the HTML
                    import re
                    download_link_match = re.search(r'href="(/uc\?export=download[^"]+)"', response.text)

                    if download_link_match:
                        download_url = f"https://drive.google.com{download_link_match.group(1)}"
                        response = session.get(download_url, stream=True)

                # Save the file
                if response.status_code == 200:
                    total_size = int(response.headers.get('content-length', 0))

                    with open(self.zip_path, 'wb') as f:
                        downloaded = 0
                        chunk_size = 8192

                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)

                                if total_size > 0 and downloaded % (chunk_size * 10) == 0:
                                    percent = (downloaded / total_size) * 100
                                    logger.info(f"📥 Downloaded {percent:.1f}% ({downloaded}/{total_size} bytes)")

                    logger.info(f"✅ Downloaded {self.zip_path} ({downloaded} bytes)")

                    # Verify file size
                    if downloaded < 10000:  # Less than 10KB is suspicious
                        logger.warning(f"⚠️ Downloaded file seems too small ({downloaded} bytes)")
                        # Read first few bytes to check if it's HTML
                        with open(self.zip_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content_start = f.read(100)
                            if '<html' in content_start.lower():
                                logger.error("❌ Downloaded file is HTML, not ZIP")
                                return False

                    return True
                else:
                    logger.error(f"❌ Download failed with status {response.status_code}")
                    return False
            else:
                logger.error(f"❌ Initial request failed with status {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Download error: {e}")
            return False

    def _extract_data(self) -> bool:
        """Extract downloaded zip file"""
        try:
            if not self.zip_path.exists():
                logger.error("❌ Zip file not found")
                return False

            # Remove existing extracted directory
            if self.extracted_dir.exists():
                shutil.rmtree(self.extracted_dir)

            # Extract zip file
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)

            logger.info(f"✅ Extracted data to {self.data_dir}")

            # Clean up zip file
            self.zip_path.unlink()
            logger.info("🗑️ Cleaned up zip file")

            return True

        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return False

    def get_book_data_path(self, book_id: str = "a_rebours_huysmans") -> Optional[str]:
        """
        Get path to specific book data
        Returns None if book data not found
        """
        book_path = self.extracted_dir / book_id

        if book_path.exists():
            return str(book_path)
        else:
            logger.warning(f"⚠️ Book data not found: {book_id}")
            return None

    def list_available_books(self) -> list:
        """List all available book datasets"""
        if not self.extracted_dir.exists():
            return []

        books = []
        for item in self.extracted_dir.iterdir():
            if item.is_dir() and (item / "vdb_entities.json").exists():
                books.append(item.name)

        return sorted(books)

    def cleanup_old_data(self):
        """Remove old data to free space"""
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
            logger.info("🗑️ Cleaned up old data directory")

# Global instance
gdrive_manager = GDriveDataManager()