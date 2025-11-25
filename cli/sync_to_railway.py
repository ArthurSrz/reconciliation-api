#!/usr/bin/env python3
"""
Railway Volume Sync CLI - Upload processed book data to Railway volume.

This CLI tool syncs book data from the local directory to the Railway
volume where the production API reads GraphRAG data.

Usage:
    python -m cli.sync_to_railway --book-dir /path/to/book_data/book_name

    # Sync all books
    python -m cli.sync_to_railway --all

Constitution Compliance:
- Principle VI: Extensible Literature Foundation - ensures new books are available in production
"""

import argparse
import asyncio
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import base64

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


BOOK_DATA_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data"
RAILWAY_API_URL = os.getenv('GRAPHRAG_API_URL', 'https://reconciliation-api-production.up.railway.app')
ADMIN_API_KEY = os.getenv('ADMIN_API_KEY', '')


async def sync_book_to_railway(
    book_dir: str,
    api_url: str = RAILWAY_API_URL,
    admin_key: str = ADMIN_API_KEY
) -> Dict[str, Any]:
    """
    Sync a processed book directory to Railway volume.

    Args:
        book_dir: Path to the book's processed data directory
        api_url: Railway API URL
        admin_key: Admin API key

    Returns:
        Dict with sync statistics
    """
    book_path = Path(book_dir)

    if not book_path.exists():
        raise ValueError(f"Book directory not found: {book_dir}")

    print(f"\n📤 Syncing book to Railway: {book_path.name}")
    print(f"   Source: {book_dir}")
    print(f"   Target: {api_url}")

    # Create tarball of book data
    print("   Creating archive...")
    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        tar_path = tmp.name

    with tarfile.open(tar_path, 'w:gz') as tar:
        for file_path in book_path.iterdir():
            if file_path.is_file():
                tar.add(file_path, arcname=f"{book_path.name}/{file_path.name}")

    # Get archive size
    archive_size = os.path.getsize(tar_path)
    print(f"   Archive size: {archive_size / 1024:.1f} KB")

    # Read and encode archive
    with open(tar_path, 'rb') as f:
        archive_data = base64.b64encode(f.read()).decode('utf-8')

    # Clean up temp file
    os.unlink(tar_path)

    # Send to Railway
    print("   Uploading to Railway volume...")
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{api_url}/admin/upload-book-data",
            headers={
                'X-Admin-API-Key': admin_key,
                'Content-Type': 'application/json'
            },
            json={
                'book_name': book_path.name,
                'archive_data': archive_data,
                'archive_format': 'tar.gz'
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Upload successful!")
            return {
                'book_name': book_path.name,
                'status': 'success',
                'message': result.get('message', 'Uploaded successfully'),
                'volume_path': result.get('volume_path')
            }
        else:
            error_msg = response.text
            print(f"   ❌ Upload failed: {error_msg}")
            return {
                'book_name': book_path.name,
                'status': 'error',
                'error': error_msg,
                'status_code': response.status_code
            }


async def sync_all_books(
    api_url: str = RAILWAY_API_URL,
    admin_key: str = ADMIN_API_KEY
) -> List[Dict[str, Any]]:
    """Sync all books in book_data directory to Railway."""
    book_data = Path(BOOK_DATA_DIR)
    results = []

    print(f"\n📚 Syncing all books to Railway...")
    print(f"   Source: {BOOK_DATA_DIR}")
    print(f"   Target: {api_url}")

    for book_dir in sorted(book_data.iterdir()):
        if book_dir.is_dir() and (book_dir / "graph_chunk_entity_relation.graphml").exists():
            try:
                result = await sync_book_to_railway(str(book_dir), api_url, admin_key)
                results.append(result)
            except Exception as e:
                results.append({
                    'book_name': book_dir.name,
                    'status': 'error',
                    'error': str(e)
                })

    return results


def list_books_to_sync():
    """List books available for syncing to Railway."""
    book_data = Path(BOOK_DATA_DIR)

    if not book_data.exists():
        print(f"❌ Book data directory not found: {BOOK_DATA_DIR}")
        return

    print(f"\n📂 Books available for Railway sync:")
    print(f"{'='*60}")

    for d in sorted(book_data.iterdir()):
        if d.is_dir():
            graphml = d / "graph_chunk_entity_relation.graphml"
            if graphml.exists():
                size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
                print(f"   📖 {d.name} ({size / (1024*1024):.1f} MB)")
            else:
                print(f"   ⚠️  {d.name} (no GraphML - not ready)")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Sync processed book data to Railway volume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync a specific book
  python -m cli.sync_to_railway --book-dir /path/to/la_maison_vide_laurent_mauvignier

  # Sync all books
  python -m cli.sync_to_railway --all

  # List books available for sync
  python -m cli.sync_to_railway --list

Environment:
  GRAPHRAG_API_URL   Railway API URL (default: https://borgesgraph-production.up.railway.app)
  ADMIN_API_KEY      Admin API key for authentication
        """
    )

    parser.add_argument('--book-dir', '-d', help='Path to processed book directory')
    parser.add_argument('--all', '-a', action='store_true', help='Sync all books in book_data directory')
    parser.add_argument('--list', '-l', action='store_true', help='List books available for sync')
    parser.add_argument('--api-url', help=f'Railway API URL (default: {RAILWAY_API_URL})')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Get admin key from env
    admin_key = os.getenv('ADMIN_API_KEY', '')
    if not admin_key and not args.list:
        print("❌ ADMIN_API_KEY environment variable required")
        sys.exit(1)

    api_url = args.api_url or RAILWAY_API_URL

    if args.list:
        list_books_to_sync()
        return

    if args.all:
        results = asyncio.run(sync_all_books(api_url, admin_key))

        # Summary
        success = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - success

        print(f"\n{'='*60}")
        print(f"📊 Sync Summary:")
        print(f"   Total: {len(results)}")
        print(f"   ✅ Success: {success}")
        print(f"   ❌ Failed: {failed}")
        print(f"{'='*60}")
        return

    if args.book_dir:
        result = asyncio.run(sync_book_to_railway(args.book_dir, api_url, admin_key))

        print(f"\n{'='*60}")
        if result['status'] == 'success':
            print(f"✅ Sync completed!")
            print(f"   Book: {result['book_name']}")
            if result.get('volume_path'):
                print(f"   Volume path: {result['volume_path']}")
        else:
            print(f"❌ Sync failed!")
            print(f"   Error: {result.get('error')}")
        print(f"{'='*60}")
        return

    parser.error("Either --book-dir, --all, or --list is required")


if __name__ == '__main__':
    main()
