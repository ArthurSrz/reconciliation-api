#!/usr/bin/env python3
"""
Book Ingestion CLI - Admin tool for adding books to the Borges Library.

This CLI tool provides an interface for administrators to ingest books
through the nano-graphRAG pipeline.

Usage:
    python -m cli.ingest_book --file /path/to/book.pdf --title "Book Title" --author "Author Name"

    # With optional metadata
    python -m cli.ingest_book --file book.txt --title "Book Title" --author "Author" --genre "Fiction"

    # Batch processing
    python -m cli.ingest_book --batch /path/to/books.json

Constitution Compliance:
- Principle VI: Extensible Literature Foundation - admin-only CLI access
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.book_ingestion import BookMetadata, IngestionStatus
from services.book_ingestion_service import BookIngestionService
from middleware.admin_auth import validate_cli_admin, AdminAuthError


# Default raw books directory
RAW_BOOKS_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data/raw_books"


class ProgressDisplay:
    """Display progress updates in terminal."""

    def __init__(self):
        self.last_progress = -1
        self.start_time = None

    def start(self, title: str):
        """Start progress display."""
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Starting ingestion: {title}")
        print(f"{'='*60}")

    def update(self, job):
        """Update progress display."""
        progress_pct = int(job.progress * 100)
        if progress_pct != self.last_progress:
            self.last_progress = progress_pct
            elapsed = time.time() - self.start_time if self.start_time else 0
            status_char = self._get_status_char(job.status)
            bar = self._progress_bar(progress_pct)

            print(f"\r{status_char} [{bar}] {progress_pct}% | "
                  f"Chunks: {job.chunks_count} | "
                  f"Entities: {job.entities_extracted} | "
                  f"Relations: {job.relationships_created} | "
                  f"Time: {elapsed:.1f}s", end='', flush=True)

    def complete(self, job):
        """Complete progress display."""
        print()  # New line after progress bar
        elapsed = time.time() - self.start_time if self.start_time else 0

        if job.status == IngestionStatus.COMPLETED:
            print(f"\n✅ Ingestion completed successfully!")
            print(f"   Job ID: {job.id}")
            print(f"   Book ID: {job.book_id}")
            print(f"   Chunks: {job.chunks_count}")
            print(f"   Entities: {job.entities_extracted}")
            print(f"   Relationships: {job.relationships_created}")
            print(f"   Inter-book links: {job.inter_book_links}")
            print(f"   Duration: {elapsed:.1f}s")
        else:
            print(f"\n❌ Ingestion failed: {job.error_log}")
            print(f"   Job ID: {job.id}")
            print(f"   Use --rollback {job.id} to clean up partial data")

    def _progress_bar(self, pct: int, width: int = 30) -> str:
        """Create ASCII progress bar."""
        filled = int(width * pct / 100)
        return '█' * filled + '░' * (width - filled)

    def _get_status_char(self, status: IngestionStatus) -> str:
        """Get status character."""
        return {
            IngestionStatus.QUEUED: '⏳',
            IngestionStatus.PROCESSING: '🔄',
            IngestionStatus.COMPLETED: '✅',
            IngestionStatus.FAILED: '❌',
            IngestionStatus.ROLLED_BACK: '↩️'
        }.get(status, '?')


def resolve_file_path(file_path: str) -> str:
    """Resolve file path, checking raw_books directory if needed."""
    path = Path(file_path)

    # If absolute path exists, use it
    if path.is_absolute() and path.exists():
        return str(path)

    # If relative path exists, use it
    if path.exists():
        return str(path.absolute())

    # Check in raw_books directory
    raw_path = Path(RAW_BOOKS_DIR) / path.name
    if raw_path.exists():
        return str(raw_path)

    # Check just the filename in raw_books
    raw_path = Path(RAW_BOOKS_DIR) / path.name
    if raw_path.exists():
        return str(raw_path)

    raise FileNotFoundError(
        f"File not found: {file_path}\n"
        f"Checked:\n"
        f"  - {path.absolute()}\n"
        f"  - {raw_path}"
    )


async def ingest_single_book(
    file_path: str,
    title: str,
    author: str,
    genre: Optional[str] = None,
    publication_date: Optional[str] = None,
    admin_id: str = "cli-admin"
) -> Dict[str, Any]:
    """Ingest a single book."""
    # Resolve file path
    resolved_path = resolve_file_path(file_path)

    # Create metadata
    metadata = BookMetadata(
        title=title,
        author=author,
        genre=genre,
        publication_date=publication_date
    )

    # Create service
    service = BookIngestionService()

    # Set up progress display
    progress = ProgressDisplay()
    progress.start(title)
    service.set_progress_callback(progress.update)

    # Run ingestion
    try:
        job = await service.ingest_book(resolved_path, metadata, admin_id)
        progress.complete(job)
        return job.to_dict()
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        raise


async def ingest_batch(batch_file: str, admin_id: str = "cli-admin") -> List[Dict[str, Any]]:
    """
    Ingest multiple books from a batch file.

    Batch file format (JSON):
    [
        {
            "file": "book1.txt",
            "title": "Book Title 1",
            "author": "Author 1",
            "genre": "Fiction"
        },
        ...
    ]
    """
    with open(batch_file) as f:
        books = json.load(f)

    results = []
    total = len(books)

    print(f"\n📚 Batch ingestion: {total} books")
    print(f"{'='*60}")

    for i, book in enumerate(books, 1):
        print(f"\n[{i}/{total}] Processing: {book.get('title', 'Unknown')}")

        try:
            result = await ingest_single_book(
                file_path=book['file'],
                title=book['title'],
                author=book['author'],
                genre=book.get('genre'),
                publication_date=book.get('publication_date'),
                admin_id=admin_id
            )
            results.append({
                'book': book['title'],
                'status': 'success',
                'result': result
            })
        except Exception as e:
            results.append({
                'book': book['title'],
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = total - successful

    print(f"\n{'='*60}")
    print(f"📊 Batch Summary:")
    print(f"   Total: {total}")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"{'='*60}")

    return results


async def rollback_job(job_id: str) -> Dict[str, Any]:
    """Rollback an ingestion job."""
    service = BookIngestionService()

    print(f"\n↩️  Rolling back job: {job_id}")
    result = await service.rollback(job_id)

    print(f"✅ Rollback completed:")
    print(f"   Entities deleted: {result['entities_deleted']}")
    print(f"   Relationships deleted: {result['relationships_deleted']}")
    print(f"   Duration: {result['duration_ms']}ms")

    return result


def list_raw_books():
    """List available books in raw_books directory."""
    raw_path = Path(RAW_BOOKS_DIR)

    if not raw_path.exists():
        print(f"❌ Raw books directory not found: {RAW_BOOKS_DIR}")
        return

    print(f"\n📂 Available books in {RAW_BOOKS_DIR}:")
    print(f"{'='*60}")

    files = list(raw_path.glob('*'))
    if not files:
        print("   (no files found)")
        return

    for f in sorted(files):
        if f.is_file():
            size = f.stat().st_size
            size_str = f"{size / 1024:.1f} KB" if size < 1024*1024 else f"{size / (1024*1024):.1f} MB"
            print(f"   📄 {f.name} ({size_str})")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Book Ingestion CLI - Add books to the Borges Library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single book
  python -m cli.ingest_book --file maison_vide.txt --title "La Maison Vide" --author "Jean Dupont"

  # With optional metadata
  python -m cli.ingest_book --file book.pdf --title "Title" --author "Author" --genre "Fiction"

  # Batch processing
  python -m cli.ingest_book --batch books.json

  # List available books
  python -m cli.ingest_book --list

  # Rollback an ingestion
  python -m cli.ingest_book --rollback ingestion-xxx-xxx

Environment:
  ADMIN_API_KEY   Required admin API key for authentication
        """
    )

    # Single book arguments
    parser.add_argument('--file', '-f', help='Path to book file (PDF, EPUB, or TXT)')
    parser.add_argument('--title', '-t', help='Book title')
    parser.add_argument('--author', '-a', help='Author name')
    parser.add_argument('--genre', '-g', help='Literary genre (optional)')
    parser.add_argument('--publication-date', '-d', help='Publication date (optional)')

    # Batch processing
    parser.add_argument('--batch', '-b', help='JSON file with batch of books to ingest')

    # Other operations
    parser.add_argument('--rollback', '-r', help='Rollback an ingestion job by ID')
    parser.add_argument('--list', '-l', action='store_true', help='List available books in raw_books directory')

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate admin access
    try:
        admin_id = validate_cli_admin()
    except AdminAuthError as e:
        print(f"❌ Authentication error: {e.message}")
        print("   Set ADMIN_API_KEY environment variable to enable admin access.")
        sys.exit(1)

    # Handle different operations
    if args.list:
        list_raw_books()
        return

    if args.rollback:
        asyncio.run(rollback_job(args.rollback))
        return

    if args.batch:
        asyncio.run(ingest_batch(args.batch, admin_id))
        return

    # Single book ingestion
    if not args.file:
        parser.error("--file is required for single book ingestion")
    if not args.title:
        parser.error("--title is required")
    if not args.author:
        parser.error("--author is required")

    asyncio.run(ingest_single_book(
        file_path=args.file,
        title=args.title,
        author=args.author,
        genre=args.genre,
        publication_date=args.publication_date,
        admin_id=admin_id
    ))


if __name__ == '__main__':
    main()
