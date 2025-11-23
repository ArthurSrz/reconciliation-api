"""
BookIngestion model for tracking book ingestion jobs.

This model tracks the status and progress of book ingestion through the
nano-graphRAG pipeline. It supports admin-only book addition as per US2.

Constitution Compliance:
- Principle VI: Extensible Literature Foundation - enables easy book addition
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
import uuid


class IngestionStatus(Enum):
    """Status values for book ingestion jobs."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class FileFormat(Enum):
    """Supported file formats for book ingestion."""
    PDF = "pdf"
    EPUB = "epub"
    TXT = "txt"


@dataclass
class BookMetadata:
    """Metadata for a book being ingested."""
    title: str
    author: str
    genre: Optional[str] = None
    publication_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "genre": self.genre,
            "publication_date": self.publication_date
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookMetadata":
        return cls(
            title=data.get("title", ""),
            author=data.get("author", ""),
            genre=data.get("genre"),
            publication_date=data.get("publication_date")
        )


@dataclass
class BookIngestion:
    """
    Model for tracking book ingestion jobs through the nano-graphRAG pipeline.

    This model stores all information needed to track, resume, and rollback
    book ingestion operations.

    Attributes:
        id: Unique identifier for the ingestion job
        book_id: Identifier for the book being created
        file_path: Path to the source file
        file_format: Format of the source file (pdf, epub, txt)
        status: Current status of the ingestion job
        progress: Progress indicator from 0.0 to 1.0
        chunks_count: Number of text chunks created
        entities_extracted: Number of entities extracted
        relationships_created: Number of relationships created
        inter_book_links: Number of cross-book connections made
        started_at: Timestamp when ingestion started
        completed_at: Timestamp when ingestion completed
        error_log: Error details if ingestion failed
        admin_id: ID of the admin who initiated ingestion
        config_hash: Hash of the nano-graphRAG configuration used
        book_metadata: Metadata about the book
    """

    # Required fields
    file_path: str
    book_metadata: BookMetadata
    admin_id: str

    # Auto-generated fields
    id: str = field(default_factory=lambda: f"ingestion-{uuid.uuid4()}")
    book_id: str = field(default_factory=lambda: f"book-{uuid.uuid4()}")

    # Status tracking
    file_format: FileFormat = FileFormat.TXT
    status: IngestionStatus = IngestionStatus.QUEUED
    progress: float = 0.0

    # Statistics
    chunks_count: int = 0
    entities_extracted: int = 0
    relationships_created: int = 0
    inter_book_links: int = 0

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Error handling
    error_log: Optional[str] = None

    # Configuration tracking
    config_hash: Optional[str] = None

    def __post_init__(self):
        """Detect file format from file path if not set."""
        if self.file_path:
            ext = self.file_path.lower().split('.')[-1]
            format_map = {
                'pdf': FileFormat.PDF,
                'epub': FileFormat.EPUB,
                'txt': FileFormat.TXT
            }
            self.file_format = format_map.get(ext, FileFormat.TXT)

    def start(self) -> None:
        """Mark the ingestion as started."""
        self.status = IngestionStatus.PROCESSING
        self.started_at = datetime.utcnow()
        self.progress = 0.0

    def update_progress(self, progress: float, **stats) -> None:
        """Update progress and optionally statistics."""
        self.progress = max(0.0, min(1.0, progress))
        for key, value in stats.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def complete(self, chunks: int, entities: int, relationships: int, inter_book: int) -> None:
        """Mark the ingestion as completed."""
        self.status = IngestionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.progress = 1.0
        self.chunks_count = chunks
        self.entities_extracted = entities
        self.relationships_created = relationships
        self.inter_book_links = inter_book

    def fail(self, error: str) -> None:
        """Mark the ingestion as failed."""
        self.status = IngestionStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_log = error

    def rollback(self) -> None:
        """Mark the ingestion as rolled back."""
        self.status = IngestionStatus.ROLLED_BACK
        self.completed_at = datetime.utcnow()

    def can_rollback(self) -> bool:
        """Check if rollback is possible."""
        return self.status in (IngestionStatus.COMPLETED, IngestionStatus.FAILED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "book_id": self.book_id,
            "file_path": self.file_path,
            "file_format": self.file_format.value,
            "status": self.status.value,
            "progress": self.progress,
            "chunks_count": self.chunks_count,
            "entities_extracted": self.entities_extracted,
            "relationships_created": self.relationships_created,
            "inter_book_links": self.inter_book_links,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_log": self.error_log,
            "admin_id": self.admin_id,
            "config_hash": self.config_hash,
            "book_metadata": self.book_metadata.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookIngestion":
        """Create instance from dictionary."""
        metadata = BookMetadata.from_dict(data.get("book_metadata", {}))
        instance = cls(
            file_path=data.get("file_path", ""),
            book_metadata=metadata,
            admin_id=data.get("admin_id", ""),
            id=data.get("id", f"ingestion-{uuid.uuid4()}"),
            book_id=data.get("book_id", f"book-{uuid.uuid4()}")
        )
        instance.file_format = FileFormat(data.get("file_format", "txt"))
        instance.status = IngestionStatus(data.get("status", "queued"))
        instance.progress = data.get("progress", 0.0)
        instance.chunks_count = data.get("chunks_count", 0)
        instance.entities_extracted = data.get("entities_extracted", 0)
        instance.relationships_created = data.get("relationships_created", 0)
        instance.inter_book_links = data.get("inter_book_links", 0)
        instance.error_log = data.get("error_log")
        instance.config_hash = data.get("config_hash")

        if data.get("started_at"):
            instance.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            instance.completed_at = datetime.fromisoformat(data["completed_at"])

        return instance


# In-memory storage for ingestion jobs (can be replaced with persistent storage)
_ingestion_jobs: Dict[str, BookIngestion] = {}


def store_ingestion_job(job: BookIngestion) -> None:
    """Store an ingestion job."""
    _ingestion_jobs[job.id] = job


def get_ingestion_job(job_id: str) -> Optional[BookIngestion]:
    """Retrieve an ingestion job by ID."""
    return _ingestion_jobs.get(job_id)


def get_all_ingestion_jobs(
    status: Optional[IngestionStatus] = None,
    limit: int = 20,
    offset: int = 0
) -> tuple[list[BookIngestion], int]:
    """Retrieve all ingestion jobs, optionally filtered by status."""
    jobs = list(_ingestion_jobs.values())

    if status:
        jobs = [j for j in jobs if j.status == status]

    # Sort by started_at descending (most recent first)
    jobs.sort(key=lambda j: j.started_at or datetime.min, reverse=True)

    total = len(jobs)
    return jobs[offset:offset + limit], total
