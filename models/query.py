"""
Query and QueryResult node models for provenance tracking
Feature: 001-interactive-graphrag-refinement
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import uuid


@dataclass
class Query:
    """
    Represents a GraphRAG query with full provenance tracking.

    Constitutional Principles:
    - Principle #5: End-to-end interpretability through complete provenance chain
    - Each query captures the complete context needed to trace answer to sources
    """

    id: str
    question: str
    answer_text: str
    timestamp: datetime
    version: int = 1
    parent_query_id: Optional[str] = None
    mode: str = "local"  # "local" or "global"
    book_context: List[str] = field(default_factory=list)
    user_id: str = "default_user"
    status: str = "active"  # "active", "superseded", "archived"

    @classmethod
    def create_new(
        cls,
        question: str,
        answer_text: str,
        mode: str = "local",
        book_context: Optional[List[str]] = None,
        user_id: str = "default_user"
    ) -> "Query":
        """Create a new Query (version 1)"""
        return cls(
            id=f"query-{uuid.uuid4()}",
            question=question,
            answer_text=answer_text,
            timestamp=datetime.now(),
            version=1,
            mode=mode,
            book_context=book_context or [],
            user_id=user_id,
            status="active"
        )

    @classmethod
    def create_revision(
        cls,
        parent: "Query",
        new_answer: str,
        user_id: Optional[str] = None
    ) -> "Query":
        """Create a new version of an existing query after graph edits"""
        return cls(
            id=f"query-{uuid.uuid4()}",
            question=parent.question,  # Same question
            answer_text=new_answer,    # New answer after edits
            timestamp=datetime.now(),
            version=parent.version + 1,
            parent_query_id=parent.id,
            mode=parent.mode,
            book_context=parent.book_context,
            user_id=user_id or parent.user_id,
            status="active"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Query":
        """Create Query from Neo4j node properties"""
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class QueryResult:
    """
    Represents the result metadata for a query execution.
    Links to entities, relationships, and communities used to generate answer.
    """

    id: str
    query_id: str
    timestamp: datetime
    execution_time_ms: int = 0
    entity_count: int = 0
    relationship_count: int = 0
    community_count: int = 0

    @classmethod
    def create_new(
        cls,
        query_id: str,
        execution_time_ms: int = 0,
        entity_count: int = 0,
        relationship_count: int = 0,
        community_count: int = 0
    ) -> "QueryResult":
        """Create a new QueryResult"""
        return cls(
            id=f"result-{uuid.uuid4()}",
            query_id=query_id,
            timestamp=datetime.now(),
            execution_time_ms=execution_time_ms,
            entity_count=entity_count,
            relationship_count=relationship_count,
            community_count=community_count
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryResult":
        """Create QueryResult from Neo4j node properties"""
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)
