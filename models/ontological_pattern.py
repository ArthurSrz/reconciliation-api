"""
OntologicalPattern and PatternInstance models for cross-domain pattern discovery
Feature: 001-interactive-graphrag-refinement
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
import uuid


@dataclass
class OntologicalPattern:
    """
    Represents a recurring structural pattern discovered across multiple books/domains.

    Constitutional Principles:
    - Principle #3: Prioritize cross-book zones for investigation
    - Patterns are significant when they appear across multiple books

    Example pattern: "PERSON -INFLUENCES-> CONCEPT -APPEARS_IN-> BOOK"
    Found in physics (Einstein-Relativity), literature (Proust-Memory), etc.
    """

    id: str
    pattern_name: str
    motif_structure: str  # Pattern signature e.g. "PERSON-INFLUENCES->CONCEPT-APPEARS_IN->BOOK"
    frequency: int  # Total occurrences across all books
    cross_domain_count: int  # Number of different books containing pattern
    significance_score: float  # Statistical significance (0.0-1.0)
    discovered_at: datetime
    saved_by: Optional[str] = None  # User who saved pattern (if saved)
    description: str = ""

    @classmethod
    def create_new(
        cls,
        pattern_name: str,
        motif_structure: str,
        frequency: int,
        cross_domain_count: int,
        significance_score: float,
        description: str = "",
        saved_by: Optional[str] = None
    ) -> "OntologicalPattern":
        """Create a new OntologicalPattern"""
        return cls(
            id=f"pattern-{uuid.uuid4()}",
            pattern_name=pattern_name,
            motif_structure=motif_structure,
            frequency=frequency,
            cross_domain_count=cross_domain_count,
            significance_score=significance_score,
            discovered_at=datetime.now(),
            saved_by=saved_by,
            description=description
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage"""
        data = asdict(self)
        data['discovered_at'] = self.discovered_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OntologicalPattern":
        """Create OntologicalPattern from Neo4j node properties"""
        data = data.copy()
        if isinstance(data.get('discovered_at'), str):
            data['discovered_at'] = datetime.fromisoformat(data['discovered_at'])
        return cls(**data)

    def is_significant(self, min_frequency: int = 3, min_domains: int = 2) -> bool:
        """Check if pattern meets significance thresholds"""
        return (
            self.frequency >= min_frequency and
            self.cross_domain_count >= min_domains and
            self.significance_score >= 0.5
        )


@dataclass
class PatternInstance:
    """
    Represents a specific occurrence of an OntologicalPattern in a particular book.
    Links the abstract pattern to concrete entities and relationships.
    """

    id: str
    pattern_id: str  # Link to OntologicalPattern
    book_id: str  # Book where instance appears
    entity_ids: List[str]  # List of entity IDs participating in this instance
    subgraph_hash: str  # Hash of subgraph structure for deduplication
    context: str = ""  # Text context around pattern

    @classmethod
    def create_new(
        cls,
        pattern_id: str,
        book_id: str,
        entity_ids: List[str],
        subgraph_hash: str,
        context: str = ""
    ) -> "PatternInstance":
        """Create a new PatternInstance"""
        return cls(
            id=f"instance-{uuid.uuid4()}",
            pattern_id=pattern_id,
            book_id=book_id,
            entity_ids=entity_ids,
            subgraph_hash=subgraph_hash,
            context=context
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternInstance":
        """Create PatternInstance from Neo4j node properties"""
        return cls(**data)

    def get_entity_count(self) -> int:
        """Get number of entities in this pattern instance"""
        return len(self.entity_ids)
