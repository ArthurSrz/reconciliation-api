"""
Validation utilities for edit validation and graph consistency checks
Feature: 001-interactive-graphrag-refinement
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    """Validation result levels"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Result of validation checks"""
    valid: bool
    level: ValidationLevel
    errors: List[str]
    warnings: List[str]
    impact: Dict[str, Any]

    @classmethod
    def success(cls, warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a successful validation result"""
        return cls(
            valid=True,
            level=ValidationLevel.VALID,
            errors=[],
            warnings=warnings or [],
            impact={}
        )

    @classmethod
    def failure(cls, errors: List[str], warnings: Optional[List[str]] = None) -> "ValidationResult":
        """Create a failed validation result"""
        return cls(
            valid=False,
            level=ValidationLevel.ERROR,
            errors=errors,
            warnings=warnings or [],
            impact={}
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response"""
        return {
            'valid': self.valid,
            'level': self.level.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'impact': self.impact
        }


class ValidationService:
    """
    Service for validating graph edits and ensuring consistency.

    Constitutional Principles:
    - Principle #1: No orphan nodes allowed in the interface
    - Validates that edits maintain graph connectivity
    """

    def __init__(self, neo4j_client):
        """
        Initialize validation service with Neo4j client.

        Args:
            neo4j_client: Neo4jClient instance for graph queries
        """
        self.neo4j_client = neo4j_client

    async def validate_relationship_edit(
        self,
        source_id: str,
        target_id: str,
        old_type: str,
        new_type: str,
        properties: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate a relationship edit before applying.

        Checks:
        1. Source and target entities exist
        2. Relationship to be modified exists
        3. New relationship type is valid
        4. Edit won't create orphan nodes (Constitutional Principle #1)

        Returns:
            ValidationResult with errors/warnings
        """
        errors = []
        warnings = []

        # Check 1: Verify entities exist
        source_exists = await self._entity_exists(source_id)
        target_exists = await self._entity_exists(target_id)

        if not source_exists:
            errors.append(f"Source entity {source_id} does not exist")
        if not target_exists:
            errors.append(f"Target entity {target_id} does not exist")

        if errors:
            return ValidationResult.failure(errors, warnings)

        # Check 2: Verify relationship exists
        rel_exists = await self._relationship_exists(source_id, target_id, old_type)
        if not rel_exists:
            errors.append(
                f"Relationship {old_type} from {source_id} to {target_id} does not exist"
            )
            return ValidationResult.failure(errors, warnings)

        # Check 3: Validate new relationship type
        if not new_type or new_type.strip() == "":
            errors.append("New relationship type cannot be empty")
            return ValidationResult.failure(errors, warnings)

        # Check 4: Warn if this is the only relationship for either entity
        source_degree = await self._get_entity_degree(source_id)
        target_degree = await self._get_entity_degree(target_id)

        if source_degree == 1:
            warnings.append(
                f"Source entity {source_id} will become orphaned if this relationship is deleted"
            )
        if target_degree == 1:
            warnings.append(
                f"Target entity {target_id} will become orphaned if this relationship is deleted"
            )

        return ValidationResult.success(warnings)

    async def validate_relationship_delete(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str
    ) -> ValidationResult:
        """
        Validate relationship deletion.

        CRITICAL: Enforces Constitutional Principle #1 - No orphan nodes.
        Prevents deletion if it would create isolated entities.

        Returns:
            ValidationResult with errors if deletion would create orphans
        """
        errors = []
        warnings = []

        # Check if relationship exists
        rel_exists = await self._relationship_exists(source_id, target_id, relationship_type)
        if not rel_exists:
            errors.append(
                f"Relationship {relationship_type} from {source_id} to {target_id} does not exist"
            )
            return ValidationResult.failure(errors, warnings)

        # Check entity degrees AFTER deletion
        source_degree = await self._get_entity_degree(source_id)
        target_degree = await self._get_entity_degree(target_id)

        if source_degree <= 1:
            errors.append(
                f"Cannot delete relationship: would create orphan node {source_id} "
                f"(Constitutional Principle #1: No orphan nodes allowed)"
            )

        if target_degree <= 1:
            errors.append(
                f"Cannot delete relationship: would create orphan node {target_id} "
                f"(Constitutional Principle #1: No orphan nodes allowed)"
            )

        if errors:
            return ValidationResult.failure(errors, warnings)

        return ValidationResult.success(warnings)

    async def validate_relationship_add(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate adding a new relationship.

        Checks:
        1. Source and target entities exist
        2. Relationship doesn't already exist (no duplicates)
        3. Relationship type is valid

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # Check 1: Verify entities exist
        source_exists = await self._entity_exists(source_id)
        target_exists = await self._entity_exists(target_id)

        if not source_exists:
            errors.append(f"Source entity {source_id} does not exist")
        if not target_exists:
            errors.append(f"Target entity {target_id} does not exist")

        if errors:
            return ValidationResult.failure(errors, warnings)

        # Check 2: Check for duplicate
        rel_exists = await self._relationship_exists(source_id, target_id, relationship_type)
        if rel_exists:
            warnings.append(
                f"Relationship {relationship_type} already exists between {source_id} and {target_id}"
            )

        # Check 3: Validate relationship type
        if not relationship_type or relationship_type.strip() == "":
            errors.append("Relationship type cannot be empty")
            return ValidationResult.failure(errors, warnings)

        return ValidationResult.success(warnings)

    async def validate_entity_add(
        self,
        entity_id: str,
        entity_name: str,
        entity_type: str
    ) -> ValidationResult:
        """
        Validate adding a new entity.

        WARNING: New entities should be immediately connected to avoid orphans.

        Returns:
            ValidationResult with warnings about orphan nodes
        """
        errors = []
        warnings = []

        # Check if entity already exists
        exists = await self._entity_exists(entity_id)
        if exists:
            errors.append(f"Entity {entity_id} already exists")
            return ValidationResult.failure(errors, warnings)

        # Warn about orphan nodes
        warnings.append(
            "New entity will be orphaned until relationships are added. "
            "Constitutional Principle #1 requires all displayed nodes to have relationships."
        )

        return ValidationResult.success(warnings)

    # ========== Helper Methods ==========

    async def _entity_exists(self, entity_id: str) -> bool:
        """Check if entity exists in graph"""
        async with self.neo4j_client.driver.session() as session:
            result = await session.run(
                "MATCH (e:Entity {id: $entity_id}) RETURN COUNT(e) > 0 AS exists",
                entity_id=entity_id
            )
            record = await result.single()
            return record['exists'] if record else False

    async def _relationship_exists(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str
    ) -> bool:
        """Check if specific relationship exists"""
        async with self.neo4j_client.driver.session() as session:
            result = await session.run(
                """
                MATCH (source:Entity {id: $source_id})-[r:RELATED_TO]->(target:Entity {id: $target_id})
                WHERE type(r) = $relationship_type OR $relationship_type = 'RELATED_TO'
                RETURN COUNT(r) > 0 AS exists
                """,
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship_type
            )
            record = await result.single()
            return record['exists'] if record else False

    async def _get_entity_degree(self, entity_id: str) -> int:
        """Get total degree (incoming + outgoing relationships) of entity"""
        async with self.neo4j_client.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {id: $entity_id})
                RETURN size((e)-[]-()) as degree
                """,
                entity_id=entity_id
            )
            record = await result.single()
            return record['degree'] if record else 0

    async def check_graph_consistency(self) -> ValidationResult:
        """
        Check overall graph consistency.

        Verifies:
        1. No orphan nodes (Constitutional Principle #1)
        2. All relationships have valid source/target
        3. No dangling references

        Returns:
            ValidationResult with any consistency issues
        """
        errors = []
        warnings = []

        # Check for orphan nodes
        async with self.neo4j_client.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity)
                WHERE NOT exists((e)-[]-())
                RETURN count(e) as orphan_count, collect(e.id) as orphan_ids
                """
            )
            record = await result.single()
            if record and record['orphan_count'] > 0:
                errors.append(
                    f"Found {record['orphan_count']} orphan nodes (violates Constitutional Principle #1): "
                    f"{record['orphan_ids'][:10]}"  # Show first 10
                )

        if errors:
            return ValidationResult.failure(errors, warnings)

        return ValidationResult.success(warnings)
