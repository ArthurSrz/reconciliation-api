"""
GraphEdit node model for tracking manual graph modifications
Feature: 001-interactive-graphrag-refinement
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import uuid
import json


@dataclass
class GraphEdit:
    """
    Represents a manual edit to the knowledge graph.

    Constitutional Principles:
    - Principle #5: Full audit trail for interpretability
    - Tracks who changed what, when, and why for complete transparency

    Edit Types:
    - entity_add: New entity manually added by user
    - entity_delete: Entity removed (must not create orphans - Principle #1)
    - entity_modify: Entity properties changed
    - relationship_add: New relationship manually added
    - relationship_modify: Relationship type or properties changed
    - relationship_delete: Relationship removed (must not create orphans)
    """

    id: str
    edit_type: str
    target_id: str
    old_value: str  # JSON string of previous state
    new_value: str  # JSON string of new state
    justification: str
    editor_id: str
    timestamp: datetime
    applied: bool = True
    validation_status: str = "valid"  # "valid", "conflict", "rolled_back"

    @classmethod
    def create_new(
        cls,
        edit_type: str,
        target_id: str,
        old_value: Dict[str, Any],
        new_value: Dict[str, Any],
        justification: str,
        editor_id: str = "default_user"
    ) -> "GraphEdit":
        """Create a new GraphEdit"""
        return cls(
            id=f"edit-{uuid.uuid4()}",
            edit_type=edit_type,
            target_id=target_id,
            old_value=json.dumps(old_value),
            new_value=json.dumps(new_value),
            justification=justification,
            editor_id=editor_id,
            timestamp=datetime.now(),
            applied=True,
            validation_status="valid"
        )

    @classmethod
    def create_rollback(cls, original_edit: "GraphEdit") -> "GraphEdit":
        """Create a rollback edit that reverses an original edit"""
        return cls(
            id=f"edit-{uuid.uuid4()}",
            edit_type=original_edit.edit_type,
            target_id=original_edit.target_id,
            old_value=original_edit.new_value,  # Swap old/new
            new_value=original_edit.old_value,
            justification=f"Rollback of edit {original_edit.id}",
            editor_id=original_edit.editor_id,
            timestamp=datetime.now(),
            applied=True,
            validation_status="valid"
        )

    def get_old_value_dict(self) -> Dict[str, Any]:
        """Parse old_value JSON string to dictionary"""
        return json.loads(self.old_value)

    def get_new_value_dict(self) -> Dict[str, Any]:
        """Parse new_value JSON string to dictionary"""
        return json.loads(self.new_value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphEdit":
        """Create GraphEdit from Neo4j node properties"""
        data = data.copy()
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

    def get_summary(self) -> str:
        """Get human-readable summary of the edit"""
        old_val = self.get_old_value_dict()
        new_val = self.get_new_value_dict()

        if self.edit_type == "relationship_modify":
            old_type = old_val.get('type', 'UNKNOWN')
            new_type = new_val.get('type', 'UNKNOWN')
            return f"Changed relationship type from {old_type} to {new_type}"
        elif self.edit_type == "relationship_add":
            rel_type = new_val.get('type', 'UNKNOWN')
            source = new_val.get('source_name', 'unknown')
            target = new_val.get('target_name', 'unknown')
            return f"Added {rel_type} relationship: {source} → {target}"
        elif self.edit_type == "entity_add":
            entity_name = new_val.get('name', 'unknown')
            entity_type = new_val.get('entity_type', 'unknown')
            return f"Added {entity_type} entity: {entity_name}"
        elif self.edit_type == "entity_modify":
            entity_name = new_val.get('name', old_val.get('name', 'unknown'))
            return f"Modified entity: {entity_name}"
        else:
            return f"{self.edit_type} on {self.target_id}"
