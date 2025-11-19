"""
Data models for Interactive GraphRAG Refinement System
Feature: 001-interactive-graphrag-refinement
"""

from .query import Query, QueryResult
from .graph_edit import GraphEdit
from .ontological_pattern import OntologicalPattern, PatternInstance

__all__ = [
    'Query',
    'QueryResult',
    'GraphEdit',
    'OntologicalPattern',
    'PatternInstance',
]
