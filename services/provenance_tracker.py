"""
Provenance Tracker Service
Builds complete provenance chains from Query → QueryResult → Entities → Books
Feature: 001-interactive-graphrag-refinement - User Story 1
"""

import logging
from typing import Dict, Any, List, Optional
from models.query import Query, QueryResult
from services.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class ProvenanceTracker:
    """
    Tracks and retrieves complete provenance chains for GraphRAG queries.

    Constitutional Principle #5: End-to-end interpretability
    - Enables navigation from answer → entities → relationships → chunks → books
    """

    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize provenance tracker with Neo4j client.

        Args:
            neo4j_client: Neo4jClient instance for graph queries
        """
        self.neo4j_client = neo4j_client

    async def build_provenance_chain(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Build complete provenance chain for a query.

        Retrieves:
        - Query node with question and answer
        - QueryResult node with execution metadata
        - Used entities with rank and relevance scores
        - Traversed relationships during query
        - Source books containing entities
        - Cited communities (if applicable)

        Args:
            query_id: Query node ID

        Returns:
            Dict with complete provenance chain or None if query not found
        """
        try:
            # Get provenance chain from Neo4j
            chain_data = await self.neo4j_client.get_provenance_chain(query_id)

            if not chain_data:
                logger.warning(f"No provenance chain found for query {query_id}")
                return None

            # Format for API response
            provenance_chain = {
                'query': {
                    'query_id': chain_data['query'].get('id'),
                    'question': chain_data['query'].get('question'),
                    'answer_text': chain_data['query'].get('answer_text'),
                    'timestamp': str(chain_data['query'].get('timestamp')),
                    'version': chain_data['query'].get('version', 1),
                    'mode': chain_data['query'].get('mode', 'local'),
                    'status': chain_data['query'].get('status', 'active')
                },
                'result': {
                    'result_id': chain_data['result'].get('id'),
                    'query_id': chain_data['result'].get('query_id'),
                    'timestamp': str(chain_data['result'].get('timestamp')),
                    'execution_time_ms': chain_data['result'].get('execution_time_ms', 0),
                    'entity_count': chain_data['result'].get('entity_count', 0),
                    'relationship_count': chain_data['result'].get('relationship_count', 0),
                    'community_count': chain_data['result'].get('community_count', 0)
                },
                'entities': self._format_entities(chain_data['entities']),
                'books': self._format_books(chain_data['books']),
                'communities': self._format_communities(chain_data.get('communities', []))
            }

            logger.info(
                f"Built provenance chain for query {query_id}: "
                f"{len(provenance_chain['entities'])} entities, "
                f"{len(provenance_chain['books'])} books"
            )

            return provenance_chain

        except Exception as e:
            logger.error(f"Error building provenance chain for query {query_id}: {e}")
            return None

    async def get_query_entities(
        self,
        query_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of entities used in a query with rank and relevance scores.

        Args:
            query_id: Query node ID
            limit: Maximum number of entities to return

        Returns:
            List of entities with metadata
        """
        try:
            entities = await self.neo4j_client.get_used_entities(query_id)

            if limit:
                entities = entities[:limit]

            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    'entity_id': entity.get('entity_id'),
                    'entity_name': entity.get('entity_name'),
                    'entity_type': entity.get('entity_type'),
                    'description': entity.get('description', ''),
                    'rank': entity.get('rank', 0),
                    'relevance_score': entity.get('relevance_score', 0.0),
                    'contribution': entity.get('contribution', 'context')
                })

            return formatted_entities

        except Exception as e:
            logger.error(f"Error getting entities for query {query_id}: {e}")
            return []

    async def get_query_relationships(self, query_id: str) -> List[Dict[str, Any]]:
        """
        Get list of relationships traversed during query execution.

        Args:
            query_id: Query node ID

        Returns:
            List of relationships with source, target, and traversal metadata
        """
        try:
            relationships = await self.neo4j_client.get_traversed_relationships(query_id)

            formatted_rels = []
            for rel in relationships:
                formatted_rels.append({
                    'source_id': rel.get('source_id'),
                    'source_name': rel.get('source_name'),
                    'target_id': rel.get('target_id'),
                    'target_name': rel.get('target_name'),
                    'relationship_type': rel.get('relationship_type', 'RELATED_TO'),
                    'description': rel.get('description', ''),
                    'order': rel.get('order', 0),
                    'weight': rel.get('weight', 1.0),
                    'hop_distance': rel.get('hop_distance', 1)
                })

            return formatted_rels

        except Exception as e:
            logger.error(f"Error getting relationships for query {query_id}: {e}")
            return []

    def _format_entities(self, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Format entities for API response"""
        formatted = []
        for entity_data in entities:
            entity = entity_data.get('entity')
            if not entity:
                continue

            formatted.append({
                'entity_id': entity.get('id'),
                'entity_name': entity.get('name'),
                'entity_type': entity.get('entity_type'),
                'description': entity.get('description', ''),
                'rank': entity_data.get('rank', 0),
                'relevance_score': entity_data.get('relevance_score', 0.0),
                'book_id': entity_data.get('book_id'),
                'book_title': entity_data.get('book_title'),
                'book_author': entity_data.get('book_author')
            })

        return formatted

    def _format_books(self, books: List[Dict]) -> List[Dict[str, Any]]:
        """Format books for API response"""
        formatted = []
        seen_books = set()

        for book_data in books:
            book = book_data.get('book')
            if not book:
                continue

            book_id = book.get('id')
            if book_id in seen_books:
                continue

            seen_books.add(book_id)
            formatted.append({
                'book_id': book_id,
                'title': book.get('title', book.get('name', '')),
                'author': book.get('author', '')
            })

        return formatted

    def _format_communities(self, communities: List[Dict]) -> List[Dict[str, Any]]:
        """Format communities for API response"""
        formatted = []
        for comm_data in communities:
            community = comm_data.get('community')
            if not community:
                continue

            formatted.append({
                'community_id': community.get('id'),
                'title': community.get('title', ''),
                'summary': community.get('report', ''),
                'relevance_score': comm_data.get('relevance_score', 0.0)
            })

        return formatted
