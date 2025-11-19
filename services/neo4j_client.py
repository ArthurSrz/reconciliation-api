"""
Neo4j client extension for provenance, edits, and pattern queries
Feature: 001-interactive-graphrag-refinement
"""

from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from datetime import datetime
import os


class Neo4jClient:
    """
    Extended Neo4j client with query execution methods for:
    - Provenance tracking (Query → QueryResult → Entities → Books)
    - Graph edits (GraphEdit nodes and relationship modifications)
    - Pattern discovery (OntologicalPattern and PatternInstance)

    Uses async Neo4j driver for non-blocking operations.
    """

    def __init__(self, uri: Optional[str] = None, auth: Optional[tuple] = None):
        """
        Initialize Neo4j client with connection parameters.

        Args:
            uri: Neo4j connection URI (defaults to env NEO4J_URI)
            auth: Tuple of (username, password) (defaults to env NEO4J_USER/NEO4J_PASSWORD)
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        username = auth[0] if auth else os.getenv('NEO4J_USER', 'neo4j')
        password = auth[1] if auth else os.getenv('NEO4J_PASSWORD', 'password')

        self.driver: AsyncDriver = AsyncGraphDatabase.driver(
            self.uri,
            auth=(username, password),
            max_connection_pool_size=50
        )

    async def verify_connection(self) -> bool:
        """Verify Neo4j connection is working"""
        try:
            await self.driver.verify_authentication()
            await self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False

    async def close(self):
        """Close the Neo4j driver connection"""
        await self.driver.close()

    # ========== Provenance Query Methods ==========

    async def get_provenance_chain(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete provenance chain for a query:
        Query → QueryResult → Used Entities → Books

        Returns:
            Dict with query, result, entities, relationships, books
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (q:Query {id: $query_id})-[:PRODUCED_RESULT]->(qr:QueryResult)
                OPTIONAL MATCH (qr)-[ue:USED_ENTITY]->(e:Entity)
                OPTIONAL MATCH (b:BOOK)-[:CONTAINS_ENTITY]->(e)
                OPTIONAL MATCH (qr)-[:CITED_COMMUNITY]->(comm:Community)
                RETURN q, qr,
                       collect(DISTINCT {
                         entity: e,
                         rank: ue.rank,
                         relevance_score: ue.relevance_score
                       }) as entities,
                       collect(DISTINCT {
                         book: b,
                         book_id: b.id,
                         title: b.title,
                         author: b.author
                       }) as books,
                       collect(DISTINCT {
                         community: comm,
                         community_id: comm.id,
                         title: comm.title
                       }) as communities
                """,
                query_id=query_id
            )
            record = await result.single()
            if not record:
                return None

            return {
                'query': dict(record['q']),
                'result': dict(record['qr']),
                'entities': [e for e in record['entities'] if e['entity'] is not None],
                'books': [b for b in record['books'] if b['book'] is not None],
                'communities': [c for c in record['communities'] if c['community'] is not None]
            }

    async def get_used_entities(self, query_id: str) -> List[Dict[str, Any]]:
        """Get all entities used in a query with rank and relevance scores"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (q:Query {id: $query_id})-[:PRODUCED_RESULT]->(qr:QueryResult)
                MATCH (qr)-[ue:USED_ENTITY]->(e:Entity)
                RETURN e.id as entity_id,
                       e.name as entity_name,
                       e.entity_type as entity_type,
                       e.description as description,
                       ue.rank as rank,
                       ue.relevance_score as relevance_score,
                       ue.contribution as contribution
                ORDER BY ue.rank
                """,
                query_id=query_id
            )
            return [dict(record) async for record in result]

    async def get_traversed_relationships(self, query_id: str) -> List[Dict[str, Any]]:
        """Get all relationships traversed during query execution"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (q:Query {id: $query_id})-[:PRODUCED_RESULT]->(qr:QueryResult)
                MATCH (qr)-[tr:TRAVERSED_RELATIONSHIP]->(source:Entity)
                MATCH (source)-[r:RELATED_TO]->(target:Entity)
                RETURN source.id as source_id,
                       source.name as source_name,
                       target.id as target_id,
                       target.name as target_name,
                       type(r) as relationship_type,
                       r.description as description,
                       tr.order as order,
                       tr.weight as weight,
                       tr.hop_distance as hop_distance
                ORDER BY tr.order
                """,
                query_id=query_id
            )
            return [dict(record) async for record in result]

    # ========== Edit Query Methods ==========

    async def apply_relationship_edit(
        self,
        source_id: str,
        target_id: str,
        old_type: str,
        new_type: str,
        properties: Dict[str, Any],
        edit_id: str
    ) -> bool:
        """
        Apply a relationship edit: change relationship type and/or properties.

        Returns:
            True if successful, False otherwise
        """
        async with self.driver.session() as session:
            try:
                # Transaction: delete old relationship, create new one
                await session.run(
                    """
                    MATCH (source:Entity {id: $source_id})-[r:RELATED_TO]->(target:Entity {id: $target_id})
                    WHERE type(r) = $old_type OR $old_type = 'RELATED_TO'
                    CREATE (source)-[new_r:RELATED_TO]->(target)
                    SET new_r = $properties
                    SET new_r.manual_flag = true
                    SET new_r.edit_id = $edit_id
                    SET new_r.edited_at = datetime()
                    DELETE r
                    """,
                    source_id=source_id,
                    target_id=target_id,
                    old_type=old_type,
                    properties=properties,
                    edit_id=edit_id
                )
                return True
            except Exception as e:
                print(f"Error applying relationship edit: {e}")
                return False

    async def get_edit_history(self, target_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get edit history for an entity or relationship"""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (edit:GraphEdit {target_id: $target_id})
                WHERE edit.applied = true
                RETURN edit
                ORDER BY edit.timestamp DESC
                LIMIT $limit
                """,
                target_id=target_id,
                limit=limit
            )
            return [dict(record['edit']) async for record in result]

    # ========== Pattern Query Methods ==========

    async def discover_patterns(
        self,
        min_frequency: int = 3,
        min_cross_domain: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Discover recurring 2-hop patterns across multiple books.

        Constitutional Principle #3: Prioritize cross-book investigation zones.

        Returns:
            List of patterns with frequency and cross-domain counts
        """
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e1:Entity)-[r1:RELATED_TO]->(e2:Entity)-[r2:RELATED_TO]->(e3:Entity)
                WITH e1.entity_type as t1, type(r1) as rel1,
                     e2.entity_type as t2, type(r2) as rel2,
                     e3.entity_type as t3,
                     collect(DISTINCT e1.book_dir) as books
                WHERE size(books) >= $min_cross_domain
                WITH t1, rel1, t2, rel2, t3, books, size(books) as book_count
                WHERE book_count >= $min_cross_domain
                RETURN t1 + '-' + rel1 + '->' + t2 + '-' + rel2 + '->' + t3 as pattern_signature,
                       t1, rel1, t2, rel2, t3,
                       book_count as cross_domain_count,
                       books
                ORDER BY book_count DESC
                LIMIT 20
                """,
                min_cross_domain=min_cross_domain
            )
            return [dict(record) async for record in result]

    async def save_pattern(
        self,
        pattern_id: str,
        pattern_name: str,
        motif_structure: str,
        frequency: int,
        cross_domain_count: int,
        significance_score: float,
        saved_by: str
    ) -> bool:
        """Save a discovered pattern to the database"""
        async with self.driver.session() as session:
            try:
                await session.run(
                    """
                    CREATE (p:OntologicalPattern {
                        id: $pattern_id,
                        pattern_name: $pattern_name,
                        motif_structure: $motif_structure,
                        frequency: $frequency,
                        cross_domain_count: $cross_domain_count,
                        significance_score: $significance_score,
                        discovered_at: datetime(),
                        saved_by: $saved_by
                    })
                    """,
                    pattern_id=pattern_id,
                    pattern_name=pattern_name,
                    motif_structure=motif_structure,
                    frequency=frequency,
                    cross_domain_count=cross_domain_count,
                    significance_score=significance_score,
                    saved_by=saved_by
                )
                return True
            except Exception as e:
                print(f"Error saving pattern: {e}")
                return False

    # ========== Utility Methods ==========

    async def create_query_node(self, query_data: Dict[str, Any]) -> bool:
        """Create a Query node in Neo4j"""
        async with self.driver.session() as session:
            try:
                await session.run(
                    """
                    CREATE (q:Query {
                        id: $id,
                        question: $question,
                        answer_text: $answer_text,
                        timestamp: datetime($timestamp),
                        version: $version,
                        parent_query_id: $parent_query_id,
                        mode: $mode,
                        user_id: $user_id,
                        status: $status,
                        book_context: $book_context
                    })
                    """,
                    **query_data
                )
                return True
            except Exception as e:
                print(f"Error creating query node: {e}")
                return False

    async def create_query_result_node(self, result_data: Dict[str, Any]) -> bool:
        """Create a QueryResult node and link to Query"""
        async with self.driver.session() as session:
            try:
                await session.run(
                    """
                    CREATE (qr:QueryResult {
                        id: $id,
                        query_id: $query_id,
                        timestamp: datetime($timestamp),
                        execution_time_ms: $execution_time_ms,
                        entity_count: $entity_count,
                        relationship_count: $relationship_count,
                        community_count: $community_count
                    })
                    WITH qr
                    MATCH (q:Query {id: $query_id})
                    CREATE (q)-[:PRODUCED_RESULT]->(qr)
                    """,
                    **result_data
                )
                return True
            except Exception as e:
                print(f"Error creating query result node: {e}")
                return False
