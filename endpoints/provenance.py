"""
Provenance API Endpoints
Handles provenance chain retrieval for GraphRAG queries
Feature: 001-interactive-graphrag-refinement - User Story 1
"""

from flask import jsonify, request
import logging
from services.provenance_tracker import ProvenanceTracker
from services.neo4j_client import Neo4jClient
import asyncio

logger = logging.getLogger(__name__)


def register_provenance_endpoints(app):
    """Register provenance-related endpoints"""

    # Initialize Neo4j client (will be shared across requests)
    neo4j_client = None

    def get_neo4j_client():
        """Get or create Neo4j client instance"""
        nonlocal neo4j_client
        if neo4j_client is None:
            neo4j_client = Neo4jClient()
        return neo4j_client

    @app.route('/api/provenance/<query_id>', methods=['GET'])
    def get_provenance_chain(query_id):
        """
        GET /api/provenance/{query_id}

        Retrieve complete provenance chain for a query.

        Returns:
            200: Complete provenance chain with query, result, entities, books
            404: Query not found
            500: Server error
        """
        try:
            logger.info(f"📊 Retrieving provenance chain for query: {query_id}")

            # Create provenance tracker
            client = get_neo4j_client()
            tracker = ProvenanceTracker(client)

            # Build provenance chain (async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            provenance_chain = loop.run_until_complete(
                tracker.build_provenance_chain(query_id)
            )
            loop.close()

            if not provenance_chain:
                return jsonify({
                    'success': False,
                    'error': f'Query {query_id} not found'
                }), 404

            return jsonify({
                'success': True,
                'provenance_chain': provenance_chain
            }), 200

        except Exception as e:
            logger.error(f"Error retrieving provenance chain: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/provenance/<query_id>/entities', methods=['GET'])
    def get_query_entities(query_id):
        """
        GET /api/provenance/{query_id}/entities

        Get list of entities used in a query with rank and relevance scores.

        Query Parameters:
            limit (int): Maximum number of entities to return

        Returns:
            200: List of entities with metadata
            500: Server error
        """
        try:
            limit = request.args.get('limit', type=int)
            logger.info(f"📊 Retrieving entities for query: {query_id}")

            # Create provenance tracker
            client = get_neo4j_client()
            tracker = ProvenanceTracker(client)

            # Get entities (async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            entities = loop.run_until_complete(
                tracker.get_query_entities(query_id, limit)
            )
            loop.close()

            return jsonify({
                'success': True,
                'query_id': query_id,
                'entities': entities,
                'count': len(entities)
            }), 200

        except Exception as e:
            logger.error(f"Error retrieving entities: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/provenance/<query_id>/relationships', methods=['GET'])
    def get_query_relationships(query_id):
        """
        GET /api/provenance/{query_id}/relationships

        Get list of relationships traversed during query execution.

        Returns:
            200: List of traversed relationships with order and weights
            500: Server error
        """
        try:
            logger.info(f"📊 Retrieving relationships for query: {query_id}")

            # Create provenance tracker
            client = get_neo4j_client()
            tracker = ProvenanceTracker(client)

            # Get relationships (async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            relationships = loop.run_until_complete(
                tracker.get_query_relationships(query_id)
            )
            loop.close()

            return jsonify({
                'success': True,
                'query_id': query_id,
                'relationships': relationships,
                'count': len(relationships)
            }), 200

        except Exception as e:
            logger.error(f"Error retrieving relationships: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/api/provenance/<query_id>/chunks', methods=['GET'])
    def get_query_chunks(query_id):
        """
        GET /api/provenance/{query_id}/chunks

        Get list of source text chunks used in query.

        Note: Currently chunks are not stored separately in Neo4j.
        This endpoint is a placeholder for future implementation.

        Returns:
            200: List of source chunks (currently empty)
            500: Server error
        """
        try:
            logger.info(f"📊 Retrieving chunks for query: {query_id}")

            # Placeholder: Chunks not yet implemented
            # In the future, this would query Chunk nodes linked to entities
            chunks = []

            return jsonify({
                'success': True,
                'query_id': query_id,
                'chunks': chunks,
                'count': len(chunks),
                'note': 'Chunk storage not yet implemented - entities link directly to books'
            }), 200

        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    logger.info("✅ Provenance endpoints registered")
