"""
Reconciliation API - Harmonizes Neo4j graph data with GraphRAG queries
This API serves as the central coordination layer between:
- Neo4j (source of truth for graph structure)
- GraphRAG (for intelligent queries on visible nodes)
- Frontend (Vercel deployment)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase
import os
import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re
import csv
from nano_graphrag import GraphRAG, QueryParam
from io import StringIO
from functools import wraps
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS for Vercel and local development
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:3001",
    "https://borges-library-web.vercel.app",
    "https://borges-library*.vercel.app",
    "https://*.vercel.app"
], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type'])

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
GRAPHRAG_API_URL = os.getenv('GRAPHRAG_API_URL', 'https://borgesgraph-production.up.railway.app')

# Neo4j driver instance
neo4j_driver = None

def get_neo4j_driver():
    """Get or create Neo4j driver instance"""
    global neo4j_driver
    if neo4j_driver is None and NEO4J_URI:
        try:
            neo4j_driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            logger.info(f"Connected to Neo4j: {NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            return None
    return neo4j_driver

def close_neo4j_driver():
    """Close Neo4j driver"""
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.close()
        neo4j_driver = None

# Local GraphRAG Configuration with dynamic data loading
GRAPHRAG_WORKING_DIR = os.getenv('GRAPHRAG_WORKING_DIR', None)  # Will be set dynamically
local_graphrag = None
gdrive_data_path = None

def get_local_graphrag(book_id: str = "a_rebours_huysmans"):
    """Get or create local GraphRAG instance with real interceptor and GDrive data"""
    global local_graphrag, gdrive_data_path

    # Always ensure we have fresh data for the requested book
    try:
        # Ensure data is available from GDrive
        logger.info(f"📥 Ensuring GraphRAG data is available for book: {book_id}")
        base_data_path = gdrive_manager.ensure_data_available()
        book_data_path = gdrive_manager.get_book_data_path(book_id)

        if not book_data_path:
            available_books = gdrive_manager.list_available_books()
            logger.error(f"❌ Book {book_id} not found. Available: {available_books}")
            return None

        # Only recreate GraphRAG if path changed or doesn't exist
        if local_graphrag is None or gdrive_data_path != book_data_path:
            from nano_graphrag._llm import gpt_4o_mini_complete

            # Créer l'intercepteur LLM comme dans test_query_analysis.py
            intercepted_llm = graphrag_interceptor.intercept_query_processing(gpt_4o_mini_complete)

            local_graphrag = GraphRAG(
                working_dir=book_data_path,
                best_model_func=intercepted_llm,
                cheap_model_func=intercepted_llm,
                embedding_func_max_async=4,
                best_model_max_async=2,
                cheap_model_max_async=4,
                embedding_batch_num=16,
                graph_cluster_algorithm="leiden"
            )

            gdrive_data_path = book_data_path
            logger.info(f"✅ Local GraphRAG initialized with GDRIVE data from: {book_data_path}")

        return local_graphrag

    except Exception as e:
        logger.error(f"❌ Failed to initialize local GraphRAG with GDrive data: {e}")
        return None

# Import du nouvel intercepteur et du gestionnaire de données
from graphrag_interceptor import graphrag_interceptor
from gdrive_data_manager import gdrive_manager
from endpoints.books import register_books_endpoints

# GraphRAG Debug Interceptor (remplacé par le vrai intercepteur)
class GraphRAGDebugInterceptor:
    """
    Debug interceptor that captures GraphRAG processing phases
    Inspired by the test files to show entity/community selection
    """

    def __init__(self):
        self.debug_data = {}
        self.processing_phases = []

    def capture_debug_info(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract debug information from GraphRAG response
        This simulates the interceptor from test_single_query.py
        """
        debug_info = {
            "processing_phases": {
                "entity_selection": {
                    "entities": [],
                    "duration_ms": 150,
                    "phase": "explosion"
                },
                "community_analysis": {
                    "communities": [],
                    "duration_ms": 300,
                    "phase": "filtering"
                },
                "relationship_mapping": {
                    "relationships": [],
                    "duration_ms": 200,
                    "phase": "synthesis"
                },
                "text_synthesis": {
                    "sources": [],
                    "duration_ms": 250,
                    "phase": "crystallization"
                }
            },
            "context_stats": {
                "total_time_ms": 900,
                "mode": response_data.get("mode", "local"),
                "prompt_length": 0
            },
            "animation_timeline": [
                {"phase": "explosion", "duration": 2000, "description": "Analyzing all entities and communities"},
                {"phase": "filtering", "duration": 3000, "description": "Selecting relevant knowledge"},
                {"phase": "synthesis", "duration": 2000, "description": "Synthesizing information"},
                {"phase": "crystallization", "duration": 1000, "description": "Generating answer"}
            ]
        }

        # Parse searchPath if available
        search_path = response_data.get('searchPath', {})

        if 'entities' in search_path:
            debug_info["processing_phases"]["entity_selection"]["entities"] = [
                {
                    "id": entity.get("id", ""),
                    "name": entity.get("name", entity.get("id", "")),
                    "type": entity.get("type", "ENTITY"),
                    "description": entity.get("description", ""),
                    "rank": entity.get("rank", 0),
                    "score": entity.get("score", 0),
                    "selected": True
                }
                for entity in search_path["entities"][:20]  # Limit to 20 like test
            ]

        if 'communities' in search_path:
            debug_info["processing_phases"]["community_analysis"]["communities"] = [
                {
                    "id": comm.get("id", ""),
                    "title": comm.get("title", f"Community {comm.get('id', '')}"),
                    "content": comm.get("content", ""),
                    "relevance": comm.get("relevance", 0),
                    "impact_rating": comm.get("rating", 0)
                }
                for comm in search_path["communities"][:4]  # Limit to 4 like test
            ]

        if 'relations' in search_path:
            debug_info["processing_phases"]["relationship_mapping"]["relationships"] = [
                {
                    "source": rel.get("source", ""),
                    "target": rel.get("target", ""),
                    "description": rel.get("description", ""),
                    "weight": rel.get("weight", 0),
                    "rank": rel.get("rank", 0),
                    "traversal_order": rel.get("traversalOrder", i)
                }
                for i, rel in enumerate(search_path["relations"][:53])  # Limit to 53 like test
            ]

        # Add text sources simulation
        debug_info["processing_phases"]["text_synthesis"]["sources"] = [
            {
                "id": f"source_{i}",
                "content": f"Text chunk {i} content preview...",
                "relevance": 0.9 - (i * 0.1)
            }
            for i in range(3)  # Simulate 3 text sources like test
        ]

        return debug_info

# Create global debug interceptor instance
debug_interceptor = GraphRAGDebugInterceptor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Reconciliation API",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": {
            "neo4j": check_neo4j_connection(),
            "graphrag": check_graphrag_connection()
        }
    })

def check_neo4j_connection():
    """Check Neo4j connection status"""
    try:
        # Skip connection check if no credentials provided
        if not NEO4J_URI or NEO4J_URI == 'bolt://localhost:7687':
            return "not_configured"

        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("RETURN 1")
            result.single()
        return "connected"
    except Exception as e:
        logger.error(f"Neo4j connection error: {e}")
        return f"error: {str(e)}"

def check_graphrag_connection():
    """Check GraphRAG API connection status"""
    try:
        response = httpx.get(f"{GRAPHRAG_API_URL}/health", timeout=5.0)
        return "connected" if response.status_code == 200 else f"error: status {response.status_code}"
    except Exception as e:
        logger.error(f"GraphRAG connection error: {e}")
        return f"error: {str(e)}"

@app.route('/graph/nodes', methods=['GET'])
def get_graph_nodes():
    """
    Get nodes from Neo4j with progressive loading
    Query params:
    - limit: number of nodes to return (default 300, max 1000)
    - centrality_type: 'degree', 'betweenness', 'eigenvector' (default 'degree')
    """
    limit = min(int(request.args.get('limit', 300)), 1000)
    centrality_type = request.args.get('centrality_type', 'degree')

    # Check if Neo4j is configured
    if not NEO4J_URI or NEO4J_URI == 'bolt://localhost:7687':
        return jsonify({
            'success': False,
            'error': 'Neo4j not configured',
            'nodes': [],
            'count': 0,
            'limit': limit
        })

    try:
        driver = get_neo4j_driver()
        if not driver:
            return jsonify({
                'success': False,
                'error': 'Neo4j driver not available',
                'nodes': [],
                'count': 0,
                'limit': limit
            })

        with driver.session() as session:
            # Query to get most central/connected nodes
            if centrality_type == 'degree':
                query = """
                MATCH (n)
                WITH n, SIZE([(n)--() | 1]) as degree
                ORDER BY degree DESC
                LIMIT $limit
                RETURN n, degree
                """
            else:
                # For other centrality measures, use default degree for now
                query = """
                MATCH (n)
                WITH n, SIZE([(n)--() | 1]) as degree
                ORDER BY degree DESC
                LIMIT $limit
                RETURN n, degree
                """

            result = session.run(query, limit=limit)
            nodes = []

            for record in result:
                node = record['n']
                degree = record['degree']

                # Convert node properties, handling special types
                properties = {}
                for key, value in dict(node).items():
                    if hasattr(value, 'isoformat'):  # DateTime objects
                        properties[key] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        properties[key] = value
                    else:
                        properties[key] = str(value) if value is not None else None

                nodes.append({
                    'id': node.element_id,
                    'labels': list(node.labels),
                    'properties': properties,
                    'degree': degree,
                    'centrality_score': degree  # Will be replaced with actual centrality
                })

            return jsonify({
                'success': True,
                'nodes': nodes,
                'count': len(nodes),
                'limit': limit
            })

    except Exception as e:
        logger.error(f"Error fetching nodes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/graph/relationships', methods=['GET'])
def get_graph_relationships():
    """
    Get relationships for the displayed nodes
    Query params:
    - node_ids: comma-separated list of node IDs
    - limit: max number of relationships to return (default 10000)
    """
    node_ids = request.args.get('node_ids', '').split(',')
    limit = min(int(request.args.get('limit', 10000)), 50000)  # Cap at 50k relationships

    if not node_ids or node_ids == ['']:
        return jsonify({
            'success': False,
            'error': 'No node IDs provided'
        }), 400

    # Filter out empty strings from node_ids
    node_ids = [node_id.strip() for node_id in node_ids if node_id.strip()]

    if len(node_ids) == 0:
        return jsonify({
            'success': False,
            'error': 'No valid node IDs provided'
        }), 400

    logger.info(f"Fetching relationships for {len(node_ids)} nodes with limit {limit}")

    try:
        driver = get_neo4j_driver()
        if driver is None:
            logger.warning("Neo4j not available, returning empty relationships for testing")
            return jsonify({
                'success': True,
                'relationships': [],
                'count': 0,
                'input_nodes': len(node_ids),
                'limit_applied': limit,
                'filtered': False
            })
        with driver.session() as session:
            query = """
            MATCH (n)-[r]-(m)
            WHERE elementId(n) IN $node_ids AND elementId(m) IN $node_ids
            RETURN DISTINCT r, n, m
            LIMIT $limit
            """

            result = session.run(query, node_ids=node_ids, limit=limit)
            relationships = []

            for record in result:
                rel = record['r']

                # Convert relationship properties, handling special types
                rel_properties = {}
                for key, value in dict(rel).items():
                    if hasattr(value, 'isoformat'):  # DateTime objects
                        rel_properties[key] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        rel_properties[key] = value
                    else:
                        rel_properties[key] = str(value) if value is not None else None

                relationships.append({
                    'id': rel.element_id,
                    'type': rel.type,
                    'source': record['n'].element_id,
                    'target': record['m'].element_id,
                    'properties': rel_properties
                })

            logger.info(f"Successfully fetched {len(relationships)} relationships for {len(node_ids)} nodes")

            return jsonify({
                'success': True,
                'relationships': relationships,
                'count': len(relationships),
                'input_nodes': len(node_ids),
                'limit_applied': limit,
                'filtered': len(relationships) >= limit
            })

    except Exception as e:
        logger.error(f"Error fetching relationships: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/query/local', methods=['POST'])
def query_local_graphrag():
    """
    Endpoint pour tester le GraphRAG local avec vrai intercepteur et données GDrive
    Comme dans test_query_analysis.py
    """
    data = request.json
    query = data.get('query', '')
    mode = data.get('mode', 'local')
    debug_mode = data.get('debug_mode', True)
    book_id = data.get('book_id', 'a_rebours_huysmans')

    if not query:
        return jsonify({'success': False, 'error': 'Query is required'}), 400

    try:
        # Utiliser le GraphRAG local avec intercepteur et données GDrive
        graphrag_instance = get_local_graphrag(book_id)
        if not graphrag_instance:
            available_books = gdrive_manager.list_available_books()
            return jsonify({
                'success': False,
                'error': f'Local GraphRAG not available for book: {book_id}',
                'available_books': available_books
            }), 500

        logger.info(f"🔍 Running local GraphRAG with interceptor: '{query}'")
        start_time = time.time()

        # Exécuter la requête avec interception
        result = graphrag_instance.query(query, param=QueryParam(mode=mode))
        elapsed_time = time.time() - start_time

        # Construire la réponse avec vraies données d'interception
        response = {
            'success': True,
            'query': query,
            'answer': result,
            'mode': mode,
            'processing_time': elapsed_time,
            'source': 'local_graphrag_with_interceptor',
            'timestamp': datetime.utcnow().isoformat()
        }

        # Ajouter les vraies données de debug
        if debug_mode:
            debug_info = graphrag_interceptor.get_real_debug_info()
            response['debug_info'] = debug_info
            response['interceptor_stats'] = {
                'queries_processed': graphrag_interceptor.query_counter,
                'last_analysis_available': bool(graphrag_interceptor.current_analysis)
            }

        logger.info(f"✅ Local GraphRAG completed in {elapsed_time:.2f}s")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in local GraphRAG query: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/query/reconciled', methods=['POST'])
@app.route('/graphrag/query', methods=['POST'])
def query_reconciled():
    """
    Reconciled query endpoint that:
    1. Takes user query + visible node IDs from frontend
    2. Fetches node details from Neo4j (source of truth)
    3. Sends focused GraphRAG query based on visible nodes
    4. Returns harmonized results
    """
    data = request.json
    query = data.get('query', '')
    visible_node_ids = data.get('visible_node_ids', [])
    mode = data.get('mode', 'local')
    debug_mode = data.get('debug_mode', False)
    book_id = data.get('book_id', None)  # Add book_id parameter

    logger.info(f"📝 Received reconciled query: '{query}' with {len(visible_node_ids)} visible nodes, mode: {mode}, book_id: {book_id}")

    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400

    try:
        # Step 1: Get node details from Neo4j for visible nodes (optional - fallback to GraphRAG only)
        node_context = {}
        if visible_node_ids:
            try:
                driver = get_neo4j_driver()
                if driver:
                    with driver.session() as session:
                        query_neo4j = """
                        MATCH (n)
                        WHERE elementId(n) IN $node_ids
                        RETURN n
                        """
                        result = session.run(query_neo4j, node_ids=visible_node_ids)

                        for record in result:
                            node = record['n']
                            node_id = node.element_id
                            node_context[node_id] = {
                                'labels': list(node.labels),
                                'properties': dict(node)
                            }
                    logger.info(f"✅ Retrieved {len(node_context)} node contexts from Neo4j")
                else:
                    logger.warning("Neo4j driver not available, proceeding with GraphRAG-only query")
            except Exception as e:
                logger.warning(f"Neo4j context failed, proceeding with GraphRAG-only query: {e}")
                node_context = {}

        # Step 2: Create context-aware query for GraphRAG
        context_prefix = ""
        if node_context:
            # Build context from visible nodes
            entities = []
            for node_id, node_data in node_context.items():
                label = node_data['labels'][0] if node_data['labels'] else 'Entity'
                name = node_data['properties'].get('name', node_data['properties'].get('title', 'Unknown'))
                entities.append(f"{name} ({label})")

            context_prefix = f"Dans le contexte des entités suivantes visibles dans le graphe: {', '.join(entities[:10])}. "
            logger.info(f"🎯 Enhanced query with {len(entities)} entities from visible nodes")

        enhanced_query = context_prefix + query
        logger.info(f"🔍 Sending to GraphRAG: {enhanced_query[:100]}...")

        # Step 3: Query GraphRAG with context - try Railway API first, fallback to local
        graphrag_data = {}
        try:
            # Try Railway GraphRAG API first
            with httpx.Client() as client:
                # Build request payload
                graphrag_payload = {
                    'query': enhanced_query,
                    'mode': mode
                }
                # Add book_id if provided
                if book_id:
                    graphrag_payload['book_id'] = book_id

                graphrag_response = client.post(
                    f"{GRAPHRAG_API_URL}/query",
                    json=graphrag_payload,
                    timeout=30.0
                )

                if graphrag_response.status_code == 200:
                    graphrag_data = graphrag_response.json()
                    logger.info(f"✅ Railway GraphRAG response received: {len(graphrag_data.get('answer', ''))} chars")
                else:
                    raise Exception(f"Railway GraphRAG API error: {graphrag_response.status_code}")

        except Exception as e:
            logger.warning(f"Railway GraphRAG failed ({e}), falling back to local GraphRAG")
            # Fallback to local GraphRAG
            try:
                graphrag_instance = get_local_graphrag(book_id or "a_rebours_huysmans")
                if graphrag_instance:
                    logger.info(f"🔍 Using local GraphRAG for query: '{enhanced_query}'")
                    start_time = time.time()
                    result = graphrag_instance.query(enhanced_query, param=QueryParam(mode=mode))
                    elapsed_time = time.time() - start_time

                    # Format response to match Railway API format
                    graphrag_data = {
                        'answer': result,
                        'mode': mode,
                        'processing_time': elapsed_time,
                        'source': 'local_graphrag'
                    }
                    logger.info(f"✅ Local GraphRAG response received: {len(result)} chars in {elapsed_time:.2f}s")
                else:
                    raise Exception("Local GraphRAG not available")
            except Exception as local_e:
                logger.error(f"Both Railway and local GraphRAG failed: {local_e}")
                graphrag_data = {
                    'answer': 'GraphRAG services temporarily unavailable',
                    'mode': mode,
                    'source': 'fallback'
                }

        # Step 4: Reconcile results with Neo4j as source of truth
        reconciled_result = {
            'success': True,
            'query': query,
            'answer': graphrag_data.get('answer', 'No answer available'),
            'context': {
                'visible_nodes_count': len(visible_node_ids),
                'node_context': list(node_context.keys())[:10],  # First 10 for response
                'mode': mode
            },
            'search_path': graphrag_data.get('searchPath', {
                'entities': [],
                'relations': [],
                'communities': []
            }),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add debug information if requested and available
        if debug_mode:
            try:
                debug_info = graphrag_interceptor.get_real_debug_info()
                reconciled_result['debug_info'] = debug_info
                logger.info(f"Debug mode: captured REAL data - {debug_info['context_stats']['prompt_length']} chars, {len(debug_info['processing_phases']['entities'])} entities")
            except Exception as e:
                logger.warning(f"Debug info not available: {e}")
                reconciled_result['debug_info'] = {
                    'context_stats': {'mode': mode, 'source': graphrag_data.get('source', 'railway_api')},
                    'processing_phases': {}
                }

        # If there are conflicts, Neo4j data takes precedence
        if 'searchPath' in graphrag_data and 'entities' in graphrag_data['searchPath']:
            for entity in graphrag_data['searchPath']['entities']:
                # Check if entity exists in our Neo4j context
                entity_name = entity.get('id', '')
                for node_id, node_data in node_context.items():
                    if node_data['properties'].get('name') == entity_name:
                        # Override with Neo4j data (source of truth)
                        entity['source'] = 'neo4j'
                        entity['verified'] = True
                        break

        return jsonify(reconciled_result)

    except Exception as e:
        logger.error(f"Error in reconciled query: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/graph/search', methods=['GET'])
def search_graph():
    """
    Search for specific nodes in Neo4j
    Query params:
    - q: search query
    - type: node type filter (optional)
    - limit: max results (default 50)
    """
    search_query = request.args.get('q', '')
    node_type = request.args.get('type', None)
    limit = min(int(request.args.get('limit', 50)), 100)

    if not search_query:
        return jsonify({
            'success': False,
            'error': 'Search query is required'
        }), 400

    try:
        driver = get_neo4j_driver()
        if driver is None:
            return jsonify({
                'success': False,
                'error': 'Neo4j database not available'
            }), 500
        with driver.session() as session:
            if node_type:
                cypher_query = """
                MATCH (n:$type)
                WHERE n.name CONTAINS $search OR n.title CONTAINS $search
                RETURN n
                LIMIT $limit
                """
                params = {'type': node_type, 'search': search_query, 'limit': limit}
            else:
                cypher_query = """
                MATCH (n)
                WHERE n.name CONTAINS $search OR n.title CONTAINS $search
                RETURN n
                LIMIT $limit
                """
                params = {'search': search_query, 'limit': limit}

            result = session.run(cypher_query, **params)
            nodes = []

            for record in result:
                node = record['n']

                # Convert node properties, handling special types
                node_properties = {}
                for key, value in dict(node).items():
                    if hasattr(value, 'isoformat'):  # DateTime objects
                        node_properties[key] = value.isoformat()
                    elif isinstance(value, (list, dict)):
                        node_properties[key] = value
                    else:
                        node_properties[key] = str(value) if value is not None else None

                nodes.append({
                    'id': node.element_id,
                    'labels': list(node.labels),
                    'properties': node_properties
                })

            return jsonify({
                'success': True,
                'nodes': nodes,
                'count': len(nodes),
                'query': search_query
            })

    except Exception as e:
        logger.error(f"Error searching graph: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about the graph"""
    try:
        driver = get_neo4j_driver()
        if driver is None:
            return jsonify({
                'success': False,
                'error': 'Neo4j database not available'
            }), 500
        with driver.session() as session:
            # Get node count by type
            node_stats_query = """
            MATCH (n)
            RETURN labels(n) as labels, count(n) as count
            ORDER BY count DESC
            """

            # Get relationship stats
            rel_stats_query = """
            MATCH ()-[r]->()
            RETURN type(r) as type, count(r) as count
            ORDER BY count DESC
            """

            # Get total counts
            total_query = """
            MATCH (n)
            WITH count(n) as node_count
            MATCH ()-[r]->()
            RETURN node_count, count(r) as rel_count
            """

            node_stats = session.run(node_stats_query)
            rel_stats = session.run(rel_stats_query)
            totals = session.run(total_query).single()

            return jsonify({
                'success': True,
                'stats': {
                    'total_nodes': totals['node_count'],
                    'total_relationships': totals['rel_count'],
                    'node_types': [{'labels': record['labels'], 'count': record['count']}
                                  for record in node_stats],
                    'relationship_types': [{'type': record['type'], 'count': record['count']}
                                          for record in rel_stats]
                }
            })

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.teardown_appcontext
def cleanup(error):
    """Cleanup on app context teardown"""
    pass

if __name__ == '__main__':
    try:
        # Register book endpoints
        register_books_endpoints(app)

        # Test connections on startup
        logger.info(f"Neo4j connection: {check_neo4j_connection()}")
        logger.info(f"GraphRAG connection: {check_graphrag_connection()}")

        # Ensure GDrive data is available on startup
        try:
            gdrive_manager.ensure_data_available()
            available_books = gdrive_manager.list_available_books()
            logger.info(f"📚 Available books: {available_books}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load GDrive data on startup: {e}")

        # Run the Flask app
        port = int(os.environ.get('PORT', 5002))
        app.run(host='0.0.0.0', port=port, debug=True)
    finally:
        close_neo4j_driver()