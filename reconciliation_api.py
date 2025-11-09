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
from pathlib import Path
import networkx as nx

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

# Local book data functions
def get_book_data_path(book_id: str = "a_rebours_huysmans") -> str:
    """Get path to local book data"""
    book_path = Path("book_data") / book_id
    if book_path.exists():
        return str(book_path)
    else:
        raise FileNotFoundError(f"Book data not found: {book_id}")

def list_available_books() -> list:
    """List all available book datasets"""
    book_data_dir = Path("book_data")
    if not book_data_dir.exists():
        return []

    books = []
    for item in book_data_dir.iterdir():
        if item.is_dir() and (item / "vdb_entities.json").exists():
            books.append(item.name)

    return sorted(books)

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
        book_data_path = get_book_data_path(book_id)

        if not book_data_path:
            available_books = list_available_books()
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
# Using local book data functions defined above
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

@app.route('/graph/relationships', methods=['GET', 'POST'])
def get_graph_relationships():
    """
    Get relationships for the displayed nodes

    GET Query params:
    - node_ids: comma-separated list of node IDs (for small requests only)
    - limit: max number of relationships to return (default 10000)

    POST Body:
    - node_ids: list of node IDs (array or comma-separated string)
    - limit: max number of relationships to return (default 10000)

    Note: Use POST for large requests (>100 nodes) to avoid HTTP header size limits
    """
    # Get parameters from GET or POST
    if request.method == 'POST':
        data = request.get_json() or {}
        node_ids_param = data.get('node_ids', '')
        limit = min(int(data.get('limit', 10000)), 50000)

        # Handle both string and array formats
        if isinstance(node_ids_param, list):
            node_ids = node_ids_param
        else:
            node_ids = node_ids_param.split(',') if node_ids_param else []
    else:
        node_ids = request.args.get('node_ids', '').split(',')
        limit = min(int(request.args.get('limit', 10000)), 50000)

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

    logger.info(f"Fetching relationships for {len(node_ids)} nodes with limit {limit} (method: {request.method})")

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
            available_books = list_available_books()
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
@app.route('/query', methods=['POST'])
def query_reconciled():
    """
    GraphRAG query endpoint
    """
    data = request.json
    query = data.get('query', '')
    mode = data.get('mode', 'local')
    debug_mode = data.get('debug_mode', False)
    book_id = data.get('book_id', None)

    logger.info(f"📝 Received query: '{query}', mode: {mode}, book_id: {book_id}")

    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400

    try:
        # Query GraphRAG - try Railway API first, fallback to local
        graphrag_data = {}
        try:
            # Try Railway GraphRAG API first
            with httpx.Client() as client:
                graphrag_payload = {
                    'query': query,
                    'mode': mode
                }
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
                    logger.info(f"🔍 Using local GraphRAG for query: '{query}'")
                    start_time = time.time()
                    result = graphrag_instance.query(query, param=QueryParam(mode=mode))
                    elapsed_time = time.time() - start_time

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

        result = {
            'success': True,
            'query': query,
            'answer': graphrag_data.get('answer', 'No answer available'),
            'context': {
                'mode': mode
            },
            'search_path': graphrag_data.get('searchPath', {
                'entities': [],
                'relations': [],
                'communities': []
            }),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Add debug information if requested
        if debug_mode:
            try:
                debug_info = graphrag_interceptor.get_real_debug_info()
                result['debug_info'] = debug_info
                logger.info(f"Debug mode: captured REAL data")
            except Exception as e:
                logger.warning(f"Debug info not available: {e}")
                result['debug_info'] = {
                    'context_stats': {'mode': mode, 'source': graphrag_data.get('source', 'railway_api')},
                    'processing_phases': {}
                }

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in query: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/graph/search-nodes', methods=['POST'])
def search_nodes_from_graphrag():
    """
    Extract and display nodes from GraphRAG based on query
    Returns entities, relationships, and communities like test_dickens_community
    """
    data = request.json
    query = data.get('query', '')
    mode = data.get('mode', 'local')
    book_id = data.get('book_id', None)

    logger.info(f"🔍 Searching nodes from GraphRAG for: '{query}'")

    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400

    try:
        # Get GraphRAG instance
        graphrag_instance = get_local_graphrag(book_id or "a_rebours_huysmans")
        if not graphrag_instance:
            raise Exception("GraphRAG not available")

        logger.info(f"🔍 Running GraphRAG query to extract entities: '{query}'")

        # Run query to trigger entity/relationship extraction
        start_time = time.time()
        result = graphrag_instance.query(query, param=QueryParam(mode=mode))
        elapsed_time = time.time() - start_time

        logger.info(f"✅ Query completed in {elapsed_time:.2f}s")

        # Now extract entities and relationships from the index
        # Access the graph structure to get nodes
        try:
            # Get the storage manager
            from nano_graphrag._storage import get_storage_class
            storage_class = get_storage_class("networkx")

            # Try to load the graph
            graph_path = Path(graphrag_instance.working_dir) / "graph_chunk_entity_relation.graphml"

            if graph_path.exists():
                logger.info(f"📊 Loading graph from: {graph_path}")
                G = nx.read_graphml(str(graph_path))

                # Extract nodes with attributes
                nodes = []
                relationships = []
                communities = set()

                for node_id, node_data in G.nodes(data=True):
                    node_obj = {
                        'id': node_id,
                        'labels': [node_data.get('entity_type', 'Entity')],
                        'properties': {
                            'name': node_data.get('entity_name', node_id),
                            'description': node_data.get('description', ''),
                        },
                        'degree': G.degree(node_id),
                        'centrality_score': G.degree(node_id)
                    }
                    nodes.append(node_obj)

                # Extract relationships
                for source, target, edge_data in G.edges(data=True):
                    rel_obj = {
                        'id': f"{source}_{target}",
                        'type': edge_data.get('weight_label', 'RELATED'),
                        'source': source,
                        'target': target,
                        'properties': {
                            'description': edge_data.get('weight_label', 'Related to'),
                            'weight': float(edge_data.get('weight', 1.0))
                        }
                    }
                    relationships.append(rel_obj)

                logger.info(f"✅ Extracted {len(nodes)} nodes and {len(relationships)} relationships")

                return jsonify({
                    'success': True,
                    'query': query,
                    'answer': result,
                    'nodes': nodes[:500],  # Limit to 500 nodes for performance
                    'relationships': relationships[:5000],  # Limit relationships
                    'graph': {
                        'total_nodes': len(nodes),
                        'total_relationships': len(relationships),
                        'node_types': list(set([node['labels'][0] for node in nodes]))
                    },
                    'processing_time': elapsed_time,
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                # Fallback: return answer without graph
                logger.warning(f"Graph file not found at {graph_path}, returning answer only")
                return jsonify({
                    'success': True,
                    'query': query,
                    'answer': result,
                    'nodes': [],
                    'relationships': [],
                    'graph': {
                        'total_nodes': 0,
                        'total_relationships': 0,
                        'note': 'Graph file not available yet'
                    },
                    'processing_time': elapsed_time,
                    'timestamp': datetime.utcnow().isoformat()
                })

        except Exception as graph_e:
            logger.warning(f"Could not extract graph data: {graph_e}")
            # Return at least the answer
            return jsonify({
                'success': True,
                'query': query,
                'answer': result,
                'nodes': [],
                'relationships': [],
                'error': 'Graph extraction failed, but answer provided',
                'processing_time': elapsed_time,
                'timestamp': datetime.utcnow().isoformat()
            })

    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/query/multi-book', methods=['POST'])
def query_multi_book():
    """
    Query ALL books sequentially with GraphRAG
    Returns aggregated results with per-book metadata
    Like test_query_analysis.py but across multiple books
    """
    data = request.json
    query = data.get('query', '')
    mode = data.get('mode', 'local')
    debug_mode = data.get('debug_mode', False)

    if not query:
        return jsonify({'success': False, 'error': 'Query is required'}), 400

    try:
        logger.info(f"🔍 Starting multi-book query: '{query}'")

        available_books = list_available_books()
        logger.info(f"📚 Available books: {available_books}")

        all_results = []
        aggregated_entities = {}
        aggregated_relationships = {}
        aggregated_communities = {}
        total_processing_time = 0

        for book_id in available_books:
            logger.info(f"\n📖 Querying book: {book_id}")
            book_start_time = time.time()

            try:
                graphrag_instance = get_local_graphrag(book_id)
                if not graphrag_instance:
                    logger.warning(f"⚠️ Could not initialize GraphRAG for {book_id}")
                    continue

                logger.info(f"🔍 Running GraphRAG query on {book_id}: '{query}'")
                result = graphrag_instance.query(query, param=QueryParam(mode=mode))

                book_processing_time = time.time() - book_start_time
                total_processing_time += book_processing_time

                debug_info = None
                if debug_mode:
                    debug_info = graphrag_interceptor.get_real_debug_info()

                book_result = {
                    'book_id': book_id,
                    'answer': result,
                    'processing_time': book_processing_time,
                    'debug_info': debug_info
                }

                if debug_info:
                    entities = debug_info.get('processing_phases', {}).get('entity_selection', {}).get('entities', [])
                    relationships = debug_info.get('processing_phases', {}).get('relationship_mapping', {}).get('relationships', [])
                    communities = debug_info.get('processing_phases', {}).get('community_analysis', {}).get('communities', [])

                    for entity in entities:
                        entity_id = entity.get('id')
                        if entity_id not in aggregated_entities:
                            aggregated_entities[entity_id] = {
                                **entity,
                                'books': [book_id],
                                'found_in': [book_id]
                            }
                        else:
                            if book_id not in aggregated_entities[entity_id]['books']:
                                aggregated_entities[entity_id]['books'].append(book_id)
                                aggregated_entities[entity_id]['found_in'].append(book_id)

                    for rel in relationships:
                        rel_key = f"{rel.get('source')}--{rel.get('target')}"
                        if rel_key not in aggregated_relationships:
                            aggregated_relationships[rel_key] = {
                                **rel,
                                'books': [book_id],
                                'found_in': [book_id]
                            }
                        else:
                            if book_id not in aggregated_relationships[rel_key]['books']:
                                aggregated_relationships[rel_key]['books'].append(book_id)
                                aggregated_relationships[rel_key]['found_in'].append(book_id)

                    for comm in communities:
                        comm_id = comm.get('id')
                        if comm_id not in aggregated_communities:
                            aggregated_communities[comm_id] = {
                                **comm,
                                'books': [book_id],
                                'found_in': [book_id]
                            }
                        else:
                            if book_id not in aggregated_communities[comm_id]['books']:
                                aggregated_communities[comm_id]['books'].append(book_id)
                                aggregated_communities[comm_id]['found_in'].append(book_id)

                logger.info(f"✅ {book_id}: {len(entities)} entities, {len(relationships)} relationships, {len(communities)} communities in {book_processing_time:.2f}s")
                all_results.append(book_result)

            except Exception as book_error:
                logger.error(f"❌ Error querying {book_id}: {book_error}")
                all_results.append({
                    'book_id': book_id,
                    'error': str(book_error),
                    'processing_time': time.time() - book_start_time
                })

        response = {
            'success': True,
            'query': query,
            'mode': mode,
            'total_processing_time': total_processing_time,
            'books_queried': available_books,
            'books_with_results': len([r for r in all_results if 'error' not in r]),
            'book_results': all_results,
            'aggregated': {
                'entities': list(aggregated_entities.values()),
                'relationships': list(aggregated_relationships.values()),
                'communities': list(aggregated_communities.values())
            },
            'summary': {
                'total_entities': len(aggregated_entities),
                'total_relationships': len(aggregated_relationships),
                'total_communities': len(aggregated_communities),
                'entities_in_multiple_books': len([e for e in aggregated_entities.values() if len(e['books']) > 1]),
                'relationships_in_multiple_books': len([r for r in aggregated_relationships.values() if len(r['books']) > 1])
            },
            'timestamp': datetime.utcnow().isoformat()
        }

        logger.info(f"✅ Multi-book query complete: {response['summary']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in multi-book query: {e}")
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

        # List available books on startup
        try:
            available_books = list_available_books()
            logger.info(f"📚 Available books: {available_books}")
        except Exception as e:
            logger.warning(f"⚠️ Could not list books on startup: {e}")

        # Run the Flask app
        port = int(os.environ.get('PORT', 5002))
        app.run(host='0.0.0.0', port=port, debug=False)
    finally:
        close_neo4j_driver()