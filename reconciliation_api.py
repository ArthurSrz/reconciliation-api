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
GRAPHRAG_API_URL = os.getenv('GRAPHRAG_API_URL', 'https://comfortable-gentleness-production-8603.up.railway.app')

# Neo4j driver instance
neo4j_driver = None

def get_neo4j_driver():
    """Get or create Neo4j driver instance"""
    global neo4j_driver
    if neo4j_driver is None and NEO4J_URI and NEO4J_URI != 'bolt://localhost:7687':
        neo4j_driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return neo4j_driver

def close_neo4j_driver():
    """Close Neo4j driver"""
    global neo4j_driver
    if neo4j_driver:
        neo4j_driver.close()
        neo4j_driver = None

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
                nodes.append({
                    'id': node.element_id,
                    'labels': list(node.labels),
                    'properties': dict(node),
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
    """
    node_ids = request.args.get('node_ids', '').split(',')

    if not node_ids or node_ids == ['']:
        return jsonify({
            'success': False,
            'error': 'No node IDs provided'
        }), 400

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            query = """
            MATCH (n)
            WHERE elementId(n) IN $node_ids
            MATCH (n)-[r]-(m)
            WHERE elementId(m) IN $node_ids
            RETURN DISTINCT n, r, m
            """

            result = session.run(query, node_ids=node_ids)
            relationships = []

            for record in result:
                rel = record['r']
                relationships.append({
                    'id': rel.element_id,
                    'type': rel.type,
                    'source': record['n'].element_id,
                    'target': record['m'].element_id,
                    'properties': dict(rel)
                })

            return jsonify({
                'success': True,
                'relationships': relationships,
                'count': len(relationships)
            })

    except Exception as e:
        logger.error(f"Error fetching relationships: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/query/reconciled', methods=['POST'])
async def query_reconciled():
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

    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400

    try:
        # Step 1: Get node details from Neo4j for visible nodes
        node_context = {}
        if visible_node_ids:
            driver = get_neo4j_driver()
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

        enhanced_query = context_prefix + query

        # Step 3: Query GraphRAG with context
        async with httpx.AsyncClient() as client:
            graphrag_response = await client.post(
                f"{GRAPHRAG_API_URL}/query",
                json={
                    'query': enhanced_query,
                    'mode': mode
                },
                timeout=30.0
            )

            if graphrag_response.status_code != 200:
                raise Exception(f"GraphRAG API error: {graphrag_response.status_code}")

            graphrag_data = graphrag_response.json()

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
                nodes.append({
                    'id': node.element_id,
                    'labels': list(node.labels),
                    'properties': dict(node)
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
        # Test connections on startup
        logger.info(f"Neo4j connection: {check_neo4j_connection()}")
        logger.info(f"GraphRAG connection: {check_graphrag_connection()}")

        # Run the Flask app
        app.run(host='0.0.0.0', port=5002, debug=True)
    finally:
        close_neo4j_driver()