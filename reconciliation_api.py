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
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re
import csv
from nano_graphrag import GraphRAG, QueryParam
from io import StringIO
from functools import wraps, lru_cache
import time
from dotenv import load_dotenv
from pathlib import Path
import networkx as nx

# Load environment variables from .env file
load_dotenv()

# Cache for chunks data - LRU cache to avoid reloading huge JSON files
@lru_cache(maxsize=20)  # Cache up to 20 book chunk files
def load_chunks_file(book_id: str) -> Dict[str, Any]:
    """
    Load chunks data for a book with caching to improve performance.

    Args:
        book_id: Book identifier

    Returns:
        Dictionary of chunks indexed by chunk_id

    Note:
        Uses LRU cache to avoid repeatedly loading large JSON files.
        Cache size of 20 covers all books (9 books) with room for reloads.
    """
    base_path = get_book_data_base_path()
    chunks_file = Path(base_path) / book_id / "kv_store_text_chunks.json"

    if not chunks_file.exists():
        raise FileNotFoundError(f"Chunks file not found for book: {book_id}")

    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    logger.info(f"📦 Loaded {len(chunks_data)} chunks for {book_id} (cached)")
    return chunks_data

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
    "https://le-graphe-de-borges.vercel.app",  # Production domain
    "https://borges-library*.vercel.app",
    "https://*.vercel.app"
], methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type', 'X-Admin-API-Key'])

# Register blueprints for modular endpoints
from endpoints.ingestion import ingestion_bp
app.register_blueprint(ingestion_bp)

# Configuration
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD', 'password')
GRAPHRAG_API_URL = os.getenv('GRAPHRAG_API_URL', 'https://borgesgraph-production.up.railway.app')

def get_book_data_base_path():
    """Get the base path for book data - Railway volume or local directory"""
    # On Railway with volume mounted
    if volume_path := os.environ.get('RAILWAY_VOLUME_MOUNT_PATH'):
        logger.info(f"📂 Using Railway volume path: {volume_path}")
        return volume_path
    # Local development
    logger.info("📂 Using local book_data directory")
    return "book_data"

def get_book_data_path(book_id: str) -> str:
    """Get the full path to a specific book's data directory"""
    base_path = get_book_data_base_path()
    book_path = os.path.join(base_path, book_id)
    return book_path

def ensure_book_data_available():
    """Check if book data is available in current path"""
    base_path = get_book_data_base_path()
    base_dir = Path(base_path)

    # Check if we have any book data
    if base_dir.exists() and any(base_dir.iterdir()):
        logger.info(f"📚 Book data already exists in {base_path}")
        return True

    logger.info("📂 No book data found")
    return False

def create_sample_book_data():
    """Create sample book data in the volume for testing"""
    volume_path = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH')
    if not volume_path:
        logger.error("No Railway volume path found")
        return False

    try:
        import json
        volume_dir = Path(volume_path)
        volume_dir.mkdir(exist_ok=True)

        # Create a sample book directory
        sample_book = volume_dir / "test_book"
        sample_book.mkdir(exist_ok=True)

        # Create a minimal vdb_entities.json file
        sample_entities = {
            "data": [
                {"id": "test_entity_1", "content": "This is a test entity"},
                {"id": "test_entity_2", "content": "This is another test entity"}
            ]
        }

        with open(sample_book / "vdb_entities.json", 'w') as f:
            json.dump(sample_entities, f)

        logger.info("📚 Created sample book data in volume")
        return True

    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        return False

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
def get_book_data_path_legacy(book_id: str = "a_rebours_huysmans") -> str:
    """Legacy function - use get_book_data_path instead"""
    base_path = get_book_data_base_path()
    book_path = Path(base_path) / book_id
    if book_path.exists():
        return str(book_path)
    else:
        raise FileNotFoundError(f"Book data not found: {book_id}")

def list_available_books() -> list:
    """List all available book datasets"""
    book_data_dir = Path(get_book_data_base_path())
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
book_data_path = None
current_book_id = None  # Track the currently loaded book

def _get_cross_book_neo4j_relationships(aggregated_entities):
    """
    Get Neo4j relationships for entities that appear in multiple books
    Returns cross-book connections and community memberships
    """
    cross_book_relationships = []

    # Find entities that appear in multiple books
    multi_book_entities = {
        entity_id: entity_data for entity_id, entity_data in aggregated_entities.items()
        if len(entity_data.get('books', [])) > 1
    }

    if not multi_book_entities:
        logger.info("📚 No multi-book entities found for cross-book enrichment")
        return []

    try:
        driver = get_neo4j_driver()
        with driver.session() as session:
            for entity_id, entity_data in multi_book_entities.items():
                entity_name = entity_data.get('name', '')
                books = entity_data.get('books', [])

                if not entity_name:
                    continue

                # Query for this entity's relationships across all contexts
                cross_query = """
                MATCH (e1:Entity {name: $entity_name})-[r]->(target)
                WHERE target:Entity OR target:Community
                RETURN DISTINCT
                    target.name as target_name,
                    labels(target) as target_labels,
                    type(r) as relation_type,
                    target.entity_type as target_type,
                    target.title as target_title,
                    r.weight as weight
                LIMIT 20
                """

                results = session.run(cross_query, entity_name=entity_name)

                for record in results:
                    target_name = record.get('target_name', '')
                    target_labels = record.get('target_labels', [])
                    relation_type = record.get('relation_type', 'RELATED')
                    weight = record.get('weight', 1.0)

                    if target_name and target_name != entity_name:
                        # Determine if target is community or entity
                        is_community = 'Community' in target_labels
                        target_title = record.get('target_title', target_name)

                        cross_book_relationships.append({
                            'source': entity_name,
                            'target': target_name,
                            'type': relation_type,
                            'description': f"{entity_name} {relation_type.lower()} {target_title}",
                            'weight': weight,
                            'books': books,  # Books where source entity was found
                            'is_cross_book': True,
                            'is_community_link': is_community,
                            'source_books': len(books),  # Number of books containing this entity
                            'traversal_order': len(cross_book_relationships) + 1
                        })

            logger.info(f"🌐 Found {len(cross_book_relationships)} cross-book Neo4j relationships for {len(multi_book_entities)} multi-book entities")

    except Exception as e:
        logger.warning(f"❌ Error fetching cross-book Neo4j relationships: {e}")
        return []

    return cross_book_relationships

def get_local_graphrag(book_id: str = "a_rebours_huysmans"):
    """Get or create local GraphRAG instance with real interceptor and book data"""
    global local_graphrag, book_data_path, current_book_id

    # Always ensure we have fresh data for the requested book
    try:
        # Ensure data is available
        logger.info(f"📥 Ensuring GraphRAG data is available for book: {book_id}")
        book_data_path = get_book_data_path(book_id)

        if not book_data_path:
            available_books = list_available_books()
            logger.error(f"❌ Book {book_id} not found. Available: {available_books}")
            return None

        # Recreate GraphRAG if it doesn't exist OR if the book changed
        if local_graphrag is None or current_book_id != book_id:
            logger.info(f"🔧 Creating new GraphRAG instance for path: {book_data_path}")

            try:
                from nano_graphrag._llm import gpt_4o_mini_complete
                logger.info("✅ Imported gpt_4o_mini_complete")
            except Exception as e:
                logger.error(f"❌ Failed to import gpt_4o_mini_complete: {e}")
                raise

            # Créer l'intercepteur LLM comme dans test_query_analysis.py
            try:
                logger.info("🔧 Creating intercepted LLM function...")
                intercepted_llm = graphrag_interceptor.intercept_query_processing(gpt_4o_mini_complete)
                logger.info("✅ LLM interceptor created")
            except Exception as e:
                logger.error(f"❌ Failed to create LLM interceptor: {e}")
                raise

            # Intercepter aussi la fonction _build_local_query_context pour capturer les vraies entités
            try:
                logger.info("🔧 Intercepting _build_local_query_context function...")
                from nano_graphrag._op import _build_local_query_context
                original_build_context = _build_local_query_context
                intercepted_build_context = graphrag_interceptor.intercept_build_local_query_context(original_build_context)

                # Remplacer temporairement la fonction dans le module
                import nano_graphrag._op
                nano_graphrag._op._build_local_query_context = intercepted_build_context
                logger.info("✅ Successfully intercepted _build_local_query_context function")
            except Exception as e:
                logger.warning(f"⚠️ Could not intercept _build_local_query_context: {e}")

            # Intercepter aussi la fonction global_query pour capturer les données du mode global
            try:
                logger.info("🔧 Intercepting global_query function...")
                from nano_graphrag._op import global_query
                original_global_query = global_query
                intercepted_global_query = graphrag_interceptor.intercept_global_query(original_global_query)

                # Remplacer temporairement la fonction dans le module
                nano_graphrag._op.global_query = intercepted_global_query
                logger.info("✅ Successfully intercepted global_query function")
            except Exception as e:
                logger.warning(f"⚠️ Could not intercept global_query: {e}")

            try:
                logger.info("🔧 Creating GraphRAG instance...")
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
                logger.info("✅ GraphRAG instance created successfully")
                current_book_id = book_id  # Track the current book
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON parsing error in GraphRAG initialization: {e}")
                logger.error(f"❌ This suggests corrupted files in production environment")
                # Try to create GraphRAG without some problematic files
                try:
                    logger.info("🔄 Attempting GraphRAG creation with minimal config...")
                    local_graphrag = GraphRAG(
                        working_dir=book_data_path,
                        best_model_func=intercepted_llm,
                        cheap_model_func=intercepted_llm
                    )
                    logger.info("✅ GraphRAG instance created with minimal config")
                    current_book_id = book_id  # Track the current book
                except Exception as e2:
                    logger.error(f"❌ Failed even with minimal config: {e2}")
                    raise e
            except Exception as e:
                logger.error(f"❌ Failed to create GraphRAG instance: {e}")
                import traceback
                logger.error(f"❌ Full traceback: {traceback.format_exc()}")
                raise

            logger.info(f"✅ Local GraphRAG initialized with book data from: {book_data_path}")

        return local_graphrag

    except Exception as e:
        logger.error(f"❌ Failed to initialize local GraphRAG with book data: {e}")
        return None

# Import du nouvel intercepteur et du gestionnaire de données
from graphrag_interceptor import graphrag_interceptor
# Using local book data functions defined above
from endpoints.books import register_books_endpoints
from endpoints.provenance import register_provenance_endpoints

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

def create_simulated_debug_info(processing_time_s: float = 2.0) -> Dict[str, Any]:
    """
    Créer des données de debug simulées pour l'animation
    Basé sur les patterns typiques observés dans les logs GraphRAG
    """
    # Simuler des entités typiques trouvées dans les livres
    simulated_entities = [
        {"id": "Gary", "name": "Gary", "type": "PERSON", "rank": 1, "score": 0.95},
        {"id": "Société", "name": "Société", "type": "CONCEPT", "rank": 2, "score": 0.90},
        {"id": "Racisme", "name": "Racisme", "type": "CONCEPT", "rank": 3, "score": 0.85},
        {"id": "Amérique", "name": "Amérique", "type": "LOCATION", "rank": 4, "score": 0.80},
        {"id": "France", "name": "France", "type": "LOCATION", "rank": 5, "score": 0.75},
        {"id": "Guerre", "name": "Guerre", "type": "EVENT", "rank": 6, "score": 0.70},
        {"id": "Civilisation", "name": "Civilisation", "type": "CONCEPT", "rank": 7, "score": 0.65},
        {"id": "Humanité", "name": "Humanité", "type": "CONCEPT", "rank": 8, "score": 0.60},
        {"id": "Écrivain", "name": "Écrivain", "type": "PERSON", "rank": 9, "score": 0.55},
        {"id": "Littérature", "name": "Littérature", "type": "CONCEPT", "rank": 10, "score": 0.50},
        {"id": "Politique", "name": "Politique", "type": "CONCEPT", "rank": 11, "score": 0.45},
        {"id": "Histoire", "name": "Histoire", "type": "CONCEPT", "rank": 12, "score": 0.40},
        {"id": "Culture", "name": "Culture", "type": "CONCEPT", "rank": 13, "score": 0.35},
        {"id": "Philosophie", "name": "Philosophie", "type": "CONCEPT", "rank": 14, "score": 0.30},
        {"id": "Morale", "name": "Morale", "type": "CONCEPT", "rank": 15, "score": 0.25},
        {"id": "Justice", "name": "Justice", "type": "CONCEPT", "rank": 16, "score": 0.20},
        {"id": "Liberté", "name": "Liberté", "type": "CONCEPT", "rank": 17, "score": 0.18},
        {"id": "Vérité", "name": "Vérité", "type": "CONCEPT", "rank": 18, "score": 0.15},
        {"id": "Europe", "name": "Europe", "type": "LOCATION", "rank": 19, "score": 0.12},
        {"id": "Monde", "name": "Monde", "type": "CONCEPT", "rank": 20, "score": 0.10}
    ]

    # Simuler des communautés
    simulated_communities = [
        {"id": "1", "title": "Critique sociale et racisme", "relevance": 0.9},
        {"id": "2", "title": "Géopolitique et civilisations", "relevance": 0.8},
        {"id": "3", "title": "Littérature et société", "relevance": 0.7},
        {"id": "4", "title": "Histoire et politique", "relevance": 0.6},
        {"id": "5", "title": "Philosophie morale", "relevance": 0.5}
    ]

    # Simuler des relations
    simulated_relationships = [
        {"source": "Gary", "target": "Société", "description": "Critique de la société"},
        {"source": "Racisme", "target": "Amérique", "description": "Racisme en Amérique"},
        {"source": "Gary", "target": "Littérature", "description": "Auteur et son œuvre"},
        {"source": "Guerre", "target": "Civilisation", "description": "Impact de la guerre"},
        {"source": "France", "target": "Europe", "description": "Contexte géographique"}
    ]

    processing_time_ms = processing_time_s * 1000

    return {
        "processing_phases": {
            "entity_selection": {
                "entities": simulated_entities,
                "duration_ms": int(processing_time_ms * 0.2),
                "phase": "explosion",
                "real_count": len(simulated_entities)
            },
            "community_analysis": {
                "communities": simulated_communities,
                "duration_ms": int(processing_time_ms * 0.4),
                "phase": "filtering",
                "real_count": len(simulated_communities)
            },
            "relationship_mapping": {
                "relationships": simulated_relationships,
                "duration_ms": int(processing_time_ms * 0.3),
                "phase": "synthesis",
                "real_count": len(simulated_relationships)
            },
            "text_synthesis": {
                "sources": [
                    {"id": "sim_source_1", "content": "Extracted text chunk 1...", "relevance": 0.9},
                    {"id": "sim_source_2", "content": "Extracted text chunk 2...", "relevance": 0.8},
                    {"id": "sim_source_3", "content": "Extracted text chunk 3...", "relevance": 0.7}
                ],
                "duration_ms": int(processing_time_ms * 0.1),
                "phase": "crystallization"
            }
        },
        "context_stats": {
            "total_time_ms": processing_time_ms,
            "mode": "local",
            "prompt_length": 1500  # Simulé
        },
        "animation_timeline": [
            {
                "phase": "explosion",
                "duration": 2000,
                "description": f"Analyzing {len(simulated_entities)} entities and {len(simulated_communities)} communities",
                "entity_count": len(simulated_entities),
                "community_count": len(simulated_communities)
            },
            {
                "phase": "filtering",
                "duration": 3000,
                "description": f"Selected {len(simulated_communities)} relevant communities",
                "community_count": len(simulated_communities)
            },
            {
                "phase": "synthesis",
                "duration": 2000,
                "description": f"Mapped {len(simulated_relationships)} relationships",
                "relationship_count": len(simulated_relationships)
            },
            {
                "phase": "crystallization",
                "duration": 1000,
                "description": "Generating contextual answer"
            }
        ]
    }

def clean_quotes(text):
    """Remove quotes from entity names and clean text"""
    if isinstance(text, str):
        return text.replace('"', '').replace("'", '').strip()
    return str(text)

def enrich_relationships_with_graphml(G, relationships, book_id=None):
    """
    Enrich relationship data with rich metadata from GraphML.
    Added book filtering to prevent cross-book contamination.

    Args:
        G: NetworkX graph loaded from GraphML
        relationships: Basic relationships from debug_info
        book_id: Current book ID for filtering (prevents cross-book contamination)

    Returns:
        Enriched relationships with weights, descriptions, source chunks, etc.
    """
    enriched = []

    logger.info(f"🔍 Enriching {len(relationships)} relationships with GraphML metadata for book: {book_id}")

    # If book_id is provided, create a filtered set of valid entity names from the current book
    valid_entities_for_book = set()
    if book_id:
        # Extract valid entity names from the current book's GraphML
        for node_id, node_data in G.nodes(data=True):
            node_name = clean_quotes(str(node_data.get('entity_name', node_id)))
            if node_name:
                valid_entities_for_book.add(node_name.upper())

        logger.info(f"📚 Found {len(valid_entities_for_book)} valid entities in book '{book_id}' for filtering")

        # Debug: show sample valid entities
        if valid_entities_for_book:
            sample_entities = list(valid_entities_for_book)[:5]
            logger.info(f"📝 Sample valid entities for {book_id}: {sample_entities}")

    for rel_data in relationships:
        source_clean = clean_quotes(str(rel_data.get('source', '')))
        target_clean = clean_quotes(str(rel_data.get('target', '')))

        # Log the first 3 relationships for debugging
        if len(enriched) < 3:
            logger.info(f"🔍 Looking for relationship: '{source_clean}' -> '{target_clean}' (type: {rel_data.get('relation', 'unknown')})")

        # BOOK FILTERING: Only enrich relationships where BOTH entities are from the current book
        # (GraphML only contains internal book entities, not cross-book relationships)
        if book_id and valid_entities_for_book:
            source_valid = any(source_clean.upper() in entity or entity in source_clean.upper()
                             for entity in valid_entities_for_book)
            target_valid = any(target_clean.upper() in entity or entity in target_clean.upper()
                             for entity in valid_entities_for_book)

            # Skip relationships involving entities outside this book
            # (These are cross-book relationships which GraphML doesn't contain)
            if not source_valid or not target_valid:
                if len(enriched) < 3:  # Log first few for debugging
                    logger.debug(f"⏭️ Skipping cross-book relationship: '{source_clean}' -> '{target_clean}' (source_valid: {source_valid}, target_valid: {target_valid})")

                # For cross-book relationships, just set no GraphML metadata
                enriched_rel = dict(rel_data)
                enriched_rel.update({
                    'has_graphml_metadata': False,
                    'filtered_for_book': book_id
                })
                enriched.append(enriched_rel)
                continue

        # Default enriched relationship
        enriched_rel = dict(rel_data)  # Copy original data

        # Try to find this relationship in GraphML with rich metadata
        graphml_edge_data = None

        # Check both directions since GraphML might be undirected
        for source_node, target_node, edge_data in G.edges(data=True):
            source_node_clean = clean_quotes(str(source_node))
            target_node_clean = clean_quotes(str(target_node))

            # Check exact match first
            if ((source_clean.upper() == source_node_clean.upper() and target_clean.upper() == target_node_clean.upper()) or
                (target_clean.upper() == source_node_clean.upper() and source_clean.upper() == target_node_clean.upper())):
                graphml_edge_data = edge_data
                logger.debug(f"✅ Exact GraphML match: {source_clean} -> {target_clean}")
                break

            # Then check fuzzy match (substring matching)
            elif ((source_clean.upper() in source_node_clean.upper() and target_clean.upper() in target_node_clean.upper()) or
                  (target_clean.upper() in source_node_clean.upper() and source_clean.upper() in target_node_clean.upper())):
                graphml_edge_data = edge_data
                logger.debug(f"🔍 Fuzzy GraphML match: {source_clean} -> {target_clean} matched with {source_node_clean} -> {target_node_clean}")
                break

        # Enrich with GraphML metadata if found
        if graphml_edge_data:
            enriched_rel.update({
                'graphml_weight': float(graphml_edge_data.get('weight') or 1.0),
                'graphml_description': clean_quotes(graphml_edge_data.get('description', '')),
                'graphml_source_chunks': graphml_edge_data.get('source_id', ''),
                'graphml_order': int(graphml_edge_data.get('order', 0)),
                'has_graphml_metadata': True,
                'filtered_for_book': book_id  # Track which book this was filtered for
            })
            logger.debug(f"✅ Enriched relationship {source_clean} -> {target_clean} with GraphML metadata")
        else:
            enriched_rel.update({
                'has_graphml_metadata': False,
                'filtered_for_book': book_id  # Track which book this was filtered for
            })
            logger.debug(f"⚠️ No GraphML metadata found for {source_clean} -> {target_clean}")

        enriched.append(enriched_rel)

    filtered_count = len(relationships) - len(enriched)
    logger.info(f"🔗 Enriched {len([r for r in enriched if r.get('has_graphml_metadata')])} relationships with GraphML metadata out of {len(enriched)}")
    if filtered_count > 0:
        logger.info(f"🚫 Filtered out {filtered_count} cross-book relationships for book '{book_id}'")

    return enriched

def enrich_nodes_with_graphml(G, entity_names, book_id=None):
    """
    Enrich node data with rich metadata from GraphML.
    Added book filtering to prevent cross-book contamination.

    Args:
        G: NetworkX graph loaded from GraphML
        entity_names: List of entity names to enrich
        book_id: Current book ID for filtering (prevents cross-book contamination)

    Returns:
        Enriched nodes with descriptions, clusters, entity types, etc.
    """
    enriched_nodes = []

    logger.info(f"🔍 Enriching {len(entity_names)} entities with GraphML metadata for book: {book_id}")

    # If book_id is provided, create a filtered set of valid entity names from the current book
    valid_entities_for_book = set()
    if book_id:
        # Extract valid entity names from the current book's GraphML
        for node_id, node_data in G.nodes(data=True):
            node_name = clean_quotes(str(node_data.get('entity_name', node_id)))
            if node_name:
                valid_entities_for_book.add(node_name.upper())

        logger.info(f"📚 Found {len(valid_entities_for_book)} valid entities in book '{book_id}' for node filtering")

    for entity_name in entity_names:
        # BOOK FILTERING: Skip entities not from the current book
        if book_id and valid_entities_for_book:
            entity_valid = any(entity_name.upper() in valid_entity or valid_entity in entity_name.upper()
                             for valid_entity in valid_entities_for_book)

            if not entity_valid:
                logger.debug(f"🚫 Filtering out entity - '{entity_name}' not in book '{book_id}'")
                continue

        # Find matching node in GraphML
        graphml_node_data = None

        for node_id, node_data in G.nodes(data=True):
            node_name = clean_quotes(str(node_data.get('entity_name', node_id)))

            # Check for fuzzy match
            if entity_name.upper() in node_name.upper() or node_name.upper() in entity_name.upper():
                graphml_node_data = node_data
                break

        # Create enriched node
        enriched_node = {
            'id': entity_name,
            'name': entity_name,
            'graphrag_node': True,
            'filtered_for_book': book_id  # Track which book this was filtered for
        }

        # Enrich with GraphML metadata if found
        if graphml_node_data:
            enriched_node.update({
                'entity_type': clean_quotes(graphml_node_data.get('entity_type', 'Unknown')),
                'description': clean_quotes(graphml_node_data.get('description', '')),
                'clusters': graphml_node_data.get('clusters', ''),
                'source_chunks': graphml_node_data.get('source_id', ''),
                'has_graphml_metadata': True
            })
            logger.debug(f"✅ Enriched node {entity_name} with GraphML metadata")
        else:
            enriched_node['has_graphml_metadata'] = False
            logger.debug(f"⚠️ No GraphML metadata found for node {entity_name}")

        enriched_nodes.append(enriched_node)

    filtered_count = len(entity_names) - len(enriched_nodes)
    logger.info(f"🎯 Enriched {len([n for n in enriched_nodes if n.get('has_graphml_metadata')])} nodes with GraphML metadata out of {len(enriched_nodes)}")
    if filtered_count > 0:
        logger.info(f"🚫 Filtered out {filtered_count} cross-book entities for book '{book_id}'")

    return enriched_nodes

def extract_selected_nodes_from_graphrag(book_id: str, debug_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extraire les nœuds et relations réellement utilisés par GraphRAG depuis le graphe principal
    Basé sur les entités mentionnées dans debug_info
    """
    try:
        # Charger le graphe principal du livre
        from pathlib import Path
        import networkx as nx

        # Use dynamic path for both local and Railway volume
        base_path = get_book_data_base_path()
        graph_path = Path(base_path) / book_id / "graph_chunk_entity_relation.graphml"

        if not graph_path.exists():
            logger.warning(f"Graph file not found: {graph_path}")
            return {"nodes": [], "relationships": []}

        G = nx.read_graphml(str(graph_path))

        # Obtenir les noms d'entités de debug_info
        entities = debug_info.get('processing_phases', {}).get('entity_selection', {}).get('entities', [])
        entity_names = [entity.get('name', entity.get('id', '')) for entity in entities]

        logger.info(f"🔍 Looking for nodes matching entities: {entity_names}")
        logger.info(f"🔍 Debug info entities count: {len(entities)}")
        logger.info(f"🔍 Debug info structure: {debug_info.get('processing_phases', {}).keys()}")

        # Si pas d'entités dans debug_info, utiliser des entités simulées réalistes pour test
        if not entity_names:
            logger.warning("⚠️ No entities found in debug_info, creating simulated selection for demo")
            # Prendre des nœuds au hasard du graphe comme sélection simulée
            all_nodes = list(G.nodes(data=True))
            simulated_count = min(8, len(all_nodes))  # Simuler ~8 entités sélectionnées comme dans les logs

            import random
            random.seed(42)  # Pour avoir des résultats consistants
            selected_nodes_sample = random.sample(all_nodes, simulated_count)

            for node_id, node_data in selected_nodes_sample:
                entity_names.append(node_data.get('entity_name', str(node_id)))

            logger.info(f"🔍 Using simulated entities for demo ({simulated_count} nodes): {entity_names[:3]}... (showing first 3)")
            logger.info(f"🎯 This simulates GraphRAG finding: 'Using {simulated_count} entites, 3 communities, {simulated_count*4} relations'")

        # Trouver les nœuds correspondants dans le graphe principal
        selected_nodes = []
        selected_node_ids = set()

        for node_id, node_data in G.nodes(data=True):
            node_name = node_data.get('entity_name', node_id)

            # Vérifier si ce nœud correspond à une entité GraphRAG
            matches = any(
                entity_name.lower() in node_name.lower() or
                node_name.lower() in entity_name.lower()
                for entity_name in entity_names
                if entity_name
            )

            if matches:
                selected_node_ids.add(node_id)

                # Pour les nœuds GraphRAG, utiliser le nom de l'entité comme ID pour correspondre aux relations
                graphrag_node_id = clean_quotes(node_name)

                node_obj = {
                    'id': graphrag_node_id,  # Utiliser le nom de l'entité pour les relations GraphRAG
                    'label': clean_quotes(node_name),  # Frontend expects 'label' field
                    'type': clean_quotes(node_data.get('entity_type', 'Entity')),  # Frontend expects 'type' field
                    'labels': [clean_quotes(node_data.get('entity_type', 'Entity'))],
                    'properties': {
                        'name': clean_quotes(node_name),
                        'description': clean_quotes(node_data.get('description', '')),
                        'entity_type': clean_quotes(node_data.get('entity_type', 'Entity')),
                        'graphrag_node': True,  # Marquer comme nœud GraphRAG
                        'original_neo4j_id': clean_quotes(str(node_id)),  # Garder l'ID original pour référence
                        # Book context for chunk retrieval (required by frontend)
                        'book_id': book_id,
                        'book_dir': book_id  # Alias for compatibility
                    },
                    'degree': G.degree(node_id),
                    'centrality_score': G.degree(node_id)
                }
                selected_nodes.append(node_obj)

        # Extraire les vraies relations GraphRAG depuis debug_info pour le chemin de traversée
        selected_relationships = []

        # Priorité 1: Utiliser les vraies relations GraphRAG du debug_info
        graphrag_relationships = debug_info.get('processing_phases', {}).get('relationship_mapping', {}).get('relationships', [])

        if graphrag_relationships:
            logger.info(f"🎯 Extracting {len(graphrag_relationships)} GraphRAG traversal relationships for book: {book_id}")

            # Enrichir avec les métadonnées GraphML directes - AVEC FILTRAGE PAR LIVRE
            graphml_enriched_relationships = enrich_relationships_with_graphml(G, graphrag_relationships, book_id)

            # Collecter toutes les entités mentionnées dans les relations GraphRAG FILTRÉES
            graphrag_entities = set()
            for rel_data in graphml_enriched_relationships:
                source_clean = clean_quotes(str(rel_data.get('source', '')))
                target_clean = clean_quotes(str(rel_data.get('target', '')))
                if source_clean:
                    graphrag_entities.add(source_clean)
                if target_clean:
                    graphrag_entities.add(target_clean)

            logger.info(f"🔍 Found {len(graphrag_entities)} unique entities in filtered GraphRAG relationships for book: {book_id}")

            # Ajouter validation supplémentaire pour éviter les entités "LIVRE_*" inappropriées
            invalid_entities = set()
            if book_id:
                for entity in list(graphrag_entities):
                    # Filtrer les entités "LIVRE_*" qui ne correspondent pas au book_id actuel
                    if entity.startswith('LIVRE_') and book_id not in entity.lower():
                        invalid_entities.add(entity)
                        logger.warning(f"🚫 Filtering out cross-book entity: '{entity}' (current book: {book_id})")

                # Retirer les entités invalides
                graphrag_entities -= invalid_entities
                logger.info(f"✅ After cross-book validation: {len(graphrag_entities)} entities remain for book: {book_id}")

            # Créer des nœuds enrichis avec métadonnées GraphML pour toutes les entités GraphRAG - AVEC FILTRAGE PAR LIVRE
            existing_node_names = {node['properties']['name'] for node in selected_nodes}
            enriched_graphrag_nodes = enrich_nodes_with_graphml(G, list(graphrag_entities), book_id)

            for enriched_node in enriched_graphrag_nodes:
                entity_name = enriched_node['name']
                if entity_name not in existing_node_names:
                    # Validation supplémentaire pour éviter les entités cross-book
                    if book_id and entity_name.startswith('LIVRE_') and book_id not in entity_name.lower():
                        logger.warning(f"🚫 Skipping cross-book synthetic node creation: '{entity_name}' (current book: {book_id})")
                        continue

                    # Créer un nœud synthétique enrichi avec métadonnées GraphML
                    synthetic_node = {
                        'id': entity_name,
                        'label': entity_name,
                        'type': 'GraphRAG_Entity',
                        'labels': ['GraphRAG_Entity'],
                        'properties': {
                            'name': entity_name,
                            'description': enriched_node.get('description', f'Entity from GraphRAG traversal: {entity_name}'),
                            'entity_type': enriched_node.get('entity_type', 'GraphRAG_Entity'),
                            'graphrag_node': True,
                            'synthetic': True,
                            'book_id': book_id,  # Track which book this entity belongs to
                            # GraphML enriched metadata
                            'clusters': enriched_node.get('clusters', ''),
                            'source_chunks': enriched_node.get('source_chunks', ''),
                            'has_graphml_metadata': enriched_node.get('has_graphml_metadata', False)
                        },
                        'degree': 1,
                        'centrality_score': 1
                    }
                    selected_nodes.append(synthetic_node)
                    existing_node_names.add(entity_name)
                    metadata_status = "with GraphML metadata" if enriched_node.get('has_graphml_metadata') else "basic"
                    logger.info(f"➕ Created synthetic GraphRAG node: {entity_name} ({metadata_status}) for book: {book_id}")

            # Maintenant créer les relations GraphRAG et s'assurer que tous les nœuds existent
            for rel_data in graphml_enriched_relationships:
                # Nettoyage des guillemets pour source et target
                source_clean = clean_quotes(str(rel_data.get('source', '')))
                target_clean = clean_quotes(str(rel_data.get('target', '')))

                # Validation supplémentaire pour éviter les relations cross-book
                if book_id:
                    if ((source_clean.startswith('LIVRE_') and book_id not in source_clean.lower()) or
                        (target_clean.startswith('LIVRE_') and book_id not in target_clean.lower())):
                        logger.warning(f"🚫 Skipping cross-book relationship: '{source_clean}' -> '{target_clean}' (current book: {book_id})")
                        continue

                # Créer des nœuds synthétiques pour source et target s'ils n'existent pas
                for entity_name in [source_clean, target_clean]:
                    if entity_name and entity_name not in existing_node_names:
                        # Validation supplémentaire pour éviter les entités cross-book
                        if book_id and entity_name.startswith('LIVRE_') and book_id not in entity_name.lower():
                            logger.warning(f"🚫 Skipping cross-book synthetic node: '{entity_name}' (current book: {book_id})")
                            continue

                        synthetic_node = {
                            'id': entity_name,
                            'label': entity_name,
                            'type': 'GraphRAG_Entity',
                            'labels': ['GraphRAG_Entity'],
                            'properties': {
                                'name': entity_name,
                                'description': f'Entity from GraphRAG relationship: {entity_name}',
                                'entity_type': 'GraphRAG_Entity',
                                'graphrag_node': True,
                                'synthetic': True,
                                'book_id': book_id  # Track which book this entity belongs to
                            },
                            'degree': 1,
                            'centrality_score': 1
                        }
                        selected_nodes.append(synthetic_node)
                        existing_node_names.add(entity_name)
                        logger.info(f"➕ Created synthetic node for relationship entity: {entity_name} (book: {book_id})")

                # Enhanced relationship object with GraphML metadata
                rel_obj = {
                    'id': f"{source_clean}_{target_clean}",
                    'type': clean_quotes(rel_data.get('type', rel_data.get('description', 'GRAPHRAG_PATH'))),
                    'source': source_clean,
                    'target': target_clean,
                    'properties': {
                        # Original GraphRAG data
                        'description': clean_quotes(rel_data.get('description', 'GraphRAG traversal path')),
                        'weight': float(rel_data.get('weight') or 1.0),
                        'traversal_order': rel_data.get('traversal_order'),
                        'graphrag_path': True,
                        'is_community_link': rel_data.get('is_community_link', False),

                        # Enhanced GraphML metadata
                        'graphml_weight': rel_data.get('graphml_weight', rel_data.get('weight', 1.0)),
                        'graphml_description': rel_data.get('graphml_description', ''),
                        'graphml_source_chunks': rel_data.get('graphml_source_chunks', ''),
                        'graphml_order': rel_data.get('graphml_order', 0),
                        'has_graphml_metadata': rel_data.get('has_graphml_metadata', False),

                        # Book context for chunk retrieval (required by frontend)
                        'book_id': book_id,
                        'book_dir': book_id  # Alias for compatibility
                    }
                }
                selected_relationships.append(rel_obj)
        else:
            # Fallback: Relations locales entre nœuds sélectionnés pour démonstration
            logger.warning("⚠️ No GraphRAG relationships found, using local graph relationships as fallback")
            for source, target, edge_data in G.edges(data=True):
                if source in selected_node_ids and target in selected_node_ids:
                    rel_obj = {
                        'id': f"{clean_quotes(str(source))}_{clean_quotes(str(target))}",
                        'type': clean_quotes(edge_data.get('weight_label', 'RELATED')),
                        'source': clean_quotes(str(source)),
                        'target': clean_quotes(str(target)),
                        'properties': {
                            'description': clean_quotes(edge_data.get('weight_label', 'Related to')),
                            'weight': float(edge_data.get('weight') or 1.0),
                            # Book context for chunk retrieval (required by frontend)
                            'book_id': book_id,
                            'book_dir': book_id  # Alias for compatibility
                        }
                    }
                    selected_relationships.append(rel_obj)

        logger.info(f"✅ Extracted {len(selected_nodes)} nodes and {len(selected_relationships)} relationships for GraphRAG")

        return {
            "nodes": selected_nodes,
            "relationships": selected_relationships
        }

    except Exception as e:
        logger.error(f"❌ Error extracting selected nodes: {e}")
        return {"nodes": [], "relationships": []}

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
            # Query CENTERED around books (Principle #2: Books as core entities)
            # Strategy: Load ALL books + their immediate neighbors to create book-centric view
            # This ensures the graph visualization is always centered around books
            query = """
                // PRINCIPLE #2: Books as core entities - CENTER graph around books
                // Step 1: Get all books with their degree
                MATCH (book:BOOK)
                WITH collect({node: book, degree: SIZE([(book)--() | 1]), isBook: true}) as allBooks

                // Step 2: Get distinct neighbors of books (1-hop from books)
                MATCH (book:BOOK)-[]-(neighbor)
                WHERE NOT neighbor:BOOK
                WITH allBooks, neighbor, SIZE([(neighbor)--() | 1]) as neighborDegree
                ORDER BY neighborDegree DESC

                // Step 3: Collect distinct neighbors and limit them
                WITH allBooks, collect(DISTINCT {node: neighbor, degree: neighborDegree, isBook: false}) as neighbors
                // FIX: Explicitly calculate bookCount in same WITH to avoid implicit grouping
                WITH allBooks, neighbors, size(allBooks) as bookCount,
                     CASE WHEN size(allBooks) < $limit
                          THEN neighbors[0..($limit - size(allBooks))]
                          ELSE []
                     END as limitedNeighbors

                // Step 4: Combine books + limited neighbors
                WITH allBooks + limitedNeighbors as allNodes
                UNWIND allNodes as item

                // Step 5: Return with books first
                RETURN item.node as n, item.degree as degree
                ORDER BY
                    CASE WHEN item.isBook THEN 0 ELSE 1 END,
                    degree DESC
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

            # Add GraphML enrichment for better metadata
            enriched_relationships = relationships
            book_id = request.args.get('book_id') or (request.get_json() or {}).get('book_id')

            if book_id:
                logger.info(f"🔍 Enriching relationships with GraphML metadata for book: {book_id}")
                try:
                    # Load GraphML file for the specified book
                    base_path = get_book_data_base_path()
                    graph_path = Path(base_path) / book_id / "graph_chunk_entity_relation.graphml"

                    if graph_path.exists():
                        import networkx as nx
                        G = nx.read_graphml(str(graph_path))

                        # Convert Neo4j relationships to format expected by enrich_relationships_with_graphml
                        # We need to fetch node labels for better matching
                        node_labels = {}
                        for node_id in node_ids:
                            try:
                                node_query = "MATCH (n) WHERE elementId(n) = $node_id RETURN n.name as name, elementId(n) as id"
                                node_result = session.run(node_query, node_id=node_id)
                                for node_record in node_result:
                                    node_labels[node_record['id']] = node_record['name'] or node_record['id']
                            except:
                                node_labels[node_id] = node_id

                        graphrag_relationships = []
                        for rel in relationships:
                            source_label = node_labels.get(rel['source'], rel['source'])
                            target_label = node_labels.get(rel['target'], rel['target'])

                            graphrag_relationships.append({
                                'source': source_label,
                                'target': target_label,
                                'relation': rel['type'],
                                'weight': rel['properties'].get('weight', 1.0),
                                'description': rel['properties'].get('description', ''),
                                'id': rel['id']
                            })

                        # Apply GraphML enrichment
                        enriched_graphrag_rels = enrich_relationships_with_graphml(G, graphrag_relationships, book_id)

                        # Merge GraphML metadata back into Neo4j relationship format
                        for i, rel in enumerate(relationships):
                            # Find corresponding enriched relationship
                            source_label = node_labels.get(rel['source'], rel['source'])
                            target_label = node_labels.get(rel['target'], rel['target'])

                            # Default to no GraphML metadata
                            rel['properties']['has_graphml_metadata'] = False

                            # Look for matching enriched relationship
                            for enriched_rel in enriched_graphrag_rels:
                                if (enriched_rel.get('source') == source_label and
                                    enriched_rel.get('target') == target_label and
                                    enriched_rel.get('relation') == rel['type']):

                                    # Merge GraphML metadata into properties
                                    if enriched_rel.get('has_graphml_metadata'):
                                        rel['properties'].update({
                                            'graphml_weight': enriched_rel.get('graphml_weight'),
                                            'graphml_description': enriched_rel.get('graphml_description'),
                                            'graphml_source_chunks': enriched_rel.get('graphml_source_chunks'),
                                            'graphml_order': enriched_rel.get('graphml_order'),
                                            'has_graphml_metadata': True,
                                            'description': enriched_rel.get('graphml_description') or rel['properties'].get('description', ''),
                                            'weight': enriched_rel.get('graphml_weight', rel['properties'].get('weight', 1.0))
                                        })
                                    else:
                                        rel['properties']['has_graphml_metadata'] = False
                                    break

                        logger.info(f"✅ Successfully merged GraphML metadata into {len(relationships)} Neo4j relationships")
                    else:
                        logger.warning(f"GraphML file not found for book {book_id}: {graph_path}")
                        # Set all relationships to have no GraphML metadata
                        for rel in relationships:
                            rel['properties']['has_graphml_metadata'] = False
                except Exception as e:
                    logger.warning(f"GraphML enrichment failed for book {book_id}: {e}")
                    # Fall back to original relationships if enrichment fails
                    for rel in relationships:
                        rel['properties']['has_graphml_metadata'] = False
            else:
                # No book_id provided, set all relationships to have no GraphML metadata
                for rel in relationships:
                    rel['properties']['has_graphml_metadata'] = False

            return jsonify({
                'success': True,
                'relationships': relationships,  # Now contains merged GraphML metadata
                'count': len(relationships),
                'input_nodes': len(node_ids),
                'limit_applied': limit,
                'filtered': len(relationships) >= limit,
                'graphml_enriched': book_id is not None
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
    Endpoint pour tester le GraphRAG local avec vrai intercepteur et données de livres
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
        # Utiliser le GraphRAG local avec intercepteur et données de livres
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
    debug_mode = data.get('debug_mode', True)  # Always enable debug mode for interceptor
    book_id = data.get('book_id', None)

    logger.info(f"📝 Received query: '{query}', mode: {mode}, book_id: {book_id}")
    logger.info(f"🔍 Request data keys: {list(data.keys())}")
    logger.info(f"📚 Book selection validation - book_id: '{book_id}' (type: {type(book_id)})")

    # Validation supplémentaire du book_id
    if book_id:
        available_books = list_available_books()
        if book_id not in available_books:
            logger.warning(f"⚠️ Book '{book_id}' not found in available books: {available_books}")
        else:
            logger.info(f"✅ Book '{book_id}' validated successfully")

    if not query:
        return jsonify({
            'success': False,
            'error': 'Query is required'
        }), 400

    try:
        # Query local GraphRAG with book data
        graphrag_data = {}
        try:
            graphrag_instance = get_local_graphrag(book_id or "a_rebours_huysmans")
            if graphrag_instance:
                logger.info(f"🔍 Using local GraphRAG for query: '{query}' on book: {book_id}")
                start_time = time.time()
                result = graphrag_instance.query(query, param=QueryParam(mode=mode))
                elapsed_time = time.time() - start_time

                graphrag_data = {
                    'answer': result,
                    'mode': mode,
                    'processing_time': elapsed_time,
                    'source': 'local_graphrag',
                    'book_id': book_id or "a_rebours_huysmans"
                }
                logger.info(f"✅ Local GraphRAG response received: {len(result)} chars in {elapsed_time:.2f}s")
            else:
                raise Exception("Local GraphRAG not available")
        except Exception as e:
            logger.error(f"Local GraphRAG failed: {e}")
            graphrag_data = {
                'answer': f'Error processing query: {str(e)}',
                'mode': mode,
                'source': 'error',
                'book_id': book_id or "a_rebours_huysmans"
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
            'timestamp': datetime.utcnow().isoformat(),
            'query_id': ""  # Will be set after provenance is saved
        }

        # Save query provenance to Neo4j (Feature: 001-interactive-graphrag-refinement)
        query_id = ""
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            query_id = loop.run_until_complete(
                graphrag_interceptor.save_query_provenance(
                    question=query,
                    answer_text=graphrag_data.get('answer', ''),
                    mode=mode,
                    user_id="default_user"
                )
            )
            loop.close()
            if query_id:
                logger.info(f"✅ Saved query provenance: {query_id}")
                result['query_id'] = query_id  # Add to response
        except Exception as e:
            logger.warning(f"⚠️ Could not save query provenance: {e}")

        # Always add debug information for node animation
        try:
            debug_info = graphrag_interceptor.get_real_debug_info()

            # Si pas de données capturées par l'intercepteur, créer des données simulées basées sur les logs
            if not debug_info.get('processing_phases', {}).get('entity_selection', {}).get('entities'):
                # Créer des entités factices basées sur les logs nano-graphrag "Using X entites..."
                debug_info = create_simulated_debug_info(graphrag_data.get('processing_time', 2.0))

            result['debug_info'] = debug_info

            # IMPORTANT: Ajouter les nœuds et relations GraphRAG pour l'animation incrémentale
            try:
                selected_graph_data = extract_selected_nodes_from_graphrag(book_id or "a_rebours_huysmans", debug_info)
                result['selected_nodes'] = selected_graph_data['nodes']
                result['selected_relationships'] = selected_graph_data['relationships']
                logger.info(f"Selected graph data: {len(selected_graph_data['nodes'])} nodes, {len(selected_graph_data['relationships'])} relationships")
            except Exception as extract_e:
                logger.warning(f"Could not extract selected nodes: {extract_e}")
                result['selected_nodes'] = []
                result['selected_relationships'] = []

            logger.info(f"Debug info captured for animation: {len(debug_info.get('processing_phases', {}).get('entity_selection', {}).get('entities', []))} entities")
        except Exception as e:
            logger.warning(f"Debug info not available: {e}")
            # Créer des données simulées basées sur les logs "Using X entites..."
            result['debug_info'] = create_simulated_debug_info(graphrag_data.get('processing_time', 2.0))
            result['selected_nodes'] = []
            result['selected_relationships'] = []

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
                        'label': node_data.get('entity_name', node_id),  # Frontend expects 'label' field
                        'type': node_data.get('entity_type', 'Entity'),  # Frontend expects 'type' field
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
                            'weight': float(edge_data.get('weight') or 1.0)
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

                # Always collect debug_info for entity aggregation in multi-book mode
                debug_info = graphrag_interceptor.get_real_debug_info()

                # NEW: Extract enriched nodes/relationships with GraphML metadata (like single-book mode)
                # This adds book_id, book_dir, graphml_source_chunks for chunk loading in EntityDetailModal
                book_selected_nodes = []
                book_selected_relationships = []
                try:
                    selected_graph_data = extract_selected_nodes_from_graphrag(book_id, debug_info)
                    book_selected_nodes = selected_graph_data.get('nodes', [])
                    book_selected_relationships = selected_graph_data.get('relationships', [])
                    logger.info(f"📦 {book_id}: Enriched {len(book_selected_nodes)} nodes, {len(book_selected_relationships)} relationships with GraphML chunks")
                except Exception as enrich_error:
                    logger.warning(f"⚠️ Could not enrich nodes for {book_id}: {enrich_error}")

                # Initialize empty lists for tracking
                entities = []
                relationships = []
                communities = []

                book_result = {
                    'book_id': book_id,
                    'answer': result,
                    'processing_time': book_processing_time,
                    'debug_info': debug_info,
                    'selected_nodes': book_selected_nodes,
                    'selected_relationships': book_selected_relationships
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

        # Enrich cross-book entities with Neo4j relationships
        logger.info("🔗 Enriching cross-book relationships with Neo4j data")
        cross_book_relationships = _get_cross_book_neo4j_relationships(aggregated_entities)

        # Add cross-book relationships to aggregated data
        for cross_rel in cross_book_relationships:
            rel_key = f"{cross_rel.get('source')}--{cross_rel.get('target')}"
            if rel_key not in aggregated_relationships:
                aggregated_relationships[rel_key] = {
                    **cross_rel,
                    'books': cross_rel.get('books', []),
                    'found_in': cross_rel.get('books', []),
                    'is_cross_book': True
                }

        logger.info(f"✅ Added {len(cross_book_relationships)} cross-book Neo4j relationships")

        # NEW: Aggregate enriched nodes and relationships from all book_results
        # These have book_id, book_dir, and graphml_source_chunks for EntityDetailModal chunk loading
        enriched_nodes_map = {}  # Deduplicate by node id
        enriched_relationships_map = {}  # Deduplicate by source--target

        for book_result in all_results:
            if 'error' in book_result:
                continue

            # Aggregate enriched nodes
            for node in book_result.get('selected_nodes', []):
                node_id = node.get('id', '')
                if node_id:
                    if node_id not in enriched_nodes_map:
                        # First occurrence - store the node
                        enriched_nodes_map[node_id] = {**node}
                        # Ensure books array exists
                        if 'properties' not in enriched_nodes_map[node_id]:
                            enriched_nodes_map[node_id]['properties'] = {}
                        enriched_nodes_map[node_id]['properties']['books'] = [book_result['book_id']]
                    else:
                        # Node exists in multiple books - add to books array
                        existing_books = enriched_nodes_map[node_id].get('properties', {}).get('books', [])
                        if book_result['book_id'] not in existing_books:
                            existing_books.append(book_result['book_id'])
                            enriched_nodes_map[node_id]['properties']['books'] = existing_books

            # Aggregate enriched relationships
            for rel in book_result.get('selected_relationships', []):
                source = rel.get('source', '')
                target = rel.get('target', '')
                rel_key = f"{source}--{target}"
                if source and target:
                    if rel_key not in enriched_relationships_map:
                        # First occurrence - store the relationship with its GraphML chunks
                        enriched_relationships_map[rel_key] = {**rel}
                        # Ensure books array exists
                        if 'properties' not in enriched_relationships_map[rel_key]:
                            enriched_relationships_map[rel_key]['properties'] = {}
                        enriched_relationships_map[rel_key]['properties']['books'] = [book_result['book_id']]
                    else:
                        # Relationship exists in multiple books - add to books array
                        existing_books = enriched_relationships_map[rel_key].get('properties', {}).get('books', [])
                        if book_result['book_id'] not in existing_books:
                            existing_books.append(book_result['book_id'])
                            enriched_relationships_map[rel_key]['properties']['books'] = existing_books

        logger.info(f"📦 Aggregated {len(enriched_nodes_map)} enriched nodes, {len(enriched_relationships_map)} enriched relationships with GraphML chunks")

        # Step 1: Build comprehensive entity mapping including all entities referenced in relationships
        entity_id_mapping = {}  # Map original IDs to final IDs
        all_referenced_entities = set()

        # First pass: collect all entities from relationships to ensure we include book entities
        for rel in aggregated_relationships.values():
            source_id = rel.get('source', '')
            target_id = rel.get('target', '')
            if source_id:
                all_referenced_entities.add(source_id)
            if target_id:
                all_referenced_entities.add(target_id)

        logger.info(f"📚 Found {len(all_referenced_entities)} entities referenced in relationships")

        # Create entity mapping for all entities (both aggregated and referenced in relationships)
        for entity_id in all_referenced_entities:
            # Check if this entity is in our aggregated entities
            if entity_id in aggregated_entities:
                entity = aggregated_entities[entity_id]
                entity_name = entity.get('name', entity_id)
                final_id = entity_name if entity_name else entity_id
            else:
                # This is a referenced entity (likely a book) not in aggregated entities
                # Create a minimal entity for it
                final_id = entity_id
                # Add to aggregated entities for consistency
                aggregated_entities[entity_id] = {
                    'id': entity_id,
                    'name': entity_id,
                    'type': 'Livres' if entity_id.startswith('LIVRE_') else 'Entity',
                    'description': f'Referenced entity: {entity_id}',
                    'books': [entity_id] if entity_id.startswith('LIVRE_') else [],
                    'found_in': []
                }

            entity_id_mapping[entity_id] = final_id

        logger.info(f"🔗 Comprehensive entity mapping created with {len(entity_id_mapping)} mappings")

        # Step 2: Convert aggregated entities to selected_nodes format
        selected_nodes = []
        for entity in aggregated_entities.values():
            entity_id = entity.get('id', '')
            final_id = entity_id_mapping.get(entity_id, entity_id)

            node_obj = {
                'id': final_id,
                'label': final_id,
                'type': entity.get('type', 'Entity'),
                'labels': [entity.get('type', 'Entity')],
                'properties': {
                    'name': final_id,
                    'description': entity.get('description', ''),
                    'books': entity.get('books', []),
                    'found_in': entity.get('found_in', []),
                    'original_id': entity_id
                },
                'degree': len(entity.get('books', [])),
                'centrality_score': len(entity.get('books', []))
            }
            selected_nodes.append(node_obj)

        logger.info(f"🔗 Entity ID mapping created with {len(entity_id_mapping)} mappings")

        # Debug: show first few mappings to understand the pattern
        if entity_id_mapping:
            sample_mappings = list(entity_id_mapping.items())[:5]
            logger.info(f"📝 Sample mappings: {sample_mappings}")

        # Create a set of valid final entity IDs for validation
        valid_entity_ids = {node['id'] for node in selected_nodes}
        logger.info(f"✅ All entities included: {len(valid_entity_ids)} total entities")

        # Step 3: Convert relationships with comprehensive mapping
        selected_relationships = []
        orphaned_relations = 0

        for rel in aggregated_relationships.values():
            source_id = rel.get('source', '')
            target_id = rel.get('target', '')

            # Map the source and target IDs to final entity IDs
            mapped_source = entity_id_mapping.get(source_id, source_id)
            mapped_target = entity_id_mapping.get(target_id, target_id)

            # Now all referenced entities should be in our valid set
            if mapped_source in valid_entity_ids and mapped_target in valid_entity_ids:
                rel_obj = {
                    'id': f"{mapped_source}_{mapped_target}",
                    'type': rel.get('description', 'RELATED'),
                    'source': mapped_source,
                    'target': mapped_target,
                    'properties': {
                        'description': rel.get('description', 'Related to'),
                        'books': rel.get('books', []),
                        'found_in': rel.get('found_in', []),
                        'weight': rel.get('weight', 1.0),
                        'original_source': source_id,
                        'original_target': target_id
                    }
                }
                selected_relationships.append(rel_obj)
            else:
                orphaned_relations += 1
                logger.warning(f"⚠️ Still orphaned after comprehensive mapping: {source_id} -> {target_id} (mapped: {mapped_source} -> {mapped_target})")

        logger.info(f"✅ Relationship mapping completed: {len(selected_relationships)} valid relationships, {orphaned_relations} orphaned")

        # Final validation: ensure all relationships reference existing entities
        final_relationships = []
        for rel in selected_relationships:
            if rel['source'] in valid_entity_ids and rel['target'] in valid_entity_ids:
                final_relationships.append(rel)
            else:
                logger.warning(f"⚠️ Final validation failed for relation: {rel['source']} -> {rel['target']}")

        selected_relationships = final_relationships
        logger.info(f"🎯 Final result: {len(selected_nodes)} nodes, {len(selected_relationships)} relationships (no orphans)")

        # NEW: Use enriched nodes/relationships which have book_dir and graphml_source_chunks
        # This enables EntityDetailModal chunk loading (Principle #5: End-to-end interpretability)
        if enriched_nodes_map:
            selected_nodes = list(enriched_nodes_map.values())
            logger.info(f"📦 Using {len(selected_nodes)} enriched nodes with GraphML metadata for chunk loading")
        if enriched_relationships_map:
            selected_relationships = list(enriched_relationships_map.values())
            logger.info(f"📦 Using {len(selected_relationships)} enriched relationships with graphml_source_chunks for chunk loading")

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
            'selected_nodes': selected_nodes,  # Add for visualization
            'selected_relationships': selected_relationships,  # Add for visualization
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

@app.route('/data/upload-local', methods=['POST'])
def upload_local_data():
    """
    Upload local book data to Railway volume (development helper)
    """
    try:
        # Only work if we have a Railway volume
        volume_path = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH')
        if not volume_path:
            return jsonify({
                'success': False,
                'error': 'No Railway volume detected - this endpoint only works on Railway'
            }), 400

        logger.info("📤 Uploading local book data to Railway volume...")

        # Create sample book data for testing
        success = create_sample_book_data()

        if success:
            available_books = list_available_books()
            return jsonify({
                'success': True,
                'message': f'Book data uploaded successfully to {volume_path}',
                'volume_path': volume_path,
                'available_books': available_books,
                'book_count': len(available_books)
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to upload book data'
            }), 500

    except Exception as e:
        logger.error(f"Error uploading local data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/debug/env', methods=['GET'])
def debug_env():
    """Debug endpoint to check environment variables"""
    try:
        volume_path = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH')
        volume_contents = []
        if volume_path and Path(volume_path).exists():
            volume_contents = [str(item) for item in Path(volume_path).iterdir()]

        return jsonify({
            'railway_volume_path': volume_path,
            'volume_mount_path': os.environ.get('RAILWAY_VOLUME_MOUNT_PATH'),
            'base_path': get_book_data_base_path(),
            'volume_exists': Path(volume_path).exists() if volume_path else False,
            'volume_contents': volume_contents
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.teardown_appcontext
def cleanup(error):
    """Cleanup on app context teardown"""
    pass

# Register book endpoints
register_books_endpoints(app)

# Register provenance endpoints
register_provenance_endpoints(app)

@app.route('/chunks/find/<chunk_id>', methods=['GET'])
def find_chunk(chunk_id):
    """
    Find a chunk by ID without knowing which book it belongs to.
    Searches all books until the chunk is found.

    Args:
        chunk_id: The ID of the chunk (e.g., 'chunk-5e8ee10d576da6545460549a4a8f8d6f')

    Returns:
        JSON with chunk content and the book it was found in
    """
    try:
        # Try Neo4j first
        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    result = session.run(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        OPTIONAL MATCH (b:BOOK)-[:HAS_CHUNK]->(c)
                        RETURN c.content as content,
                               c.tokens as tokens,
                               c.chunk_order_index as chunk_order_index,
                               c.full_doc_id as full_doc_id,
                               b.book_dir as book_dir
                        """,
                        chunk_id=chunk_id
                    )
                    record = result.single()
                    if record and record['content']:
                        book_dir = record['book_dir'] or 'unknown'
                        logger.info(f"📚 Found chunk {chunk_id} in Neo4j (book: {book_dir})")
                        return jsonify({
                            'success': True,
                            'found_in': book_dir,
                            'chunk_id': chunk_id,
                            'content': record['content'],
                            'tokens': record['tokens'] or 0,
                            'chunk_order_index': record['chunk_order_index'] or 0,
                            'full_doc_id': record['full_doc_id'] or '',
                            'source': 'neo4j'
                        })
            except Exception as neo_error:
                logger.warning(f"Neo4j chunk search failed: {neo_error}")

        # Search all books
        available_books = list_available_books()
        for book_dir in available_books:
            try:
                chunks_data = load_chunks_file(book_dir)
                if chunk_id in chunks_data:
                    chunk_content = chunks_data[chunk_id]
                    logger.info(f"✅ Found chunk {chunk_id} in {book_dir}")
                    return jsonify({
                        'success': True,
                        'found_in': book_dir,
                        'chunk_id': chunk_id,
                        'content': chunk_content.get('content', ''),
                        'tokens': chunk_content.get('tokens', 0),
                        'chunk_order_index': chunk_content.get('chunk_order_index', 0),
                        'full_doc_id': chunk_content.get('full_doc_id', ''),
                        'source': 'file'
                    })
            except FileNotFoundError:
                continue

        return jsonify({
            'success': False,
            'error': f'Chunk not found in any book: {chunk_id}',
            'searched_books': available_books
        }), 404

    except Exception as e:
        logger.error(f"Error finding chunk {chunk_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/chunks/<book_id>/<chunk_id>', methods=['GET'])
def get_chunk_content(book_id, chunk_id):
    """
    Get the text content of a specific chunk for end-to-end traceability

    Args:
        book_id: The ID of the book (e.g., 'a_rebours_huysmans')
        chunk_id: The ID of the chunk (e.g., 'chunk-5e8ee10d576da6545460549a4a8f8d6f')

    Returns:
        JSON with chunk content, metadata, and traceability info
    """
    try:
        # Try Neo4j first (chunks stored as nodes)
        driver = get_neo4j_driver()
        if driver:
            try:
                with driver.session() as session:
                    # Query Chunk node by ID and find its actual book via HAS_CHUNK relationship
                    result = session.run(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        OPTIONAL MATCH (b:BOOK)-[:HAS_CHUNK]->(c)
                        RETURN c.content as content,
                               c.tokens as tokens,
                               c.chunk_order_index as chunk_order_index,
                               c.full_doc_id as full_doc_id,
                               b.book_dir as actual_book_dir,
                               b.id as actual_book_id
                        """,
                        chunk_id=chunk_id
                    )

                    record = result.single()
                    if record and record['content']:
                        # Use actual book from relationship, or search files if not found
                        actual_book = record['actual_book_dir'] or record['actual_book_id']
                        if actual_book and actual_book.startswith('LIVRE_'):
                            # Convert LIVRE_Chien Blanc -> chien_blanc_gary by searching files
                            actual_book = None  # Will fallback to file search below

                        # If no book found in Neo4j, search files to find the right one
                        found_in_book = actual_book
                        if not found_in_book:
                            available_books = list_available_books()
                            for book_dir in available_books:
                                try:
                                    chunks_data = load_chunks_file(book_dir)
                                    if chunk_id in chunks_data:
                                        found_in_book = book_dir
                                        logger.info(f"✅ Found chunk {chunk_id} belongs to {book_dir}")
                                        break
                                except FileNotFoundError:
                                    continue

                        logger.info(f"📚 Retrieved chunk {chunk_id} from Neo4j: {len(record['content'] or '')} chars, found_in: {found_in_book}")
                        return jsonify({
                            'success': True,
                            'book_id': book_id,  # Requested book
                            'found_in': found_in_book or book_id,  # Actual book where chunk belongs
                            'chunk_id': chunk_id,
                            'content': record['content'] or '',
                            'tokens': record['tokens'] or 0,
                            'chunk_order_index': record['chunk_order_index'] or 0,
                            'full_doc_id': record['full_doc_id'] or '',
                            'source': 'neo4j',
                            'traceability': {
                                'pipeline': ['Source Text', 'Text Chunking', 'GraphRAG Entity Extraction', 'Neo4j Storage', '3D Visualization'],
                                'source_type': 'neo4j_chunk_node',
                                'processing_chain': 'Book → Chunk → GraphRAG → Neo4j → 3D Graph'
                            }
                        })
            except Exception as neo_error:
                logger.warning(f"Neo4j chunk lookup failed, trying file fallback: {neo_error}")

        # Fallback: Try loading from cached file
        # First try the specified book_id
        found_in_book = None
        chunk_content = None

        try:
            chunks_data = load_chunks_file(book_id)
            if chunk_id in chunks_data:
                chunk_content = chunks_data[chunk_id]
                found_in_book = book_id
        except FileNotFoundError:
            pass  # Will try other books

        # If not found in specified book, search ALL books (for inter-book entities)
        if chunk_content is None:
            logger.info(f"🔍 Chunk {chunk_id} not found in {book_id}, searching all books...")
            available_books = list_available_books()
            for other_book in available_books:
                if other_book == book_id:
                    continue  # Already tried this one
                try:
                    chunks_data = load_chunks_file(other_book)
                    if chunk_id in chunks_data:
                        chunk_content = chunks_data[chunk_id]
                        found_in_book = other_book
                        logger.info(f"✅ Found chunk {chunk_id} in {other_book} (not in requested {book_id})")
                        break
                except FileNotFoundError:
                    continue

        # If still not found, return 404
        if chunk_content is None:
            return jsonify({
                'success': False,
                'error': f'Chunk not found in Neo4j or any book files: {chunk_id}',
                'book_id': book_id,
                'searched_books': list_available_books()
            }), 404

        # Enhance with traceability metadata
        result = {
            'success': True,
            'book_id': book_id,  # Original requested book
            'found_in': found_in_book,  # Actual book where chunk was found
            'chunk_id': chunk_id,
            'content': chunk_content.get('content', ''),
            'tokens': chunk_content.get('tokens', 0),
            'chunk_order_index': chunk_content.get('chunk_order_index', 0),
            'full_doc_id': chunk_content.get('full_doc_id', ''),
            'source': 'file',
            'traceability': {
                'pipeline': ['Source Text', 'Text Chunking', 'GraphRAG Entity Extraction', 'GraphML Generation', 'Neo4j Import', '3D Visualization'],
                'source_type': 'original_text',
                'processing_chain': f'Book → Chunk → GraphRAG → GraphML → Neo4j → 3D Graph'
            }
        }

        logger.info(f"📚 Retrieved chunk {chunk_id} from file {found_in_book}: {len(chunk_content.get('content', ''))} chars")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error fetching chunk {chunk_id} from {book_id}: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Ensure book data is available
        ensure_book_data_available()

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