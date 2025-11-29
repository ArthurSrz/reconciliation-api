"""
Books endpoints for the Reconciliation API
Handles book data management and listing
"""

from flask import jsonify, request
import logging
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)

# Local book data functions with Railway volume support
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
    """Get path to local book data"""
    base_path = get_book_data_base_path()
    book_path = Path(base_path) / book_id
    if book_path.exists():
        return str(book_path)
    else:
        raise FileNotFoundError(f"Book data not found: {book_id}")

def list_available_books() -> list:
    """List all available book datasets from filesystem (legacy, used for chunk retrieval)"""
    book_data_dir = Path(get_book_data_base_path())
    if not book_data_dir.exists():
        return []

    books = []
    for item in book_data_dir.iterdir():
        if item.is_dir() and (item / "vdb_entities.json").exists():
            books.append(item.name)

    return sorted(books)

def list_books_from_neo4j() -> list:
    """List books from Neo4j BOOK nodes (single source of truth for book list)"""
    try:
        from reconciliation_api import get_neo4j_driver

        driver = get_neo4j_driver()
        if not driver:
            logger.warning("Neo4j driver not available, falling back to filesystem")
            return None

        query = """
        MATCH (b:BOOK)
        OPTIONAL MATCH (b)-[:CONTAINS_ENTITY]->(e)
        OPTIONAL MATCH (b)-[:HAS_COMMUNITY]->(c)
        WITH b, count(DISTINCT e) as entity_count, count(DISTINCT c) as community_count
        RETURN b.id as id, b.name as name, entity_count, community_count
        ORDER BY b.name
        """

        with driver.session() as session:
            result = session.run(query)
            books = []
            for record in result:
                books.append({
                    'id': record['id'],
                    'name': record['name'] or record['id'],
                    'entity_count': record['entity_count'],
                    'community_count': record['community_count']
                })
            logger.info(f"📚 Loaded {len(books)} books from Neo4j")
            return books
    except Exception as e:
        logger.error(f"Error querying Neo4j for books: {e}")
        return None

def register_books_endpoints(app):
    """Register book-related endpoints"""

    @app.route('/books', methods=['GET'])
    def list_books():
        """
        Get list of available books in the GraphRAG dataset.
        Uses Neo4j BOOK nodes as the single source of truth.
        """
        try:
            # Try Neo4j first (single source of truth)
            neo4j_books = list_books_from_neo4j()

            if neo4j_books is not None:
                # Format Neo4j results to match frontend expectations
                books_info = []
                for book in neo4j_books:
                    books_info.append({
                        'id': book['id'],
                        'name': book['name'],
                        'path': None,  # Path not needed for Neo4j-sourced books
                        'stats': {
                            'entities': book['entity_count'],
                            'communities': book['community_count']
                        }
                    })

                return jsonify({
                    'success': True,
                    'books': books_info,
                    'count': len(books_info),
                    'source': 'neo4j'
                })

            # Fallback to filesystem if Neo4j unavailable
            logger.warning("Falling back to filesystem for book list")
            available_books = list_available_books()

            books_info = []
            for book_id in available_books:
                try:
                    book_path = get_book_data_path(book_id)
                    books_info.append({
                        'id': book_id,
                        'name': book_id.replace('_', ' ').title(),
                        'path': book_path,
                        'stats': {}
                    })
                except Exception as e:
                    logger.warning(f"Could not get path for {book_id}: {e}")

            return jsonify({
                'success': True,
                'books': books_info,
                'count': len(books_info),
                'source': 'filesystem'
            })

        except Exception as e:
            logger.error(f"Error listing books: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/books/<book_id>/info', methods=['GET'])
    def get_book_info(book_id: str):
        """
        Get detailed information about a specific book
        """
        try:
            book_path = get_book_data_path(book_id)
            if not book_path:
                return jsonify({
                    'success': False,
                    'error': f'Book {book_id} not found'
                }), 404

            import json
            from pathlib import Path

            book_info = {
                'id': book_id,
                'name': book_id.replace('_', ' ').title(),
                'path': book_path,
                'files': {}
            }

            # Check each expected GraphRAG file
            expected_files = {
                'entities': 'vdb_entities.json',
                'communities': 'kv_store_community_reports.json',
                'documents': 'kv_store_full_docs.json',
                'text_chunks': 'kv_store_text_chunks.json',
                'graph': 'graph_chunk_entity_relation.graphml'
            }

            for file_type, filename in expected_files.items():
                file_path = Path(book_path) / filename
                if file_path.exists():
                    try:
                        if filename.endswith('.json'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    item_count = len(data)
                                elif isinstance(data, list):
                                    item_count = len(data)
                                else:
                                    item_count = 1
                        else:
                            item_count = file_path.stat().st_size

                        book_info['files'][file_type] = {
                            'exists': True,
                            'size': file_path.stat().st_size,
                            'count': item_count
                        }
                    except Exception as e:
                        book_info['files'][file_type] = {
                            'exists': True,
                            'size': file_path.stat().st_size,
                            'error': str(e)
                        }
                else:
                    book_info['files'][file_type] = {'exists': False}

            return jsonify({
                'success': True,
                'book': book_info
            })

        except Exception as e:
            logger.error(f"Error getting book info for {book_id}: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/data/refresh', methods=['POST'])
    def refresh_data():
        """
        Force refresh of data from local book data
        """
        try:
            logger.info("🔄 Refreshing local book data...")

            # Get fresh list of available books
            available_books = list_available_books()

            return jsonify({
                'success': True,
                'message': 'Local book data refreshed successfully',
                'data_path': 'book_data',
                'available_books': available_books,
                'book_count': len(available_books)
            })

        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500