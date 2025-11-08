"""
Books endpoints for the Reconciliation API
Handles book data management and listing
"""

from flask import jsonify, request
from gdrive_data_manager import gdrive_manager
import logging

logger = logging.getLogger(__name__)

def register_books_endpoints(app):
    """Register book-related endpoints"""

    @app.route('/books', methods=['GET'])
    def list_books():
        """
        Get list of available books in the GraphRAG dataset
        """
        try:
            # Ensure data is available
            gdrive_manager.ensure_data_available()

            # Get list of available books
            available_books = gdrive_manager.list_available_books()

            # Get detailed info for each book
            books_info = []
            for book_id in available_books:
                book_path = gdrive_manager.get_book_data_path(book_id)
                if book_path:
                    # Try to get some metadata about the book
                    try:
                        import json
                        from pathlib import Path

                        # Check for full docs to get book info
                        full_docs_path = Path(book_path) / "kv_store_full_docs.json"
                        doc_count = 0
                        if full_docs_path.exists():
                            with open(full_docs_path, 'r', encoding='utf-8') as f:
                                docs_data = json.load(f)
                                doc_count = len(docs_data)

                        # Check for entities
                        entities_path = Path(book_path) / "vdb_entities.json"
                        entity_count = 0
                        if entities_path.exists():
                            with open(entities_path, 'r', encoding='utf-8') as f:
                                entities_data = json.load(f)
                                if 'entities' in entities_data:
                                    entity_count = len(entities_data['entities'])

                        books_info.append({
                            'id': book_id,
                            'name': book_id.replace('_', ' ').title(),
                            'path': book_path,
                            'stats': {
                                'documents': doc_count,
                                'entities': entity_count
                            }
                        })

                    except Exception as e:
                        logger.warning(f"Could not get metadata for {book_id}: {e}")
                        books_info.append({
                            'id': book_id,
                            'name': book_id.replace('_', ' ').title(),
                            'path': book_path,
                            'stats': {}
                        })

            return jsonify({
                'success': True,
                'books': books_info,
                'count': len(books_info)
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
            book_path = gdrive_manager.get_book_data_path(book_id)
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
        Force refresh of data from Google Drive
        """
        try:
            logger.info("🔄 Forcing data refresh from Google Drive...")

            # Clean up old data
            gdrive_manager.cleanup_old_data()

            # Download fresh data
            data_path = gdrive_manager.ensure_data_available()
            available_books = gdrive_manager.list_available_books()

            return jsonify({
                'success': True,
                'message': 'Data refreshed successfully',
                'data_path': data_path,
                'available_books': available_books,
                'book_count': len(available_books)
            })

        except Exception as e:
            logger.error(f"Error refreshing data: {e}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500