"""
Book Ingestion API endpoints (Admin only).

Provides REST API for book ingestion operations:
- POST /admin/ingest - Start book ingestion
- GET /admin/ingest/{job_id} - Get job status
- POST /admin/ingest/{job_id}/rollback - Rollback ingestion
- GET /admin/ingest/jobs - List all jobs

All endpoints require admin authentication via X-Admin-API-Key header.

Constitution Compliance:
- Principle VI: Extensible Literature Foundation - admin-only access
"""

import os
import asyncio
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename

from middleware.admin_auth import require_admin_key, get_admin_id
from models.book_ingestion import (
    BookIngestion,
    BookMetadata,
    IngestionStatus,
    get_ingestion_job,
    get_all_ingestion_jobs,
    store_ingestion_job
)
from services.book_ingestion_service import BookIngestionService


# Blueprint for ingestion endpoints
ingestion_bp = Blueprint('ingestion', __name__, url_prefix='/admin')

# Upload folder for temporary file storage
UPLOAD_FOLDER = '/tmp/book_ingestion_uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'epub'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_ingestion_service() -> BookIngestionService:
    """Get or create the book ingestion service."""
    if not hasattr(current_app, '_ingestion_service'):
        # Get Neo4j client if available
        neo4j_client = getattr(current_app, 'neo4j_client', None)
        current_app._ingestion_service = BookIngestionService(
            neo4j_client=neo4j_client
        )
    return current_app._ingestion_service


@ingestion_bp.route('/ingest', methods=['POST'])
@require_admin_key
def start_ingestion():
    """
    Start book ingestion.

    Request body (multipart/form-data):
        - file: Book file (PDF, EPUB, or TXT)
        - title: Book title (required)
        - author: Author name (required)
        - genre: Literary genre (optional)
        - publication_date: Publication date (optional)

    Returns:
        202: Ingestion job started
        400: Invalid request
        401: Admin authentication required
        409: Book already exists
    """
    # Validate required fields
    if 'title' not in request.form:
        return jsonify({
            'error': 'missing_field',
            'message': 'Book title is required'
        }), 400

    if 'author' not in request.form:
        return jsonify({
            'error': 'missing_field',
            'message': 'Author name is required'
        }), 400

    # Get file from request or file_path parameter
    file_path = None

    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'no_file',
                'message': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'invalid_format',
                'message': f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Save file to upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

    elif 'file_path' in request.form:
        # Allow specifying file path directly (for CLI and testing)
        file_path = request.form['file_path']
        if not os.path.exists(file_path):
            # Try in raw_books directory
            raw_books_path = os.path.join(
                '/Users/arthursarazin/Documents/nano-graphrag/book_data/raw_books',
                os.path.basename(file_path)
            )
            if os.path.exists(raw_books_path):
                file_path = raw_books_path
            else:
                return jsonify({
                    'error': 'file_not_found',
                    'message': f'File not found: {file_path}'
                }), 400
    else:
        return jsonify({
            'error': 'no_file',
            'message': 'No file provided. Use file upload or file_path parameter.'
        }), 400

    # Create metadata
    metadata = BookMetadata(
        title=request.form['title'],
        author=request.form['author'],
        genre=request.form.get('genre'),
        publication_date=request.form.get('publication_date')
    )

    # Get admin ID from auth
    admin_id = get_admin_id()

    # Create ingestion job
    job = BookIngestion(
        file_path=file_path,
        book_metadata=metadata,
        admin_id=admin_id
    )
    store_ingestion_job(job)

    # Start async ingestion in background
    service = get_ingestion_service()

    def run_ingestion():
        """Run ingestion in background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                service.ingest_book(file_path, metadata, admin_id)
            )
        except Exception as e:
            print(f"Ingestion failed: {e}")
            job.fail(str(e))
            store_ingestion_job(job)
        finally:
            loop.close()

    # Run in background thread
    import threading
    thread = threading.Thread(target=run_ingestion)
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job.id,
        'book_id': job.book_id,
        'status': job.status.value,
        'message': f'Book ingestion started. Use GET /admin/ingest/{job.id} to track progress.'
    }), 202


@ingestion_bp.route('/ingest/<job_id>', methods=['GET'])
@require_admin_key
def get_ingestion_status(job_id: str):
    """
    Get ingestion job status.

    Path parameters:
        - job_id: Ingestion job identifier

    Returns:
        200: Job status retrieved
        404: Job not found
    """
    job = get_ingestion_job(job_id)
    if not job:
        return jsonify({
            'error': 'not_found',
            'message': f'Ingestion job not found: {job_id}'
        }), 404

    return jsonify(job.to_dict()), 200


@ingestion_bp.route('/ingest/<job_id>/rollback', methods=['POST'])
@require_admin_key
def rollback_ingestion(job_id: str):
    """
    Rollback book ingestion.

    Removes all entities and relationships created by this ingestion job.

    Path parameters:
        - job_id: Ingestion job identifier

    Returns:
        200: Rollback completed
        400: Cannot rollback (job still processing or already rolled back)
        404: Job not found
    """
    job = get_ingestion_job(job_id)
    if not job:
        return jsonify({
            'error': 'not_found',
            'message': f'Ingestion job not found: {job_id}'
        }), 404

    if not job.can_rollback():
        return jsonify({
            'error': 'invalid_state',
            'message': f'Cannot rollback job in status: {job.status.value}. '
                      f'Only completed or failed jobs can be rolled back.'
        }), 400

    service = get_ingestion_service()

    # Run rollback
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(service.rollback(job_id))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({
            'error': 'rollback_failed',
            'message': str(e)
        }), 500
    finally:
        loop.close()


@ingestion_bp.route('/upload-book-data', methods=['POST'])
@require_admin_key
def upload_book_data():
    """
    Upload processed book data to Railway volume.

    This endpoint receives a tar.gz archive of processed book data
    (GraphML, community reports, chunks, etc.) and extracts it to
    the Railway volume for production use.

    Request body (JSON):
        - book_name: Name of the book directory (required)
        - archive_data: Base64-encoded tar.gz archive (required)
        - archive_format: Format of archive, e.g., 'tar.gz' (required)

    Returns:
        200: Upload successful
        400: Invalid request
        401: Admin authentication required
        500: Server error
    """
    import base64
    import tarfile
    import tempfile
    import shutil
    from pathlib import Path

    data = request.get_json()

    if not data:
        return jsonify({
            'error': 'missing_body',
            'message': 'Request body is required'
        }), 400

    book_name = data.get('book_name')
    archive_data = data.get('archive_data')
    archive_format = data.get('archive_format')

    if not book_name:
        return jsonify({
            'error': 'missing_field',
            'message': 'book_name is required'
        }), 400

    if not archive_data:
        return jsonify({
            'error': 'missing_field',
            'message': 'archive_data is required'
        }), 400

    if archive_format != 'tar.gz':
        return jsonify({
            'error': 'unsupported_format',
            'message': f'Unsupported archive format: {archive_format}. Only tar.gz is supported.'
        }), 400

    # Get target directory (Railway volume or local)
    volume_path = os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', 'book_data')
    target_dir = Path(volume_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    book_target = target_dir / book_name

    try:
        # Decode archive
        archive_bytes = base64.b64decode(archive_data)

        # Write to temp file and extract
        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            tmp.write(archive_bytes)
            tmp_path = tmp.name

        # Remove existing book data if present
        if book_target.exists():
            shutil.rmtree(book_target)

        # Extract archive
        with tarfile.open(tmp_path, 'r:gz') as tar:
            tar.extractall(path=target_dir)

        # Clean up temp file
        os.unlink(tmp_path)

        # Verify extraction
        if not book_target.exists():
            return jsonify({
                'error': 'extraction_failed',
                'message': f'Book directory not found after extraction: {book_name}'
            }), 500

        # Count extracted files
        file_count = sum(1 for _ in book_target.iterdir() if _.is_file())

        return jsonify({
            'success': True,
            'message': f'Book data uploaded successfully',
            'book_name': book_name,
            'volume_path': str(book_target),
            'files_extracted': file_count
        }), 200

    except base64.binascii.Error as e:
        return jsonify({
            'error': 'decode_error',
            'message': f'Failed to decode archive data: {str(e)}'
        }), 400
    except tarfile.TarError as e:
        return jsonify({
            'error': 'extract_error',
            'message': f'Failed to extract archive: {str(e)}'
        }), 500
    except Exception as e:
        return jsonify({
            'error': 'server_error',
            'message': f'Upload failed: {str(e)}'
        }), 500


@ingestion_bp.route('/ingest/jobs', methods=['GET'])
@require_admin_key
def list_ingestion_jobs():
    """
    List all ingestion jobs.

    Query parameters:
        - status: Filter by job status (queued, processing, completed, failed, rolled_back)
        - limit: Maximum number of jobs to return (default: 20, max: 100)
        - offset: Offset for pagination (default: 0)

    Returns:
        200: Job list retrieved
    """
    # Parse query parameters
    status_filter = request.args.get('status')
    limit = min(int(request.args.get('limit', 20)), 100)
    offset = int(request.args.get('offset', 0))

    # Convert status filter
    status = None
    if status_filter:
        try:
            status = IngestionStatus(status_filter)
        except ValueError:
            return jsonify({
                'error': 'invalid_status',
                'message': f'Invalid status: {status_filter}. '
                          f'Valid values: {[s.value for s in IngestionStatus]}'
            }), 400

    # Get jobs
    jobs, total = get_all_ingestion_jobs(status=status, limit=limit, offset=offset)

    return jsonify({
        'jobs': [job.to_dict() for job in jobs],
        'total': total,
        'limit': limit,
        'offset': offset
    }), 200


@ingestion_bp.route('/ingest/schema', methods=['GET'])
@require_admin_key
def get_schema():
    """
    Get the existing Neo4j schema for reference.

    Returns the expected node labels, relationship types, and properties
    that new books should conform to.

    Returns:
        200: Schema retrieved
    """
    service = get_ingestion_service()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        schema = loop.run_until_complete(service.get_existing_schema())
        return jsonify({
            'schema': schema,
            'config_hash': service.config_hash,
            'message': 'New books will be validated against this schema.'
        }), 200
    finally:
        loop.close()
