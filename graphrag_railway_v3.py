#!/usr/bin/env python3
"""
GraphRAG API v3 - Improved data management for Railway
With robust Google Drive download and data persistence
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import random
import requests
import zipfile
import html
import traceback
import time

app = Flask(__name__)

# CORS configuration for Railway deployment
CORS(app, origins=[
    "http://localhost:3000",
    "http://localhost:3001",
    "https://borges-library-web.vercel.app",
    "https://*.vercel.app",
    "https://reconciliation-api-production.up.railway.app"
],
methods=['GET', 'POST', 'OPTIONS'],
allow_headers=['Content-Type', 'Authorization'],
supports_credentials=True)

# Global cache for parsed data
PARSED_DATA_CACHE = {}

def download_file_from_google_drive(file_id, destination):
    """Download file from Google Drive with improved error handling"""
    try:
        print(f"📥 Downloading from Google Drive to: {destination}")

        # Create a session for better connection handling
        session = requests.Session()

        # Try direct download first
        base_url = "https://drive.google.com/uc"
        params = {'export': 'download', 'id': file_id}

        response = session.get(base_url, params=params, stream=True)

        # Check if we need to handle virus scan warning
        if response.status_code == 200:
            # Look for virus scan warning
            if b"confirm=" in response.content[:1000] or b"virus" in response.content[:1000].lower():
                print("⚠️  Large file detected, getting confirmation token...")

                # Parse the confirmation token
                for line in response.iter_lines():
                    if b'confirm=' in line:
                        import re
                        confirm_match = re.search(b'confirm=([0-9A-Za-z_-]+)', line)
                        if confirm_match:
                            confirm_token = confirm_match.group(1).decode('utf-8')
                            print(f"✅ Got confirmation token: {confirm_token[:10]}...")

                            # Download with confirmation
                            params['confirm'] = confirm_token
                            response = session.get(base_url, params=params, stream=True)
                            break

            # Save the file
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"📊 Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

            print(f"\n✅ Download complete: {destination} ({downloaded} bytes)")
            return True
        else:
            print(f"❌ Download failed with status: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Download error: {str(e)}")
        traceback.print_exc()
        return False

def ensure_book_data():
    """Ensure book data is available, download if necessary"""
    print("🔍 Checking for book data...")

    # Check if data already exists
    required_books = [
        'vallee_sans_hommes_frison', 'racines_ciel_gary', 'policeman_decoin',
        'a_rebours_huysmans', 'chien_blanc_gary', 'peau_bison_frison',
        'tilleul_soir_anglade', 'villa_triste_modiano'
    ]

    # Check how many books are already present
    existing_books = []
    for book_dir in required_books:
        graph_file = f"{book_dir}/graph_chunk_entity_relation.graphml"
        if os.path.exists(graph_file):
            existing_books.append(book_dir)
            print(f"  ✅ Found: {book_dir}")
        else:
            print(f"  ❌ Missing: {book_dir}")

    print(f"\n📊 Status: {len(existing_books)}/{len(required_books)} books available")

    # If all books are present, we're done
    if len(existing_books) == len(required_books):
        print("✅ All book data is available!")
        return True

    # Try to download missing data
    print("\n📦 Attempting to download missing book data...")

    # Google Drive file ID (hardcoded as fallback)
    drive_file_id = os.environ.get('BOOK_DATA_DRIVE_ID', '1NTgs97rvlVHYozTfodNo5kKsambOpXr1')

    if not drive_file_id:
        print("❌ No Google Drive file ID available")
        return False

    archive_path = "book_data.zip"

    # Download the archive
    if download_file_from_google_drive(drive_file_id, archive_path):
        print("📂 Extracting archive...")

        try:
            # Extract the archive
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # List contents first
                file_list = zip_ref.namelist()
                print(f"📋 Archive contains {len(file_list)} files")

                # Extract all files
                zip_ref.extractall('.')
                print("✅ Extraction complete")

            # Clean up archive
            os.remove(archive_path)
            print("🗑️  Archive removed")

            # Verify extraction
            newly_found = 0
            for book_dir in required_books:
                graph_file = f"{book_dir}/graph_chunk_entity_relation.graphml"
                if os.path.exists(graph_file) and book_dir not in existing_books:
                    newly_found += 1
                    print(f"  ✨ Newly available: {book_dir}")

            print(f"\n✅ Successfully added {newly_found} books")
            return True

        except Exception as e:
            print(f"❌ Extraction failed: {str(e)}")
            traceback.print_exc()

            # Try to clean up
            if os.path.exists(archive_path):
                os.remove(archive_path)

            return False
    else:
        print("❌ Failed to download book data from Google Drive")
        return False

def find_books():
    """Find all available books with GraphRAG data"""
    books = []

    # Look for book directories
    for item in os.listdir('.'):
        if os.path.isdir(item) and not item.startswith('.'):
            # Check for GraphML file
            graph_path = f"{item}/graph_chunk_entity_relation.graphml"
            if os.path.exists(graph_path):
                books.append({
                    "id": item,
                    "name": item.replace('_', ' ').title(),
                    "has_data": True,
                    "graph_path": graph_path
                })

    return books

def clean_text(text):
    """Clean text from HTML entities and quotes"""
    if not text:
        return ""
    return html.unescape(text).strip('"').strip()

def parse_graphml_cached(book_id, graph_path):
    """Parse GraphML file with caching"""

    # Check cache first
    if book_id in PARSED_DATA_CACHE:
        print(f"📚 Using cached data for {book_id}")
        return PARSED_DATA_CACHE[book_id]

    print(f"📄 Parsing GraphML for {book_id}: {graph_path}")

    try:
        tree = ET.parse(graph_path)
        root = tree.getroot()

        # Namespace
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        # Build key mapping
        key_mapping = {}
        for key in root.findall('.//g:key', ns):
            key_id = key.get('id')
            key_name = key.get('attr.name')
            if key_id and key_name:
                key_mapping[key_id] = key_name

        entities = []
        relationships = []

        # Parse nodes
        for node in root.findall('.//g:node', ns):
            node_id = node.get('id')
            if not node_id:
                continue

            entity = {
                'id': clean_text(node_id),
                'entity_type': 'UNKNOWN',
                'description': ''
            }

            # Extract attributes
            for data in node.findall('g:data', ns):
                key = data.get('key')
                if key in key_mapping:
                    attr_name = key_mapping[key]
                    value = data.text or ''
                    entity[attr_name] = clean_text(value)

            # Ensure description
            if not entity.get('description'):
                entity['description'] = f"Entity {entity['id']}"

            entities.append(entity)

        # Parse edges
        for edge in root.findall('.//g:edge', ns):
            source = edge.get('source')
            target = edge.get('target')

            if not source or not target:
                continue

            rel = {
                'source': clean_text(source),
                'target': clean_text(target),
                'description': 'Relation',
                'weight': 1.0
            }

            # Extract attributes
            for data in edge.findall('g:data', ns):
                key = data.get('key')
                if key in key_mapping:
                    attr_name = key_mapping[key]
                    value = data.text or ''

                    if attr_name == 'weight':
                        try:
                            rel[attr_name] = float(value)
                        except:
                            rel[attr_name] = 1.0
                    else:
                        rel[attr_name] = clean_text(value)

            relationships.append(rel)

        print(f"✅ Parsed {len(entities)} entities, {len(relationships)} relationships")

        # Cache the result
        PARSED_DATA_CACHE[book_id] = (entities, relationships)

        return entities, relationships

    except Exception as e:
        print(f"❌ Error parsing GraphML: {str(e)}")
        traceback.print_exc()
        return [], []

def generate_answer(query, book_id, entities, relationships):
    """Generate an answer based on the query and graph data"""

    # Simple keyword matching
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]

    # Score entities
    scored_entities = []
    for entity in entities:
        score = 0
        entity_text = ' '.join(str(v).lower() for v in entity.values() if v)

        for word in query_words:
            if word in entity_text:
                score += 1

        if score > 0:
            scored_entities.append({
                **entity,
                'relevance_score': score
            })

    # Sort by relevance
    scored_entities.sort(key=lambda x: x['relevance_score'], reverse=True)
    relevant_entities = scored_entities[:20]

    # If no relevant entities, use first few
    if not relevant_entities and entities:
        relevant_entities = entities[:10]

    # Find relevant relationships
    entity_ids = {e['id'] for e in relevant_entities}
    relevant_rels = [
        r for r in relationships
        if r['source'] in entity_ids or r['target'] in entity_ids
    ][:20]

    # Generate contextual answer
    book_title = book_id.replace('_', ' ').title()

    if 'thème' in query_lower or 'theme' in query_lower:
        answer = f"Les thèmes principaux de '{book_title}' incluent l'exploration littéraire, le développement des personnages et la structure narrative complexe. L'œuvre présente {len(relevant_entities)} éléments thématiques interconnectés."
    elif 'personnage' in query_lower or 'character' in query_lower:
        person_entities = [e for e in relevant_entities if 'person' in str(e.get('entity_type', '')).lower()]
        if person_entities:
            names = [e['id'][:30] for e in person_entities[:3]]
            answer = f"Les personnages principaux de '{book_title}' incluent : {', '.join(names)}."
        else:
            answer = f"'{book_title}' présente une galerie de personnages complexes et interconnectés."
    elif 'lieu' in query_lower or 'location' in query_lower:
        location_entities = [e for e in relevant_entities if 'location' in str(e.get('entity_type', '')).lower() or 'geo' in str(e.get('entity_type', '')).lower()]
        if location_entities:
            places = [e['id'][:30] for e in location_entities[:3]]
            answer = f"L'action de '{book_title}' se déroule dans : {', '.join(places)}."
        else:
            answer = f"'{book_title}' explore différents espaces géographiques et symboliques."
    else:
        answer = f"Concernant votre question sur '{book_title}', j'ai identifié {len(relevant_entities)} éléments pertinents dans le graphe de connaissances."

    return {
        "success": True,
        "answer": answer,
        "searchPath": {
            "entities": [
                {
                    "id": e.get('id', ''),
                    "type": e.get('entity_type', 'ENTITY'),
                    "description": e.get('description', '')[:200],
                    "rank": i + 1,
                    "order": i + 1,
                    "score": e.get('relevance_score', 1) / max(len(query_words), 1)
                }
                for i, e in enumerate(relevant_entities[:10])
            ],
            "relations": [
                {
                    "source": r.get('source', ''),
                    "target": r.get('target', ''),
                    "description": r.get('description', ''),
                    "weight": r.get('weight', 1.0),
                    "rank": i + 1,
                    "traversalOrder": i + 1
                }
                for i, r in enumerate(relevant_rels[:15])
            ],
            "communities": []
        }
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    books = find_books()
    return jsonify({
        "status": "healthy",
        "service": "GraphRAG API",
        "version": "3.0",
        "books_available": len(books),
        "cache_size": len(PARSED_DATA_CACHE)
    })

@app.route('/books', methods=['GET'])
def list_books():
    """List all available books"""
    books = find_books()
    return jsonify({"books": books})

@app.route('/query', methods=['POST'])
def query_graph():
    """Process a GraphRAG query"""
    try:
        data = request.json
        query = data.get('query', '')
        book_id = data.get('book_id')
        mode = data.get('mode', 'local')

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Find available books
        books = find_books()

        if not books:
            return jsonify({"error": "No books available. Data may still be downloading."}), 503

        # Use first book if not specified
        if not book_id:
            book_id = books[0]['id']

        # Find the book
        book_data = None
        for book in books:
            if book['id'] == book_id:
                book_data = book
                break

        if not book_data:
            available = [b['id'] for b in books]
            return jsonify({
                "error": f"Book '{book_id}' not found",
                "available_books": available
            }), 404

        # Parse the graph data (with caching)
        entities, relationships = parse_graphml_cached(book_id, book_data['graph_path'])

        if not entities:
            return jsonify({
                "error": f"Failed to parse data for book '{book_id}'",
                "suggestion": "Try another book or wait for data to load"
            }), 500

        # Generate response
        response = generate_answer(query, book_id, entities, relationships)

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error in query_graph: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Detailed status endpoint"""
    books = find_books()
    return jsonify({
        "status": "operational",
        "books": [
            {
                "id": b['id'],
                "name": b['name'],
                "cached": b['id'] in PARSED_DATA_CACHE
            }
            for b in books
        ],
        "cache_entries": len(PARSED_DATA_CACHE),
        "environment": {
            "has_drive_id": bool(os.environ.get('BOOK_DATA_DRIVE_ID')),
            "port": os.environ.get('PORT', '5000')
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 GraphRAG API v3.0 - Railway Edition")
    print("=" * 60)

    # Ensure book data is available
    print("\n📚 Initializing book data...")
    data_ready = ensure_book_data()

    if not data_ready:
        print("\n⚠️  WARNING: Book data download failed!")
        print("The API will start but may have limited functionality.")

    # List available books
    books = find_books()
    print(f"\n📖 Books available: {len(books)}")
    for book in books:
        print(f"  • {book['name']} ({book['id']})")

    if not books:
        print("\n❌ No books found! The API will return errors for queries.")

    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🌐 Starting server on port {port}...")
    print("=" * 60)

    app.run(host='0.0.0.0', port=port, debug=False)