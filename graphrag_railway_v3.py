#!/usr/bin/env python3
"""
GraphRAG API v3 - Improved data management for Railway
With robust Google Drive download and data persistence
Now with actual NanoGraphRAG integration for detailed responses
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
import sys

# Add nano-graphrag to path
sys.path.append('/opt/render/project/src/nano-graphrag')
sys.path.append('./nano-graphrag')
sys.path.append('./nano_graphrag')

try:
    from nano_graphrag import GraphRAG, QueryParam
    NANO_GRAPHRAG_AVAILABLE = True
    print("✅ NanoGraphRAG imported successfully")
except ImportError as e:
    print(f"⚠️ NanoGraphRAG not available: {e}")
    try:
        # Try importing from local copy
        from nano_graphrag import GraphRAG, QueryParam
        NANO_GRAPHRAG_AVAILABLE = True
        print("✅ NanoGraphRAG imported from local copy")
    except ImportError as e2:
        print(f"⚠️ NanoGraphRAG not available locally either: {e2}")
        NANO_GRAPHRAG_AVAILABLE = False

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

# Global cache for GraphRAG instances
GRAPHRAG_INSTANCES = {}

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

def select_best_book_for_query(query, books):
    """Select the best book for a query using keyword matching"""
    query_lower = query.lower()

    # Book-specific keywords for better matching
    book_keywords = {
        'a_rebours_huysmans': ['esseintes', 'des esseintes', 'huysmans', 'rebours', 'parfums', 'décadence', 'esthète'],
        'racines_ciel_gary': ['gary', 'racines', 'ciel', 'éléphants', 'afrique', 'morel'],
        'chien_blanc_gary': ['chien blanc', 'romain gary', 'racisme', 'amérique', 'hollywood'],
        'peau_bison_frison': ['bison', 'peau', 'frison', 'indiens', 'western', 'prairie'],
        'policeman_decoin': ['policeman', 'police', 'decoin', 'enquête', 'crime'],
        'tilleul_soir_anglade': ['tilleul', 'soir', 'anglade', 'provence', 'village'],
        'vallee_sans_hommes_frison': ['vallée sans hommes', 'vallée', 'frison', 'solitude'],
        'villa_triste_modiano': ['villa triste', 'modiano', 'nostalgie', 'mémoire', 'nice']
    }

    # Score each book based on keyword matches
    best_score = 0
    best_book = None

    for book in books:
        book_id = book['id']
        if book_id in book_keywords:
            score = 0
            for keyword in book_keywords[book_id]:
                if keyword in query_lower:
                    score += len(keyword)  # Longer keywords get higher scores

            if score > best_score:
                best_score = score
                best_book = book_id

    # If no specific match found, search across all books or use a default strategy
    if not best_book:
        print(f"🔍 No specific book keywords found for query: '{query[:50]}...', using multi-book strategy")
        # For general queries like "quels livres sont disponibles", try to search multiple books
        if any(word in query_lower for word in ['livres', 'books', 'disponibles', 'available', 'liste', 'catalog']):
            # For catalog queries, use a representative book or implement multi-book search
            best_book = 'racines_ciel_gary'  # Use a different default to show variety
        else:
            # For other queries, use the first book
            best_book = books[0]['id'] if books else None

    return best_book

def should_search_multiple_books(query):
    """Determine if query should search across multiple books"""
    query_lower = query.lower()
    multi_book_keywords = [
        'livres', 'books', 'disponibles', 'available', 'catalogue', 'catalog',
        'liste', 'list', 'collection', 'bibliothèque', 'library',
        'quels livres', 'which books', 'tous les livres', 'all books',
        'différents livres', 'different books'
    ]

    return any(keyword in query_lower for keyword in multi_book_keywords)

def generate_multi_book_answer(query, books):
    """Generate answer by searching across ALL books with GraphRAG"""
    print(f"🔍 Searching query '{query}' across {len(books)} books...")

    best_response = None
    best_score = 0
    all_entities = []
    all_relations = []

    # Search each book with GraphRAG
    for book in books:
        try:
            print(f"📖 Searching in book: {book['name']}")

            # Parse book data
            entities, relationships = parse_book_data(book['graph_path'])
            if not entities:
                print(f"  ⚠️ No entities found for {book['id']}")
                continue

            # Try to initialize GraphRAG for this book
            graph_rag_instance = initialize_graphrag_for_book(book['id'], book['graph_path'])

            # Generate answer for this book
            book_response = generate_answer(query, book['id'], entities, relationships, graph_rag_instance)

            if book_response and book_response.get('success'):
                answer = book_response.get('answer', '')

                # Score based on answer length and relevance (simple heuristic)
                score = len(answer) if answer and answer != "Sorry, I'm not able to provide an answer to that question." else 0

                print(f"  📊 Book {book['id']} score: {score}")

                if score > best_score:
                    best_score = score
                    best_response = book_response
                    best_response['source_book'] = book['name']

                # Collect entities and relations from all books
                if 'searchPath' in book_response:
                    search_path = book_response['searchPath']
                    if 'entities' in search_path:
                        for entity in search_path['entities'][:3]:  # Limit per book
                            entity['source_book'] = book['name']
                            all_entities.append(entity)
                    if 'relations' in search_path:
                        for relation in search_path['relations'][:3]:  # Limit per book
                            relation['source_book'] = book['name']
                            all_relations.append(relation)

        except Exception as e:
            print(f"❌ Error searching book {book['id']}: {e}")
            continue

    # Return the best response found, or a combined response
    if best_response:
        print(f"✅ Best response from book: {best_response.get('source_book', 'unknown')}")

        # Enhance with multi-book entities
        if 'searchPath' in best_response:
            best_response['searchPath']['entities'] = all_entities[:20]  # Limit total entities
            best_response['searchPath']['relations'] = all_relations[:15]  # Limit total relations

        return best_response
    else:
        # No good response found, return general info about available books
        book_list = ", ".join([book['name'] for book in books])
        return {
            'success': True,
            'answer': f"Je n'ai pas trouvé d'information spécifique pour votre question dans les {len(books)} livres disponibles: {book_list}. Essayez une question plus spécifique ou mentionnez un livre particulier.",
            'searchPath': {
                'entities': all_entities[:10],
                'relations': all_relations[:5],
                'communities': []
            }
        }

# Global unified GraphRAG instance
UNIFIED_GRAPHRAG = None

def create_unified_working_directory():
    """Create a unified working directory by merging all book data"""
    import shutil

    unified_dir = "unified_books"

    # Clean up existing unified directory
    if os.path.exists(unified_dir):
        shutil.rmtree(unified_dir)

    os.makedirs(unified_dir)
    print(f"📁 Created unified directory: {unified_dir}")

    books = find_books()
    all_entities = {}
    all_relationships = []
    all_text_chunks = {}
    all_full_docs = {}

    # Merge data from all books
    for book in books:
        book_dir = book['id']
        print(f"🔄 Merging data from: {book_dir}")

        try:
            # Load entities
            entities_file = f"{book_dir}/vdb_entities.json"
            if os.path.exists(entities_file):
                with open(entities_file, 'r', encoding='utf-8') as f:
                    entities_data = json.load(f)
                    if 'entities' in entities_data:
                        for entity in entities_data['entities']:
                            entity_name = entity.get('entity_name', '')
                            if entity_name:
                                # Prefix entity with book name to avoid conflicts
                                prefixed_name = f"{book_dir}_{entity_name}"
                                entity['entity_name'] = prefixed_name
                                all_entities[prefixed_name] = entity

            # Load text chunks
            chunks_file = f"{book_dir}/kv_store_text_chunks.json"
            if os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    for key, value in chunks_data.items():
                        all_text_chunks[f"{book_dir}_{key}"] = value

            # Load full docs
            docs_file = f"{book_dir}/kv_store_full_docs.json"
            if os.path.exists(docs_file):
                with open(docs_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                    for key, value in docs_data.items():
                        all_full_docs[f"{book_dir}_{key}"] = value

        except Exception as e:
            print(f"❌ Error merging {book_dir}: {e}")
            continue

    # Write unified files
    try:
        # Write unified entities
        unified_entities = {
            "entities": list(all_entities.values())
        }
        with open(f"{unified_dir}/vdb_entities.json", 'w', encoding='utf-8') as f:
            json.dump(unified_entities, f, ensure_ascii=False, indent=2)

        # Write unified text chunks
        with open(f"{unified_dir}/kv_store_text_chunks.json", 'w', encoding='utf-8') as f:
            json.dump(all_text_chunks, f, ensure_ascii=False, indent=2)

        # Write unified full docs
        with open(f"{unified_dir}/kv_store_full_docs.json", 'w', encoding='utf-8') as f:
            json.dump(all_full_docs, f, ensure_ascii=False, indent=2)

        print(f"✅ Unified data created:")
        print(f"  📊 {len(all_entities)} entities")
        print(f"  📄 {len(all_text_chunks)} text chunks")
        print(f"  📚 {len(all_full_docs)} documents")

        return unified_dir

    except Exception as e:
        print(f"❌ Error writing unified data: {e}")
        return None

def get_unified_graphrag():
    """Get or create unified GraphRAG instance"""
    global UNIFIED_GRAPHRAG

    if UNIFIED_GRAPHRAG is None and NANO_GRAPHRAG_AVAILABLE:
        print("🔧 Creating unified GraphRAG instance...")

        # Create unified working directory
        unified_dir = create_unified_working_directory()
        if not unified_dir:
            return None

        try:
            from nano_graphrag._llm import gpt_4o_mini_complete

            UNIFIED_GRAPHRAG = GraphRAG(
                working_dir=unified_dir,
                best_model_func=gpt_4o_mini_complete,
                cheap_model_func=gpt_4o_mini_complete,
                embedding_func_max_async=4,
                best_model_max_async=2,
                cheap_model_max_async=4,
                embedding_batch_num=16,
                graph_cluster_algorithm="leiden"
            )
            print("✅ Unified GraphRAG instance created!")

        except Exception as e:
            print(f"❌ Error creating unified GraphRAG: {e}")
            UNIFIED_GRAPHRAG = None

    return UNIFIED_GRAPHRAG

def generate_multi_graphrag_answer(query, books):
    """Generate answer using multiple GraphRAG instances (one per book) and aggregate results"""
    print(f"🔍 Generating multi-GraphRAG answer for: '{query}'")

    if not NANO_GRAPHRAG_AVAILABLE:
        return generate_multi_book_answer(query, books)

    from nano_graphrag._llm import gpt_4o_mini_complete

    all_results = []
    successful_books = []

    for book_dir in books:
        if not os.path.exists(book_dir):
            print(f"⚠️ Directory {book_dir} not found, skipping")
            continue

        # Check if this book has GraphRAG data
        required_files = [
            f"{book_dir}/vdb_entities.json",
            f"{book_dir}/kv_store_community_reports.json",
            f"{book_dir}/graph_chunk_entity_relation.graphml"
        ]

        if not all(os.path.exists(f) for f in required_files):
            print(f"⚠️ {book_dir} missing GraphRAG files, skipping")
            continue

        try:
            print(f"📚 Querying {book_dir}...")

            # Create GraphRAG instance for this book
            book_rag = GraphRAG(
                working_dir=book_dir,
                best_model_func=gpt_4o_mini_complete,
                cheap_model_func=gpt_4o_mini_complete,
                embedding_func_max_async=4,
                best_model_max_async=2,
                cheap_model_max_async=4,
                embedding_batch_num=16,
                graph_cluster_algorithm="leiden"
            )

            # Query using global mode like test_query_analysis.py
            result = book_rag.query(query, param=QueryParam(mode="global"))

            if result and len(result.strip()) > 10:  # Valid non-empty result
                all_results.append({
                    'book': book_dir,
                    'result': result
                })
                successful_books.append(book_dir)
                print(f"✅ Found relevant content in {book_dir}")
            else:
                print(f"🔍 No relevant content found in {book_dir}")

        except Exception as e:
            print(f"❌ Error querying {book_dir}: {e}")
            continue

    # Aggregate and format results
    if not all_results:
        return {
            'success': False,
            'answer': "Je n'ai pas trouvé d'informations spécifiques sur cette question dans les livres disponibles.",
            'books_searched': len(books),
            'books_with_content': 0
        }

    # Format the aggregated answer
    formatted_answer = f"Voici les informations que j'ai trouvées :\n\n"

    for i, result_data in enumerate(all_results, 1):
        book_name = result_data['book'].replace('_', ' ').title()
        formatted_answer += f"**{book_name}:**\n{result_data['result']}\n\n"

    return {
        'success': True,
        'answer': formatted_answer.strip(),
        'books_searched': len(books),
        'books_with_content': len(successful_books),
        'successful_books': successful_books,
        'searchPath': {
            'entities': [],  # Could be enhanced to extract from result
            'relations': [],
            'communities': []
        }
    }

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

def initialize_graphrag_for_book(book_id, book_path):
    """Initialize GraphRAG instance for a specific book"""
    global GRAPHRAG_INSTANCES

    if not NANO_GRAPHRAG_AVAILABLE:
        print(f"⚠️ NanoGraphRAG not available, cannot initialize for {book_id}")
        return None

    # Check if already initialized
    if book_id in GRAPHRAG_INSTANCES:
        print(f"📚 Using existing GraphRAG instance for {book_id}")
        return GRAPHRAG_INSTANCES[book_id]

    try:
        print(f"🔄 Initializing GraphRAG for {book_id} at {book_path}")

        # Check if book directory has the necessary files for GraphRAG
        working_dir = Path(book_path).parent

        # Look for required GraphRAG files (these should be in the Google Drive data)
        required_files = [
            'kv_store_full_docs.json',
            'kv_store_text_chunks.json',
            'vdb_entities.json',
            'kv_store_community_reports.json'
        ]

        # Check what files are actually available
        available_files = list(working_dir.glob('*.json'))
        print(f"📄 Available files in {working_dir}: {[f.name for f in available_files]}")

        missing_files = []
        for file in required_files:
            if not (working_dir / file).exists():
                missing_files.append(file)

        if missing_files:
            print(f"⚠️ Missing GraphRAG files for {book_id}: {missing_files}")
            # Don't return None immediately - try to initialize anyway
            print(f"🔄 Attempting to initialize GraphRAG anyway...")
        else:
            print(f"✅ All required GraphRAG files found for {book_id}")

        # Initialize GraphRAG
        graph_rag = GraphRAG(
            working_dir=str(working_dir),
            embedding_func_max_async=4,
            best_model_max_async=2,
            cheap_model_max_async=4,
            embedding_batch_num=16,
            graph_cluster_algorithm="leiden"
        )

        # Cache the instance
        GRAPHRAG_INSTANCES[book_id] = graph_rag
        print(f"✅ GraphRAG initialized successfully for {book_id}")
        return graph_rag

    except Exception as e:
        print(f"❌ Failed to initialize GraphRAG for {book_id}: {e}")
        traceback.print_exc()
        return None

def generate_answer_with_graphrag(query, book_id, graph_rag_instance):
    """Generate answer using actual GraphRAG processing"""

    if not graph_rag_instance:
        print(f"⚠️ No GraphRAG instance available for {book_id}, falling back to simple method")
        return None

    try:
        print(f"🔍 Processing query with GraphRAG: '{query}' for book {book_id}")

        # Use GraphRAG to generate a comprehensive answer (try global mode first)
        result = graph_rag_instance.query(query, param=QueryParam(mode="global"))

        print(f"✅ GraphRAG query completed, result length: {len(result)} characters")

        return {
            "success": True,
            "answer": result,
            "method": "graphrag",
            "book_id": book_id,
            "searchPath": {
                "entities": [],
                "relations": [],
                "communities": []
            }
        }

    except Exception as e:
        print(f"❌ GraphRAG query failed for {book_id}: {e}")
        traceback.print_exc()
        return None

def generate_answer(query, book_id, entities, relationships, graph_rag_instance=None):
    """Generate an answer based on the query and graph data"""

    # Try GraphRAG first if available
    if graph_rag_instance and NANO_GRAPHRAG_AVAILABLE:
        graphrag_result = generate_answer_with_graphrag(query, book_id, graph_rag_instance)
        if graphrag_result:
            print(f"✅ Using GraphRAG answer for {book_id}")
            return graphrag_result

    # Fall back to simple keyword matching
    print(f"📝 Using simple answer generation for {book_id}")

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
        "method": "simple",
        "book_id": book_id,
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

        # If no specific book is requested, search across all books using GraphRAG
        if not book_id:
            print("🔍 No specific book requested, searching across all books with GraphRAG...")
            # Get book directories for GraphRAG
            book_dirs = [book['graph_path'].replace('/graph_chunk_entity_relation.graphml', '') for book in books]
            response = generate_multi_graphrag_answer(query, book_dirs)
        else:
            print(f"📖 Using specified book: {book_id}")

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

            # Try to initialize GraphRAG instance for this book
            graph_rag_instance = initialize_graphrag_for_book(book_id, book_data['graph_path'])

            # Generate response (GraphRAG if available, otherwise simple)
            response = generate_answer(query, book_id, entities, relationships, graph_rag_instance)

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