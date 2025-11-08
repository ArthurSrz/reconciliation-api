#!/usr/bin/env python3
"""
Test du système d'intégration Google Drive pour l'API de réconciliation
Vérifie le téléchargement, l'extraction et l'utilisation des données
"""

import requests
import json
import time
from pprint import pprint

# Configuration pour test local
LOCAL_API_URL = "http://localhost:5002"

def test_gdrive_integration():
    """Tester l'intégration complète avec Google Drive"""

    print("🚀 TESTING GDRIVE INTEGRATION")
    print("=" * 80)

    # Test 1: Vérifier que l'API démarre et charge les données
    print("\n📡 TEST 1: Health check with GDrive data")
    test_health_with_data()

    # Test 2: Lister les livres disponibles
    print("\n📚 TEST 2: List available books")
    test_list_books()

    # Test 3: Obtenir les infos détaillées d'un livre
    print("\n📖 TEST 3: Get book details")
    test_book_details()

    # Test 4: Tester une requête GraphRAG avec données GDrive
    print("\n🔍 TEST 4: GraphRAG query with GDrive data")
    test_graphrag_query()

    # Test 5: Tester le refresh des données
    print("\n🔄 TEST 5: Data refresh")
    test_data_refresh()

def test_health_with_data():
    """Test que l'API démarre avec les données GDrive"""
    try:
        response = requests.get(f"{LOCAL_API_URL}/health", timeout=10)

        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy")
            print(f"   Service: {data.get('service')}")
            print(f"   Neo4j: {data.get('connections', {}).get('neo4j', 'unknown')}")
            print(f"   GraphRAG: {data.get('connections', {}).get('graphrag', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_list_books():
    """Test la liste des livres disponibles"""
    try:
        response = requests.get(f"{LOCAL_API_URL}/books", timeout=15)

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                books = data.get('books', [])
                print(f"✅ Found {len(books)} books:")

                for book in books:
                    print(f"   📖 {book['id']}")
                    print(f"      Name: {book.get('name', 'N/A')}")
                    print(f"      Path: {book.get('path', 'N/A')}")

                    stats = book.get('stats', {})
                    if stats:
                        print(f"      Documents: {stats.get('documents', 'N/A')}")
                        print(f"      Entities: {stats.get('entities', 'N/A')}")

                return books
            else:
                print(f"❌ API error: {data.get('error', 'Unknown error')}")
                return []
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return []

    except Exception as e:
        print(f"❌ Error listing books: {e}")
        return []

def test_book_details():
    """Test les détails d'un livre spécifique"""
    book_id = "a_rebours_huysmans"

    try:
        response = requests.get(f"{LOCAL_API_URL}/books/{book_id}/info", timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                book = data.get('book', {})
                print(f"✅ Book details for {book_id}:")
                print(f"   Name: {book.get('name', 'N/A')}")
                print(f"   Path: {book.get('path', 'N/A')}")

                files = book.get('files', {})
                print(f"   Files available:")
                for file_type, file_info in files.items():
                    if file_info.get('exists'):
                        print(f"      ✅ {file_type}: {file_info.get('count', 'N/A')} items ({file_info.get('size', 0)} bytes)")
                    else:
                        print(f"      ❌ {file_type}: missing")

                return True
            else:
                print(f"❌ API error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error getting book details: {e}")
        return False

def test_graphrag_query():
    """Test une requête GraphRAG avec les données GDrive"""

    query_data = {
        'query': 'What are the main themes in this literary work?',
        'mode': 'local',
        'debug_mode': True,
        'book_id': 'a_rebours_huysmans'
    }

    try:
        print(f"🔍 Testing GraphRAG query with GDrive data...")
        start_time = time.time()

        response = requests.post(
            f"{LOCAL_API_URL}/query/local",
            json=query_data,
            timeout=45
        )

        elapsed = time.time() - start_time
        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print("✅ GraphRAG query successful!")
                print(f"📝 Answer length: {len(data.get('answer', ''))} chars")
                print(f"🔄 Processing time: {data.get('processing_time', 0):.2f}s")
                print(f"📚 Source: {data.get('source', 'N/A')}")

                # Afficher les stats d'interception
                debug_info = data.get('debug_info', {})
                if debug_info:
                    context_stats = debug_info.get('context_stats', {})
                    print(f"\n🔍 INTERCEPTOR DATA:")
                    print(f"   Prompt length: {context_stats.get('prompt_length', 0)} chars")
                    print(f"   Mode: {context_stats.get('mode', 'N/A')}")
                    print(f"   Query ID: {context_stats.get('query_id', 'N/A')}")

                    # Vérifier qu'on a des vraies données
                    processing_phases = debug_info.get('processing_phases', {})
                    entities = processing_phases.get('entity_selection', {}).get('entities', [])
                    communities = processing_phases.get('community_analysis', {}).get('communities', [])

                    print(f"   Entities captured: {len(entities)}")
                    print(f"   Communities captured: {len(communities)}")

                    # Vérifier les stats de l'intercepteur
                    interceptor_stats = data.get('interceptor_stats', {})
                    print(f"\n📈 INTERCEPTOR STATS:")
                    print(f"   Queries processed: {interceptor_stats.get('queries_processed', 0)}")
                    print(f"   Analysis available: {interceptor_stats.get('last_analysis_available', False)}")

                    # Valider que les données ne sont pas statiques
                    is_real_data = (
                        context_stats.get('prompt_length', 0) > 0 or
                        len(entities) > 0 or
                        len(communities) > 0
                    )

                    print(f"\n🎯 DATA VALIDATION:")
                    print(f"   Real intercepted data: {'✅ YES' if is_real_data else '❌ NO'}")

                return True
            else:
                print(f"❌ GraphRAG query failed: {data.get('error', 'Unknown error')}")
                if 'available_books' in data:
                    print(f"Available books: {data['available_books']}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            print(f"Response: {response.text[:300]}...")
            return False

    except Exception as e:
        print(f"❌ Error in GraphRAG query: {e}")
        return False

def test_data_refresh():
    """Test le rafraîchissement forcé des données"""

    try:
        print("🔄 Testing data refresh...")
        start_time = time.time()

        response = requests.post(f"{LOCAL_API_URL}/data/refresh", timeout=60)

        elapsed = time.time() - start_time
        print(f"⏱️  Refresh time: {elapsed:.2f}s")

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print("✅ Data refresh successful!")
                print(f"📂 Data path: {data.get('data_path', 'N/A')}")
                print(f"📚 Books found: {data.get('book_count', 0)}")
                print(f"📋 Available books: {data.get('available_books', [])}")
                return True
            else:
                print(f"❌ Data refresh failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Request failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Error in data refresh: {e}")
        return False

if __name__ == "__main__":
    print("🧪 STARTING GDRIVE INTEGRATION TESTS")
    print("=" * 80)
    print("⚠️  Make sure the API is running locally on port 5002")
    print("⚠️  This will download data from Google Drive (may take time)")

    input("Press Enter to continue...")

    test_gdrive_integration()

    print("\n" + "=" * 80)
    print("✅ GDRIVE INTEGRATION TESTS COMPLETED")