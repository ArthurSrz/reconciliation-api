#!/usr/bin/env python3
"""
Test du nouveau système d'interception GraphRAG
Vérifie que l'API fournit maintenant des données réelles comme test_query_analysis.py
"""

import requests
import json
import time
from pprint import pprint

# Configuration
API_BASE_URL = "https://reconciliation-api-production.up.railway.app"
LOCAL_API_URL = "http://localhost:5002"

def test_new_interceptor_api():
    """Tester le nouveau système d'interception"""

    print("🧪 TESTING NEW GRAPHRAG INTERCEPTOR")
    print("=" * 80)

    test_queries = [
        "What are the main themes in A Christmas Carol?",
        "Who is Ebenezer Scrooge and what is his relationship with Bob Cratchit?",
        "How do the Christmas ghosts influence Scrooge's transformation?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 TEST #{i}: {query}")
        print("-" * 60)

        # Test avec l'API déployée sur Railway
        test_query_api(API_BASE_URL, query, f"Production API - Query {i}")

def test_query_api(base_url: str, query: str, test_name: str):
    """Test une requête sur une API spécifique"""

    print(f"\n📡 Testing {test_name}")
    print(f"URL: {base_url}")

    try:
        # Test l'endpoint /query/local avec le nouvel intercepteur
        payload = {
            'query': query,
            'mode': 'local',
            'debug_mode': True
        }

        start_time = time.time()
        response = requests.post(
            f"{base_url}/query/local",
            json=payload,
            timeout=30
        )
        elapsed = time.time() - start_time

        print(f"⏱️  Response time: {elapsed:.2f}s")
        print(f"📊 Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()

            print(f"✅ SUCCESS!")
            print(f"📝 Answer length: {len(data.get('answer', ''))} chars")

            # Analyser les données de debug
            debug_info = data.get('debug_info', {})
            if debug_info:
                context_stats = debug_info.get('context_stats', {})
                processing_phases = debug_info.get('processing_phases', {})

                print(f"\n📊 INTERCEPTOR ANALYSIS:")
                print(f"   🔍 Prompt length: {context_stats.get('prompt_length', 0)} chars")
                print(f"   ⏱️  Total processing: {context_stats.get('total_time_ms', 0)}ms")
                print(f"   🆔 Query ID: {context_stats.get('query_id', 'N/A')}")

                # Entités interceptées
                entities = processing_phases.get('entity_selection', {}).get('entities', [])
                print(f"   👥 Entities captured: {len(entities)}")
                for j, entity in enumerate(entities[:5], 1):  # Premier 5
                    print(f"      {j}. {entity.get('name', 'N/A')} (score: {entity.get('score', 0):.3f})")

                # Communautés interceptées
                communities = processing_phases.get('community_analysis', {}).get('communities', [])
                print(f"   🏘️  Communities captured: {len(communities)}")
                for j, comm in enumerate(communities[:3], 1):  # Premier 3
                    print(f"      {j}. Community {comm.get('id', 'N/A')}: {comm.get('title', 'No title')}")

                # Relations interceptées
                relationships = processing_phases.get('relationship_mapping', {}).get('relationships', [])
                print(f"   🔗 Relationships captured: {len(relationships)}")
                for j, rel in enumerate(relationships[:3], 1):  # Premier 3
                    print(f"      {j}. {rel.get('source', 'N/A')} → {rel.get('target', 'N/A')}")

                # Stats de l'intercepteur
                interceptor_stats = data.get('interceptor_stats', {})
                print(f"\n🔧 INTERCEPTOR STATS:")
                print(f"   📊 Queries processed: {interceptor_stats.get('queries_processed', 0)}")
                print(f"   💾 Last analysis available: {interceptor_stats.get('last_analysis_available', False)}")

                # Vérification que c'est des vraies données (pas statiques)
                is_real_data = any([
                    context_stats.get('prompt_length', 0) > 0,
                    len(entities) > 0,
                    len(communities) > 0,
                    len(relationships) > 0
                ])

                print(f"\n🎯 DATA QUALITY:")
                print(f"   ✅ Real intercepted data: {'YES' if is_real_data else 'NO'}")
                print(f"   📊 Mode: {context_stats.get('mode', 'unknown')}")

            else:
                print("❌ No debug info available")

        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(f"Response: {response.text[:200]}...")

    except requests.exceptions.Timeout:
        print("⏰ TIMEOUT after 30s")
    except requests.exceptions.ConnectionError:
        print("🔌 CONNECTION ERROR")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def test_health_endpoint():
    """Test que l'API est en ligne"""
    print("🏥 Testing health endpoint...")

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ API is healthy")
            print(f"   Neo4j: {data.get('connections', {}).get('neo4j', 'unknown')}")
            print(f"   GraphRAG: {data.get('connections', {}).get('graphrag', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING INTERCEPTOR TESTS")
    print("=" * 80)

    # Test santé de l'API
    if test_health_endpoint():
        # Test du nouveau système d'interception
        test_new_interceptor_api()
    else:
        print("❌ API not available, skipping tests")

    print("\n" + "=" * 80)
    print("✅ TESTS COMPLETED")