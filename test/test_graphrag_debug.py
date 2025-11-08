#!/usr/bin/env python3
"""
Test de diagnostic pour identifier le problème GraphRAG
"""

import requests
import json

GRAPHRAG_API_URL = "https://comfortable-gentleness-production-8603.up.railway.app"

def test_graphrag_endpoints():
    """Test tous les endpoints GraphRAG pour identifier le problème"""
    print("🔍 DIAGNOSTIC API GRAPHRAG")
    print("=" * 50)
    print(f"URL: {GRAPHRAG_API_URL}")
    print("=" * 50)

    # Test 1: Health
    print("\n1. Test /health")
    try:
        response = requests.get(f"{GRAPHRAG_API_URL}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 2: Root endpoint
    print("\n2. Test / (root)")
    try:
        response = requests.get(f"{GRAPHRAG_API_URL}/", timeout=10)
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 3: Query endpoint avec différents payloads
    print("\n3. Test /query avec payload minimal")
    try:
        response = requests.post(
            f"{GRAPHRAG_API_URL}/query",
            json={"query": "test"},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 4: Query avec mode spécifié
    print("\n4. Test /query avec mode local")
    try:
        response = requests.post(
            f"{GRAPHRAG_API_URL}/query",
            json={"query": "test", "mode": "local"},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

    # Test 5: Vérifier les headers
    print("\n5. Test headers et content-type")
    try:
        response = requests.post(
            f"{GRAPHRAG_API_URL}/query",
            json={"query": "hello", "mode": "local"},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        print(f"   Status: {response.status_code}")
        print(f"   Headers: {dict(response.headers)}")
        print(f"   Response: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Erreur: {e}")

if __name__ == "__main__":
    test_graphrag_endpoints()