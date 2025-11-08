#!/usr/bin/env python3
"""
Test final du workflow Neo4j de l'API de réconciliation
Ce test vérifie que toute la partie Neo4j fonctionne correctement
"""

import requests
import json
import time
from typing import List, Dict, Any

RECONCILIATION_API_URL = "https://reconciliation-api-production.up.railway.app"

def test_neo4j_complete_workflow():
    """Test complet du workflow Neo4j uniquement"""
    print("🚀 TEST COMPLET WORKFLOW NEO4J")
    print("=" * 60)

    # Test 1: Health check
    print("🔍 1. Health Check")
    response = requests.get(f"{RECONCILIATION_API_URL}/health")
    health = response.json()
    neo4j_status = health['connections']['neo4j']

    if neo4j_status != 'connected':
        print(f"❌ Neo4j non connecté: {neo4j_status}")
        return False
    print(f"✅ Neo4j connecté")

    # Test 2: Récupération des nœuds les plus connectés
    print("\n🔍 2. Récupération des nœuds les plus connectés")
    response = requests.get(f"{RECONCILIATION_API_URL}/graph/nodes?limit=10")

    if response.status_code != 200:
        print(f"❌ Erreur {response.status_code}: {response.text}")
        return False

    nodes_data = response.json()
    if not nodes_data['success']:
        print(f"❌ Échec API: {nodes_data.get('error', 'Erreur inconnue')}")
        return False

    nodes = nodes_data['nodes']
    print(f"✅ {len(nodes)} nœuds récupérés")

    # Vérifier que nous avons bien les nœuds avec le plus de connexions
    if nodes:
        top_node = nodes[0]
        print(f"✅ Nœud le plus connecté: {top_node['properties'].get('name', 'N/A')} ({top_node['degree']} connexions)")

    # Test 3: Relations entre nœuds
    print("\n🔍 3. Test des relations")
    if len(nodes) >= 2:
        node_ids = [node['id'] for node in nodes[:5]]
        response = requests.get(
            f"{RECONCILIATION_API_URL}/graph/relationships",
            params={'node_ids': ','.join(node_ids)}
        )

        if response.status_code == 200:
            relations_data = response.json()
            if relations_data['success']:
                relations = relations_data['relationships']
                print(f"✅ {len(relations)} relations trouvées")
            else:
                print(f"❌ Erreur relations: {relations_data.get('error')}")
                return False
        else:
            print(f"❌ Erreur HTTP relations: {response.status_code}")
            return False

    # Test 4: Recherche
    print("\n🔍 4. Test de recherche")
    search_queries = ["Premier", "cordée", "livre"]

    for query in search_queries:
        response = requests.get(
            f"{RECONCILIATION_API_URL}/graph/search",
            params={'q': query, 'limit': 3}
        )

        if response.status_code == 200:
            search_data = response.json()
            if search_data['success']:
                results = search_data['nodes']
                print(f"✅ Recherche '{query}': {len(results)} résultats")
            else:
                print(f"❌ Erreur recherche '{query}': {search_data.get('error')}")
                return False
        else:
            print(f"❌ Erreur HTTP recherche '{query}': {response.status_code}")
            return False

    # Test 5: Statistiques
    print("\n🔍 5. Test des statistiques")
    response = requests.get(f"{RECONCILIATION_API_URL}/stats")

    if response.status_code == 200:
        stats_data = response.json()
        if stats_data['success']:
            stats = stats_data['stats']
            total_nodes = stats['total_nodes']
            total_rels = stats['total_relationships']
            print(f"✅ Statistiques: {total_nodes:,} nœuds, {total_rels:,} relations")
        else:
            print(f"❌ Erreur stats: {stats_data.get('error')}")
            return False
    else:
        print(f"❌ Erreur HTTP stats: {response.status_code}")
        return False

    # Test 6: Chargement progressif (comme prévu pour le frontend)
    print("\n🔍 6. Test chargement progressif")
    limits = [300, 500, 1000]

    for limit in limits:
        start_time = time.time()
        response = requests.get(f"{RECONCILIATION_API_URL}/graph/nodes?limit={limit}")
        end_time = time.time()

        if response.status_code == 200:
            data = response.json()
            if data['success']:
                count = data['count']
                duration = (end_time - start_time) * 1000
                print(f"✅ {limit} nœuds → {count} reçus en {duration:.0f}ms")
            else:
                print(f"❌ Erreur limite {limit}: {data.get('error')}")
                return False
        else:
            print(f"❌ Erreur HTTP limite {limit}: {response.status_code}")
            return False

    return True

def test_graphrag_connectivity():
    """Test de connectivité GraphRAG pour diagnostic"""
    print("\n🔍 7. Diagnostic GraphRAG")
    response = requests.get(f"{RECONCILIATION_API_URL}/health")
    health = response.json()
    graphrag_status = health['connections']['graphrag']

    print(f"GraphRAG Status: {graphrag_status}")

    if graphrag_status == 'connected':
        print("⚠️  GraphRAG health OK mais /query échoue (problème interne)")
        print("   Causes possibles:")
        print("   - Données GDrive pas synchronisées")
        print("   - Configuration nano-graphrag incorrecte")
        print("   - Problème avec les index GraphRAG")
    else:
        print(f"❌ GraphRAG non accessible: {graphrag_status}")

def main():
    """Test principal"""
    start_time = time.time()

    print("DÉBUT DU TEST DE L'API DE RÉCONCILIATION")
    print("Ce test vérifie que la partie Neo4j est fonctionnelle")
    print("=" * 60)

    # Test du workflow Neo4j
    neo4j_success = test_neo4j_complete_workflow()

    # Diagnostic GraphRAG
    test_graphrag_connectivity()

    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print(f"TEST TERMINÉ en {duration:.1f} secondes")
    print("=" * 60)

    if neo4j_success:
        print("✅ RÉSULTAT: API Neo4j COMPLÈTEMENT FONCTIONNELLE")
        print("\n🚀 PRÊT POUR LE FRONTEND:")
        print("   - Récupération des nœuds les plus connectés")
        print("   - Chargement progressif (300→500→1000)")
        print("   - Relations entre nœuds visibles")
        print("   - Recherche dans le graphe")
        print("   - Statistiques du graphe")

        print("\n⚠️  À CORRIGER:")
        print("   - API GraphRAG: problème interne à résoudre")
        print("   - Endpoint /query/reconciled indisponible temporairement")

        print(f"\n🔗 API URL: {RECONCILIATION_API_URL}")

        return True
    else:
        print("❌ RÉSULTAT: PROBLÈMES DÉTECTÉS DANS L'API")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)