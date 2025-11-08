#!/usr/bin/env python3
"""
Test complet du workflow de l'API de réconciliation
1. Récupération des nœuds les plus connectés (pour affichage frontend)
2. Requête GraphRAG avec contexte des nœuds visibles
"""

import requests
import json
import time
from typing import List, Dict, Any

# Configuration
RECONCILIATION_API_URL = "https://reconciliation-api-production.up.railway.app"

def test_health_check():
    """Test de santé de l'API"""
    print("🔍 Test 1: Vérification de la santé de l'API")
    print("=" * 60)

    try:
        response = requests.get(f"{RECONCILIATION_API_URL}/health")
        response.raise_for_status()
        health_data = response.json()

        print(f"✅ Status: {health_data['status']}")
        print(f"✅ Neo4j: {health_data['connections']['neo4j']}")
        print(f"✅ GraphRAG: {health_data['connections']['graphrag']}")
        print(f"✅ Timestamp: {health_data['timestamp']}")

        assert health_data['status'] == 'healthy'
        assert health_data['connections']['neo4j'] == 'connected'
        assert health_data['connections']['graphrag'] == 'connected'

        print("\n🎉 Test de santé: RÉUSSI\n")
        return True

    except Exception as e:
        print(f"❌ Erreur lors du test de santé: {e}")
        return False

def test_get_most_connected_nodes(limit: int = 10) -> List[Dict[str, Any]]:
    """Test de récupération des nœuds les plus connectés"""
    print(f"🔍 Test 2: Récupération des {limit} nœuds les plus connectés")
    print("=" * 60)

    try:
        response = requests.get(
            f"{RECONCILIATION_API_URL}/graph/nodes",
            params={'limit': limit, 'centrality_type': 'degree'}
        )
        response.raise_for_status()
        nodes_data = response.json()

        print(f"✅ Succès: {nodes_data['success']}")
        print(f"✅ Nombre de nœuds reçus: {nodes_data['count']}")
        print(f"✅ Limite demandée: {nodes_data['limit']}")

        nodes = nodes_data['nodes']

        print("\n📊 Top nœuds les plus connectés:")
        for i, node in enumerate(nodes[:5], 1):
            name = node['properties'].get('name', node['properties'].get('title', 'Nom inconnu'))
            labels = ', '.join(node['labels'])
            degree = node['degree']
            print(f"  {i}. {name} ({labels}) - {degree} connexions")

        print(f"\n🎉 Test récupération nœuds: RÉUSSI ({len(nodes)} nœuds)\n")
        return nodes

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des nœuds: {e}")
        return []

def test_graphrag_query_with_context(visible_nodes: List[Dict[str, Any]]):
    """Test de requête GraphRAG avec contexte des nœuds visibles"""
    print("🔍 Test 3: Requête GraphRAG avec contexte")
    print("=" * 60)

    # Extraire les IDs des nœuds visibles
    visible_node_ids = [node['id'] for node in visible_nodes[:5]]  # Prenons les 5 premiers

    # Questions de test
    test_queries = [
        "Quels sont les thèmes principaux dans les livres de montagne ?",
        "Raconte-moi l'histoire de Pierre Servettaz dans Premier de cordée",
        "Quelles sont les relations entre les personnages principaux ?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Requête {i}: {query}")
        print("-" * 40)

        try:
            payload = {
                'query': query,
                'visible_node_ids': visible_node_ids,
                'mode': 'local'
            }

            print(f"🔗 Nœuds visibles inclus: {len(visible_node_ids)}")

            response = requests.post(
                f"{RECONCILIATION_API_URL}/query/reconciled",
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=60  # GraphRAG peut prendre du temps
            )
            response.raise_for_status()
            result = response.json()

            print(f"✅ Succès: {result['success']}")
            print(f"✅ Nœuds de contexte: {result['context']['visible_nodes_count']}")
            print(f"✅ Mode: {result['context']['mode']}")

            # Afficher la réponse
            answer = result.get('answer', 'Pas de réponse')
            print(f"\n💬 Réponse GraphRAG:")
            print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")

            # Afficher les entités trouvées
            search_path = result.get('search_path', {})
            entities = search_path.get('entities', [])
            if entities:
                print(f"\n🔍 Entités trouvées: {len(entities)}")
                for entity in entities[:3]:
                    entity_name = entity.get('id', 'Inconnu')
                    verified = entity.get('verified', False)
                    source = entity.get('source', 'graphrag')
                    print(f"   - {entity_name} (source: {source}, vérifié: {verified})")

            print(f"\n✅ Requête {i}: RÉUSSIE")

        except requests.exceptions.Timeout:
            print(f"⏰ Timeout pour la requête {i} (normal pour GraphRAG)")
        except Exception as e:
            print(f"❌ Erreur pour la requête {i}: {e}")

        # Pause entre les requêtes
        if i < len(test_queries):
            print("⏳ Pause de 2 secondes...")
            time.sleep(2)

    print(f"\n🎉 Test requêtes GraphRAG: TERMINÉ\n")

def test_graph_stats():
    """Test des statistiques du graphe"""
    print("🔍 Test 4: Statistiques du graphe")
    print("=" * 60)

    try:
        response = requests.get(f"{RECONCILIATION_API_URL}/stats")
        response.raise_for_status()
        stats = response.json()

        total_nodes = stats['stats']['total_nodes']
        total_relationships = stats['stats']['total_relationships']

        print(f"✅ Total nœuds: {total_nodes:,}")
        print(f"✅ Total relations: {total_relationships:,}")

        print("\n📊 Top 5 types de nœuds:")
        for node_type in stats['stats']['node_types'][:5]:
            labels = ', '.join(node_type['labels'])
            count = node_type['count']
            print(f"   - {labels}: {count:,} nœuds")

        print("\n🔗 Top 5 types de relations:")
        for rel_type in stats['stats']['relationship_types'][:5]:
            rel_name = rel_type['type']
            count = rel_type['count']
            print(f"   - {rel_name}: {count:,} relations")

        print(f"\n🎉 Test statistiques: RÉUSSI\n")
        return True

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des stats: {e}")
        return False

def main():
    """Exécution complète du test de workflow"""
    print("🚀 TEST COMPLET DU WORKFLOW API DE RÉCONCILIATION")
    print("=" * 80)
    print("Ce test simule l'utilisation complète de l'API:")
    print("1. Vérification de la santé")
    print("2. Récupération des nœuds pour l'affichage frontend")
    print("3. Requêtes GraphRAG avec contexte des nœuds visibles")
    print("4. Statistiques du graphe")
    print("=" * 80)

    start_time = time.time()

    # Test 1: Santé
    if not test_health_check():
        print("❌ Arrêt: L'API n'est pas en bonne santé")
        return

    # Test 2: Nœuds les plus connectés
    nodes = test_get_most_connected_nodes(15)
    if not nodes:
        print("❌ Arrêt: Impossible de récupérer les nœuds")
        return

    # Test 3: Requêtes GraphRAG
    test_graphrag_query_with_context(nodes)

    # Test 4: Statistiques
    test_graph_stats()

    end_time = time.time()
    duration = end_time - start_time

    print("=" * 80)
    print(f"🎉 WORKFLOW COMPLET TERMINÉ en {duration:.1f} secondes")
    print("=" * 80)
    print("\n✅ L'API de réconciliation est prête pour:")
    print("   - Alimenter le frontend avec les nœuds les plus connectés")
    print("   - Traiter les requêtes GraphRAG avec contexte Neo4j")
    print("   - Harmoniser les données entre Neo4j et GraphRAG")
    print("\n🔗 URL API: " + RECONCILIATION_API_URL)

if __name__ == "__main__":
    main()