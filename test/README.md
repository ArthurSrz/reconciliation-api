# Tests de l'API de Réconciliation

Ce dossier contient tous les tests et la documentation de test pour l'API.

## Fichiers

- `test_neo4j_connection.py` - Script pour tester différentes méthodes de connexion Neo4j
- `TESTING_RESULTS.md` - Résultats détaillés de tous les tests effectués
- `BLOCKERS.md` - Explication détaillée de ce qui empêche l'API de fonctionner

## Exécuter les tests

### Test de connexion Neo4j
```bash
cd test
python3 test_neo4j_connection.py
```

Ce script teste automatiquement tous les protocoles de connexion supportés par Neo4j :
- neo4j+s:// (recommandé pour Aura)
- neo4j+ssc://
- bolt+s://
- bolt+ssc://
- neo4j://
- bolt://

## Voir les résultats

Consultez `TESTING_RESULTS.md` pour les résultats complets des tests effectués.
