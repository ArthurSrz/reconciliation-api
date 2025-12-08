# Reconciliation API - Borges Library

Central coordination layer that harmonizes data between Neo4j graph database and GraphRAG queries.

## Production API

**Base URL**: `https://reconciliation-api-production.up.railway.app`

All examples use Marcel Proust's *Du côté de chez Swann* as reference.

### 1. List all books

```bash
curl https://reconciliation-api-production.up.railway.app/books
```

```json
{
  "success": true,
  "source": "neo4j",
  "books": [
    {
      "id": "du_côté_de_chez_swann_marcel_proust",
      "name": "Du côté de chez Swann",
      "neo4j_id": "LIVRE_Du côté de chez Swann",
      "entity_count": 1819,
      "community_count": 243
    }
  ]
}
```

### 2. Get graph nodes (most central)

```bash
curl "https://reconciliation-api-production.up.railway.app/graph/nodes?limit=100"
```

```json
{
  "success": true,
  "count": 100,
  "limit": 100,
  "nodes": [
    {
      "id": "LIVRE_Du côté de chez Swann",
      "labels": ["Entity", "BOOK"],
      "centrality_score": 5240,
      "degree": 189,
      "properties": {
        "title": "Du côté de chez Swann",
        "author": "Marcel Proust",
        "entity_type": "BOOK",
        "filesystem_id": "du_côté_de_chez_swann_marcel_proust",
        "genre": "Fiction"
      }
    },
    {
      "id": "MARCELO PROUST",
      "labels": ["Entity"],
      "properties": {
        "entity_type": "PERSON",
        "description": "Marcel Proust est un auteur français, né en 1871 à Auteuil, connu pour son œuvre majeure 'À la recherche du temps perdu'."
      }
    }
  ]
}
```

### 3. Get relationships for nodes

```bash
curl "https://reconciliation-api-production.up.railway.app/graph/relationships?node_ids=MARCELO%20PROUST,AUTEUIL"
```

```json
{
  "success": true,
  "count": 5,
  "relationships": [
    {
      "source": "MARCELO PROUST",
      "target": "ACHILLE ADRIEN PROUST",
      "type": "RELATED_TO",
      "properties": {
        "description": "Marcel Proust est le fils d'Achille Adrien Proust, qui était un médecin important dans le domaine de l'épidémiologie."
      }
    },
    {
      "source": "MARCELO PROUST",
      "target": "À LA RECHERCHE DU TEMPS PERDU",
      "type": "RELATED_TO",
      "properties": {
        "description": "Marcel Proust est l'auteur de 'À la recherche du temps perdu', une œuvre fondatrice de la littérature moderne."
      }
    }
  ]
}
```

### 4. Search nodes

```bash
curl "https://reconciliation-api-production.up.railway.app/graph/search?q=Swann&limit=10"
```

```json
{
  "success": true,
  "count": 1,
  "query": "Swann",
  "nodes": [
    {
      "id": "LIVRE_Du côté de chez Swann",
      "labels": ["Entity", "BOOK"],
      "properties": {
        "title": "Du côté de chez Swann",
        "author": "Marcel Proust"
      }
    }
  ]
}
```

### 5. Get chunk content (traceability)

Retrieve the original text passage from which entities were extracted:

```bash
curl "https://reconciliation-api-production.up.railway.app/chunks/du_côté_de_chez_swann_marcel_proust/chunk-e63c089bf9368c76a3ca3ce21d3c88dc"
```

```json
{
  "success": true,
  "book_id": "du_côté_de_chez_swann_marcel_proust",
  "chunk_id": "chunk-e63c089bf9368c76a3ca3ce21d3c88dc",
  "chunk_order_index": 0,
  "content": "Marcel Proust est né le 10 juillet 1871 à Auteuil...",
  "tokens": 1200,
  "full_doc_id": "doc-4a1735a42b2e0323fe261ef10103d395",
  "source": "filesystem",
  "index_source": "neo4j_mdm",
  "traceability": {
    "source_type": "filesystem_chunk",
    "pipeline": ["tokenization", "embedding", "entity_extraction"],
    "processing_chain": "raw_text → chunks → entities → graph"
  }
}
```

### 6. Query GraphRAG

```bash
curl -X POST https://reconciliation-api-production.up.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quelle est la relation entre Marcel Proust et son père?",
    "book_ids": ["du_côté_de_chez_swann_marcel_proust"]
  }'
```

### 7. Query multiple books in parallel

```bash
curl -X POST https://reconciliation-api-production.up.railway.app/query/multi-book \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quels sont les thèmes de la mémoire dans ces œuvres?"
  }'
```

### 8. Health check

```bash
curl https://reconciliation-api-production.up.railway.app/health
```

```json
{
  "status": "healthy",
  "timestamp": "2025-12-08T11:02:36Z",
  "connections": {
    "neo4j": "connected",
    "graphrag": "connected"
  }
}
```

### 9. Graph statistics

```bash
curl https://reconciliation-api-production.up.railway.app/stats
```

```json
{
  "success": true,
  "nodes": {
    "total": 35767,
    "by_type": {
      "Entity": 26785,
      "Community": 3761,
      "Chunk": 2890,
      "BOOK": 20,
      "PERSON": 129,
      "GEO": 300
    }
  },
  "relationships": {
    "total": 143687,
    "by_type": {
      "RELATED_TO": 51961,
      "EXTRACTED_FROM": 45573,
      "CONTAINS_ENTITY": 32401,
      "HAS_COMMUNITY": 3407
    }
  }
}
```

---

## Architecture

```
Frontend (Vercel) → Reconciliation API (Railway) → {
    Neo4j (source of truth for graph structure)
    GraphRAG API (Railway) → Google Drive books
}
```

## Features

- **Progressive Graph Loading**: Load 300 → 400 → 500 → 1000 most central nodes
- **Context-Aware GraphRAG**: Query GraphRAG with visible nodes as context
- **Data Reconciliation**: Neo4j as source of truth for conflicts
- **Real-time Graph Search**: Search and filter nodes dynamically

## Endpoints

### Books
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/books` | List all books with metadata (entity_count, community_count) |

### Graph Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/graph/nodes?limit=300` | Get most central nodes (progressive loading) |
| GET | `/graph/relationships?node_ids=id1,id2` | Get relationships for specific nodes |
| GET | `/graph/search?q=term&type=PERSON&limit=50` | Search nodes by name and type |
| POST | `/graph/search-nodes` | Extract nodes from GraphRAG query |

### Chunk Retrieval (Traceability)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/chunks/<book_id>/<chunk_id>` | Get chunk text content |
| GET | `/chunks/find/<chunk_id>` | Find chunk without knowing book |

### Query Operations
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | GraphRAG query (alias: `/query/reconciled`, `/graphrag/query`) |
| POST | `/query/multi-book` | Query ALL books in parallel |
| POST | `/query/local` | Test local GraphRAG |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with Neo4j/GraphRAG status |
| GET | `/stats` | Graph statistics (node/relationship counts) |

## Environment Variables

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GRAPHRAG_API_URL=https://your-graphrag-api.railway.app
PORT=5002
```

## Development

```bash
pip install -r requirements.txt
python reconciliation_api.py
```

## Deployment (Railway)

1. Create new Railway project
2. Connect to this repository
3. Set environment variables
4. Deploy automatically

## Query Flow

1. **Frontend** sends query + visible node IDs
2. **Reconciliation API** fetches node details from Neo4j
3. **Context Enhancement** adds visible nodes to GraphRAG query
4. **GraphRAG Query** processes enhanced query
5. **Reconciliation** merges results with Neo4j as source of truth
6. **Return** harmonized response to frontend