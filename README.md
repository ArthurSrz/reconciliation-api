# Reconciliation API - Borges Library

Central coordination layer that harmonizes data between Neo4j graph database and GraphRAG queries.

## Production API

**Base URL**: `https://reconciliation-api-production.up.railway.app`

### Quick Start Examples

#### Get all books
```bash
curl https://reconciliation-api-production.up.railway.app/books
```

Response:
```json
{
  "success": true,
  "source": "neo4j",
  "books": [
    {
      "id": "chien_blanc_gary",
      "name": "Chien blanc",
      "neo4j_id": "LIVRE_Chien blanc",
      "entity_count": 1245,
      "community_count": 156
    }
  ]
}
```

#### Get graph nodes (most central)
```bash
curl "https://reconciliation-api-production.up.railway.app/graph/nodes?limit=100"
```

Response:
```json
{
  "success": true,
  "count": 100,
  "limit": 100,
  "nodes": [
    {
      "id": "LIVRE_Chien blanc",
      "labels": ["Entity", "BOOK"],
      "centrality_score": 7065,
      "degree": 234,
      "properties": {
        "title": "Chien blanc",
        "author": "Romain Gary",
        "entity_type": "BOOK",
        "filesystem_id": "chien_blanc_gary"
      }
    }
  ]
}
```

#### Get relationships for specific nodes
```bash
curl "https://reconciliation-api-production.up.railway.app/graph/relationships?node_ids=LIVRE_Chien%20blanc,entity_123"
```

#### Search nodes
```bash
curl "https://reconciliation-api-production.up.railway.app/graph/search?q=Gary&type=PERSON&limit=50"
```

#### Get chunk content (for traceability)
```bash
curl "https://reconciliation-api-production.up.railway.app/chunks/chien_blanc_gary/chunk-abc123"
```

#### Query GraphRAG
```bash
curl -X POST https://reconciliation-api-production.up.railway.app/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main themes in Romain Gary works?", "book_ids": ["chien_blanc_gary"]}'
```

#### Get graph statistics
```bash
curl https://reconciliation-api-production.up.railway.app/stats
```

Response includes node counts (35,767 total), relationship counts (143,687 total), and breakdowns by type.

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