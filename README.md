# Reconciliation API - Borges Library

Central coordination layer that harmonizes data between Neo4j graph database and GraphRAG queries.

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

### Graph Operations
- `GET /graph/nodes?limit=300&centrality_type=degree` - Get most central nodes
- `GET /graph/relationships?node_ids=id1,id2,id3` - Get relationships for nodes
- `GET /graph/search?q=search_term&type=node_type&limit=50` - Search nodes

### Query Operations
- `POST /query/reconciled` - Reconciled GraphRAG query with Neo4j context

### System
- `GET /health` - Health check with connection status
- `GET /stats` - Graph statistics

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