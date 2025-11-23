#!/usr/bin/env python3
"""
Neo4j Sync CLI - Upload processed book data to Neo4j.

This CLI tool syncs book data from the local nano-graphRAG output
to the Neo4j database.

Usage:
    python -m cli.sync_to_neo4j --book-dir /path/to/book_data/book_name

    # With book metadata
    python -m cli.sync_to_neo4j --book-dir /path/to/book_data/book_name --title "Book Title" --author "Author"

Constitution Compliance:
- Principle II: Books are core entities - creates BOOK node as hub
- Principle III: No orphan nodes - links all entities to book
- Principle V: Inter-book exploration - links to existing entities
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from services.neo4j_client import Neo4jClient


BOOK_DATA_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data"


class Neo4jSyncService:
    """Service to sync local GraphRAG data to Neo4j."""

    def __init__(self):
        self.client = Neo4jClient()

    async def verify_connection(self) -> bool:
        """Verify Neo4j connection."""
        return await self.client.verify_connection()

    async def close(self):
        """Close connection."""
        await self.client.close()

    async def sync_book(
        self,
        book_dir: str,
        title: str,
        author: str,
        genre: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sync a processed book to Neo4j.

        Args:
            book_dir: Path to the book's processed data directory
            title: Book title
            author: Book author
            genre: Optional genre

        Returns:
            Dict with sync statistics
        """
        book_path = Path(book_dir)

        # Validate directory exists
        if not book_path.exists():
            raise ValueError(f"Book directory not found: {book_dir}")

        # Load GraphML
        graphml_path = book_path / "graph_chunk_entity_relation.graphml"
        if not graphml_path.exists():
            raise ValueError(f"GraphML file not found: {graphml_path}")

        print(f"\n📚 Syncing book to Neo4j: {title}")
        print(f"   Directory: {book_dir}")

        # Load the graph
        print("   Loading GraphML...")
        G = nx.read_graphml(str(graphml_path))

        # Generate book ID from directory name
        book_id = f"LIVRE_{title}"

        # Stats
        stats = {
            'book_id': book_id,
            'nodes_processed': 0,
            'edges_processed': 0,
            'communities_synced': 0,
            'chunks_synced': 0,
            'errors': []
        }

        async with self.client.driver.session() as session:
            # Step 1: Create BOOK node
            print("   Creating BOOK node...")
            await session.run(
                """
                MERGE (b:BOOK {id: $book_id})
                SET b.title = $title,
                    b.author = $author,
                    b.genre = $genre,
                    b.source_dir = $source_dir,
                    b.synced_at = datetime()
                """,
                book_id=book_id,
                title=title,
                author=author,
                genre=genre,
                source_dir=str(book_dir)
            )

            # Step 2: Create Entity nodes
            print(f"   Creating {G.number_of_nodes()} entity nodes...")
            for node_id, node_data in G.nodes(data=True):
                try:
                    # Clean node ID (remove quotes)
                    clean_id = node_id.strip('"')
                    entity_type = node_data.get('entity_type', 'UNKNOWN').strip('"')
                    description = node_data.get('description', '')
                    source_chunks = node_data.get('source_id', '')
                    clusters = node_data.get('clusters', '')

                    await session.run(
                        """
                        MERGE (e:Entity {id: $entity_id})
                        SET e.entity_type = $entity_type,
                            e.description = $description,
                            e.source_id = $book_id,
                            e.source_chunks = $source_chunks,
                            e.clusters = $clusters,
                            e.book_dir = $book_dir
                        WITH e
                        MATCH (b:BOOK {id: $book_id})
                        MERGE (b)-[:CONTAINS_ENTITY]->(e)
                        """,
                        entity_id=clean_id,
                        entity_type=entity_type,
                        description=description,
                        book_id=book_id,
                        source_chunks=source_chunks,
                        clusters=clusters,
                        book_dir=str(book_dir)
                    )
                    stats['nodes_processed'] += 1
                except Exception as e:
                    stats['errors'].append(f"Node {node_id}: {str(e)}")

            # Step 3: Create relationships
            print(f"   Creating {G.number_of_edges()} relationships...")
            for source, target, edge_data in G.edges(data=True):
                try:
                    clean_source = source.strip('"')
                    clean_target = target.strip('"')
                    weight = float(edge_data.get('weight', 1.0))
                    description = edge_data.get('description', '')
                    source_chunks = edge_data.get('source_id', '')

                    await session.run(
                        """
                        MATCH (s:Entity {id: $source})
                        MATCH (t:Entity {id: $target})
                        MERGE (s)-[r:RELATED_TO]->(t)
                        SET r.weight = $weight,
                            r.description = $description,
                            r.source_id = $book_id,
                            r.source_chunks = $source_chunks
                        """,
                        source=clean_source,
                        target=clean_target,
                        weight=weight,
                        description=description,
                        book_id=book_id,
                        source_chunks=source_chunks
                    )
                    stats['edges_processed'] += 1
                except Exception as e:
                    stats['errors'].append(f"Edge {source}->{target}: {str(e)}")

            # Step 4: Sync community reports
            community_file = book_path / "kv_store_community_reports.json"
            if community_file.exists():
                print("   Syncing community reports...")
                with open(community_file) as f:
                    communities = json.load(f)

                for comm_id, comm_data in communities.items():
                    try:
                        # comm_data is typically a string with the report
                        report_text = comm_data if isinstance(comm_data, str) else json.dumps(comm_data)

                        await session.run(
                            """
                            MERGE (c:Community {id: $comm_id, book_id: $book_id})
                            SET c.report = $report,
                                c.source_id = $book_id
                            WITH c
                            MATCH (b:BOOK {id: $book_id})
                            MERGE (b)-[:HAS_COMMUNITY]->(c)
                            """,
                            comm_id=comm_id,
                            book_id=book_id,
                            report=report_text[:5000]  # Truncate long reports
                        )
                        stats['communities_synced'] += 1
                    except Exception as e:
                        stats['errors'].append(f"Community {comm_id}: {str(e)}")

            # Step 5: Sync text chunks
            chunks_file = book_path / "kv_store_text_chunks.json"
            if chunks_file.exists():
                print("   Syncing text chunks...")
                with open(chunks_file) as f:
                    chunks = json.load(f)

                for chunk_id, chunk_data in chunks.items():
                    try:
                        chunk_text = chunk_data if isinstance(chunk_data, str) else chunk_data.get('content', '')

                        await session.run(
                            """
                            MERGE (ch:Chunk {id: $chunk_id})
                            SET ch.content = $content,
                                ch.source_id = $book_id
                            WITH ch
                            MATCH (b:BOOK {id: $book_id})
                            MERGE (b)-[:HAS_CHUNK]->(ch)
                            """,
                            chunk_id=chunk_id,
                            book_id=book_id,
                            content=chunk_text[:5000]  # Truncate long chunks
                        )
                        stats['chunks_synced'] += 1
                    except Exception as e:
                        stats['errors'].append(f"Chunk {chunk_id}: {str(e)}")

            # Step 6: Create inter-book links
            print("   Creating inter-book entity links...")
            inter_book_result = await session.run(
                """
                MATCH (new_entity:Entity {source_id: $book_id})
                MATCH (existing:Entity)
                WHERE existing.source_id <> $book_id
                AND toLower(new_entity.id) = toLower(existing.id)
                MERGE (new_entity)-[r:SAME_AS]->(existing)
                SET r.inter_book = true,
                    r.created_at = datetime()
                RETURN count(r) as link_count
                """,
                book_id=book_id
            )
            record = await inter_book_result.single()
            stats['inter_book_links'] = record['link_count'] if record else 0

        return stats


async def sync_book_to_neo4j(
    book_dir: str,
    title: str,
    author: str,
    genre: Optional[str] = None
) -> Dict[str, Any]:
    """Main sync function."""
    service = Neo4jSyncService()

    try:
        # Verify connection
        print("\n🔌 Connecting to Neo4j...")
        if not await service.verify_connection():
            raise ConnectionError("Could not connect to Neo4j. Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD env vars.")

        print("   Connected!")

        # Sync
        stats = await service.sync_book(book_dir, title, author, genre)

        return stats

    finally:
        await service.close()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Sync processed book data to Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync a specific book directory
  python -m cli.sync_to_neo4j --book-dir /path/to/la_maison_vide_laurent_mauvignier --title "La Maison Vide" --author "Laurent Mauvignier"

  # List available books to sync
  python -m cli.sync_to_neo4j --list

Environment:
  NEO4J_URI        Neo4j connection URI (default: bolt://localhost:7687)
  NEO4J_USER       Neo4j username (default: neo4j)
  NEO4J_PASSWORD   Neo4j password (required)
        """
    )

    parser.add_argument('--book-dir', '-d', help='Path to processed book directory')
    parser.add_argument('--title', '-t', help='Book title')
    parser.add_argument('--author', '-a', help='Author name')
    parser.add_argument('--genre', '-g', help='Literary genre (optional)')
    parser.add_argument('--list', '-l', action='store_true', help='List available books in book_data directory')

    return parser


def list_available_books():
    """List books available for syncing."""
    book_data = Path(BOOK_DATA_DIR)

    if not book_data.exists():
        print(f"❌ Book data directory not found: {BOOK_DATA_DIR}")
        return

    print(f"\n📂 Available books in {BOOK_DATA_DIR}:")
    print(f"{'='*60}")

    for d in sorted(book_data.iterdir()):
        if d.is_dir():
            graphml = d / "graph_chunk_entity_relation.graphml"
            if graphml.exists():
                # Count entities
                try:
                    G = nx.read_graphml(str(graphml))
                    print(f"   📖 {d.name}")
                    print(f"      Entities: {G.number_of_nodes()}, Relations: {G.number_of_edges()}")
                except:
                    print(f"   📖 {d.name} (error reading GraphML)")
            else:
                print(f"   ⚠️  {d.name} (no GraphML file)")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.list:
        list_available_books()
        return

    # Validate arguments
    if not args.book_dir:
        parser.error("--book-dir is required")
    if not args.title:
        parser.error("--title is required")
    if not args.author:
        parser.error("--author is required")

    # Run sync
    stats = asyncio.run(sync_book_to_neo4j(
        book_dir=args.book_dir,
        title=args.title,
        author=args.author,
        genre=args.genre
    ))

    # Print results
    print(f"\n{'='*60}")
    print(f"✅ Sync completed!")
    print(f"   Book ID: {stats['book_id']}")
    print(f"   Entities synced: {stats['nodes_processed']}")
    print(f"   Relationships synced: {stats['edges_processed']}")
    print(f"   Communities synced: {stats['communities_synced']}")
    print(f"   Chunks synced: {stats['chunks_synced']}")
    print(f"   Inter-book links: {stats.get('inter_book_links', 0)}")

    if stats['errors']:
        print(f"\n⚠️  Errors ({len(stats['errors'])}):")
        for err in stats['errors'][:5]:
            print(f"      - {err}")
        if len(stats['errors']) > 5:
            print(f"      ... and {len(stats['errors']) - 5} more")

    print(f"{'='*60}")


if __name__ == '__main__':
    main()
