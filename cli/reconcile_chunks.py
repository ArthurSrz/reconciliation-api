#!/usr/bin/env python3
"""
Chunk Reconciliation CLI - Index all filesystem chunks into Neo4j.

This script builds Neo4j as the Master Data Management (MDM) index for chunks:
- Scans all kv_store_text_chunks.json files in filesystem
- Creates lightweight Chunk nodes (id + book_filesystem_id, no content)
- Links chunks to BOOK nodes via HAS_CHUNK
- Links entities to chunks via EXTRACTED_FROM

Usage:
    python -m cli.reconcile_chunks

    # Dry run (show what would be done)
    python -m cli.reconcile_chunks --dry-run

    # Only index chunks (skip entity linking)
    python -m cli.reconcile_chunks --index-only

Constitution Compliance:
- Principle V: End-to-end traceability - links entities to source chunks
- Principle II: Books are core entities - chunks linked via BOOK nodes
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.neo4j_client import Neo4jClient


BOOK_DATA_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data"


class ChunkReconciliationService:
    """Service to reconcile chunks between filesystem and Neo4j."""

    def __init__(self):
        self.client = Neo4jClient()

    async def verify_connection(self) -> bool:
        """Verify Neo4j connection."""
        return await self.client.verify_connection()

    async def close(self):
        """Close connection."""
        await self.client.close()

    async def get_book_filesystem_mapping(self) -> Dict[str, str]:
        """Get mapping of BOOK id to filesystem_id from Neo4j."""
        query = """
        MATCH (b:BOOK)
        WHERE b.filesystem_id IS NOT NULL
        RETURN b.id as neo4j_id, b.filesystem_id as filesystem_id
        """
        async with self.client.driver.session() as session:
            result = await session.run(query)
            records = await result.data()
            return {r['filesystem_id']: r['neo4j_id'] for r in records}

    async def index_filesystem_chunks(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Index all chunks from filesystem into Neo4j.

        Creates lightweight Chunk nodes with just id and book_filesystem_id.
        Links chunks to BOOK nodes via HAS_CHUNK relationship.
        """
        book_data_dir = Path(BOOK_DATA_DIR)
        stats = {
            'books_processed': 0,
            'chunks_indexed': 0,
            'chunks_skipped': 0,
            'errors': []
        }

        # Get book filesystem mapping
        book_mapping = await self.get_book_filesystem_mapping()
        print(f"\n📚 Found {len(book_mapping)} books with filesystem_id in Neo4j")

        for book_dir in sorted(book_data_dir.iterdir()):
            if not book_dir.is_dir():
                continue

            chunks_file = book_dir / "kv_store_text_chunks.json"
            if not chunks_file.exists():
                continue

            filesystem_id = book_dir.name

            # Check if book exists in Neo4j
            if filesystem_id not in book_mapping:
                print(f"   ⚠️  {filesystem_id}: No matching BOOK node in Neo4j, skipping")
                stats['errors'].append(f"No BOOK node for {filesystem_id}")
                continue

            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)

                chunk_count = len(chunks)
                print(f"\n📖 {filesystem_id}: {chunk_count} chunks")

                if dry_run:
                    print(f"   [DRY RUN] Would index {chunk_count} chunks")
                    stats['chunks_indexed'] += chunk_count
                else:
                    # Batch index chunks
                    indexed = await self._index_chunks_batch(
                        filesystem_id,
                        list(chunks.keys())
                    )
                    stats['chunks_indexed'] += indexed
                    print(f"   ✅ Indexed {indexed} chunks")

                stats['books_processed'] += 1

            except Exception as e:
                print(f"   ❌ Error: {e}")
                stats['errors'].append(f"{filesystem_id}: {str(e)}")

        return stats

    async def _index_chunks_batch(
        self,
        filesystem_id: str,
        chunk_ids: List[str],
        batch_size: int = 500
    ) -> int:
        """Index chunks in batches."""
        total_indexed = 0

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i:i + batch_size]

            query = """
            UNWIND $chunks as chunk_id
            MERGE (c:Chunk {id: chunk_id})
            SET c.book_filesystem_id = $filesystem_id,
                c.indexed_at = datetime()
            REMOVE c.content
            WITH c
            MATCH (b:BOOK {filesystem_id: $filesystem_id})
            MERGE (b)-[:HAS_CHUNK]->(c)
            RETURN count(c) as indexed
            """

            async with self.client.driver.session() as session:
                result = await session.run(
                    query,
                    chunks=batch,
                    filesystem_id=filesystem_id
                )
                record = await result.single()
                total_indexed += record['indexed'] if record else 0

        return total_indexed

    async def link_entities_to_chunks(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Link Entity nodes to Chunk nodes via EXTRACTED_FROM relationship.

        Parses compound source_ids (split by <SEP>) and creates relationships.
        """
        stats = {
            'entities_processed': 0,
            'relationships_created': 0,
            'unmatched_chunks': 0
        }

        if dry_run:
            # Count what would be done
            query = """
            MATCH (e:Entity)
            WHERE e.source_id IS NOT NULL AND e.source_id CONTAINS 'chunk-'
            RETURN count(e) as entity_count
            """
            async with self.client.driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                stats['entities_processed'] = record['entity_count'] if record else 0
            print(f"\n[DRY RUN] Would process {stats['entities_processed']} entities")
            return stats

        print("\n🔗 Linking entities to chunks...")

        # Link entities to chunks
        query = """
        MATCH (e:Entity)
        WHERE e.source_id IS NOT NULL AND e.source_id CONTAINS 'chunk-'
        WITH e, split(e.source_id, '<SEP>') as chunk_ids
        UNWIND chunk_ids as chunk_id_raw
        WITH e, trim(chunk_id_raw) as chunk_id
        WHERE chunk_id <> '' AND chunk_id STARTS WITH 'chunk-'
        MATCH (c:Chunk {id: chunk_id})
        MERGE (e)-[r:EXTRACTED_FROM]->(c)
        RETURN count(DISTINCT e) as entities, count(r) as relationships
        """

        async with self.client.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            if record:
                stats['entities_processed'] = record['entities']
                stats['relationships_created'] = record['relationships']

        print(f"   ✅ Linked {stats['entities_processed']} entities via {stats['relationships_created']} relationships")

        # Count unmatched (entities with source_id but no EXTRACTED_FROM relationship)
        query = """
        MATCH (e:Entity)
        WHERE e.source_id IS NOT NULL AND e.source_id CONTAINS 'chunk-'
        AND NOT (e)-[:EXTRACTED_FROM]->(:Chunk)
        RETURN count(e) as unmatched
        """
        async with self.client.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            stats['unmatched_chunks'] = record['unmatched'] if record else 0

        if stats['unmatched_chunks'] > 0:
            print(f"   ⚠️  {stats['unmatched_chunks']} entities have unmatched chunk references")

        return stats

    async def get_reconciliation_status(self) -> Dict[str, Any]:
        """Get current reconciliation status."""
        status = {}

        # Count Chunk nodes
        query = "MATCH (c:Chunk) RETURN count(c) as count"
        async with self.client.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            status['chunk_nodes'] = record['count'] if record else 0

        # Count chunks with book_filesystem_id
        query = "MATCH (c:Chunk) WHERE c.book_filesystem_id IS NOT NULL RETURN count(c) as count"
        async with self.client.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            status['chunks_with_filesystem_id'] = record['count'] if record else 0

        # Count EXTRACTED_FROM relationships
        query = "MATCH ()-[r:EXTRACTED_FROM]->() RETURN count(r) as count"
        async with self.client.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            status['extracted_from_relationships'] = record['count'] if record else 0

        # Count unique source_ids in entities
        query = """
        MATCH (e:Entity)
        WHERE e.source_id IS NOT NULL AND e.source_id CONTAINS 'chunk-'
        RETURN count(DISTINCT e.source_id) as count
        """
        async with self.client.driver.session() as session:
            result = await session.run(query)
            record = await result.single()
            status['unique_source_ids'] = record['count'] if record else 0

        return status


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Reconcile chunks between filesystem and Neo4j',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full reconciliation
  python -m cli.reconcile_chunks

  # Dry run (show what would be done)
  python -m cli.reconcile_chunks --dry-run

  # Only index chunks (skip entity linking)
  python -m cli.reconcile_chunks --index-only

  # Show current status
  python -m cli.reconcile_chunks --status
        """
    )

    parser.add_argument('--dry-run', '-n', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--index-only', action='store_true',
                        help='Only index chunks, skip entity linking')
    parser.add_argument('--link-only', action='store_true',
                        help='Only link entities, skip chunk indexing')
    parser.add_argument('--status', '-s', action='store_true',
                        help='Show current reconciliation status')

    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    service = ChunkReconciliationService()

    try:
        # Verify connection
        if not await service.verify_connection():
            print("❌ Failed to connect to Neo4j")
            sys.exit(1)

        print("✅ Connected to Neo4j")

        # Status only
        if args.status:
            status = await service.get_reconciliation_status()
            print("\n📊 Current Reconciliation Status:")
            print(f"   Chunk nodes: {status['chunk_nodes']}")
            print(f"   Chunks with filesystem_id: {status['chunks_with_filesystem_id']}")
            print(f"   EXTRACTED_FROM relationships: {status['extracted_from_relationships']}")
            print(f"   Unique entity source_ids: {status['unique_source_ids']}")
            return

        # Full or partial reconciliation
        if not args.link_only:
            print("\n" + "="*60)
            print("PHASE 1: Index Filesystem Chunks")
            print("="*60)
            index_stats = await service.index_filesystem_chunks(dry_run=args.dry_run)
            print(f"\n📊 Indexing Summary:")
            print(f"   Books processed: {index_stats['books_processed']}")
            print(f"   Chunks indexed: {index_stats['chunks_indexed']}")
            if index_stats['errors']:
                print(f"   Errors: {len(index_stats['errors'])}")

        if not args.index_only:
            print("\n" + "="*60)
            print("PHASE 2: Link Entities to Chunks")
            print("="*60)
            link_stats = await service.link_entities_to_chunks(dry_run=args.dry_run)
            print(f"\n📊 Linking Summary:")
            print(f"   Entities processed: {link_stats['entities_processed']}")
            print(f"   Relationships created: {link_stats['relationships_created']}")
            if link_stats.get('unmatched_chunks', 0) > 0:
                print(f"   ⚠️  Unmatched: {link_stats['unmatched_chunks']}")

        print("\n" + "="*60)
        print("✅ Reconciliation complete!")
        print("="*60)

    finally:
        await service.close()


if __name__ == '__main__':
    asyncio.run(main())
