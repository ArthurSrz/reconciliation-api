#!/usr/bin/env python3
"""
Batch sync all books to Neo4j with fixed chunk linkage
"""

import asyncio
from cli.sync_to_neo4j import sync_book_to_neo4j
from pathlib import Path

BOOKS = [
    {"dir": "a_rebours_huysmans", "title": "À Rebours", "author": "Joris-Karl Huysmans", "genre": "Fiction"},
    {"dir": "chien_blanc_gary", "title": "Chien Blanc", "author": "Romain Gary", "genre": "Fiction"},
    {"dir": "la_maison_vide_laurent_mauvignier", "title": "La Maison Vide", "author": "Laurent Mauvignier", "genre": "Fiction"},
    {"dir": "peau_bison_frison", "title": "Peau de Bison", "author": "Frison-Roche", "genre": "Fiction"},
    {"dir": "policeman_decoin", "title": "Policeman", "author": "Didier Decoin", "genre": "Fiction"},
    {"dir": "racines_ciel_gary", "title": "Les Racines du Ciel", "author": "Romain Gary", "genre": "Fiction"},
    {"dir": "tilleul_soir_anglade", "title": "Le Tilleul du Soir", "author": "Jean Anglade", "genre": "Fiction"},
    {"dir": "vallee_sans_hommes_frison", "title": "La Vallée Sans Hommes", "author": "Frison-Roche", "genre": "Fiction"},
]

BOOK_DATA_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data"


async def main():
    """Sync all books sequentially"""
    print("=" * 70)
    print("🔄 BATCH SYNC: Syncing all books with fixed chunk linkage")
    print("=" * 70)

    total_stats = {
        "books_synced": 0,
        "total_entities": 0,
        "total_relationships": 0,
        "total_chunks": 0,
        "total_entity_chunk_links": 0,
        "errors": []
    }

    for book in BOOKS:
        book_dir = Path(BOOK_DATA_DIR) / book["dir"]

        if not book_dir.exists():
            print(f"⚠️  Skipping {book['title']}: directory not found")
            continue

        print(f"\n{'='*70}")
        print(f"📖 Syncing: {book['title']} by {book['author']}")
        print(f"{'='*70}")

        try:
            stats = await sync_book_to_neo4j(
                book_dir=str(book_dir),
                title=book["title"],
                author=book["author"],
                genre=book.get("genre")
            )

            total_stats["books_synced"] += 1
            total_stats["total_entities"] += stats["nodes_processed"]
            total_stats["total_relationships"] += stats["edges_processed"]
            total_stats["total_chunks"] += stats["chunks_synced"]
            total_stats["total_entity_chunk_links"] += stats.get("entity_chunk_links", 0)

            print(f"✅ {book['title']}: {stats['nodes_processed']} entities, "
                  f"{stats.get('entity_chunk_links', 0)} chunk links created")

        except Exception as e:
            error_msg = f"{book['title']}: {str(e)}"
            total_stats["errors"].append(error_msg)
            print(f"❌ Error syncing {book['title']}: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("📊 BATCH SYNC SUMMARY")
    print("=" * 70)
    print(f"Books synced: {total_stats['books_synced']}/{len(BOOKS)}")
    print(f"Total entities: {total_stats['total_entities']}")
    print(f"Total relationships: {total_stats['total_relationships']}")
    print(f"Total chunks: {total_stats['total_chunks']}")
    print(f"Total Entity→Chunk links: {total_stats['total_entity_chunk_links']}")

    if total_stats["errors"]:
        print(f"\n⚠️  Errors ({len(total_stats['errors'])}):")
        for err in total_stats["errors"]:
            print(f"   - {err}")

    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
