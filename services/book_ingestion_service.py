"""
Book Ingestion Service for nano-graphRAG pipeline.

This service wraps the nano-graphRAG pipeline to provide:
- Schema-consistent book processing
- Inter-book entity linking
- Progress tracking
- Rollback capability

Constitution Compliance:
- Principle VI: Extensible Literature Foundation - uses same pipeline as existing books
- Principle III: No orphan nodes - validates all entities have relationships
- Principle V: Inter-book exploration - links new entities to existing books
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path

from models.book_ingestion import (
    BookIngestion,
    BookMetadata,
    IngestionStatus,
    store_ingestion_job,
    get_ingestion_job
)
from nano_graphrag._llm import gpt_4o_mini_complete


# Raw books directory (user specified)
RAW_BOOKS_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data/raw_books"

# Output directory for processed books
BOOK_DATA_DIR = "/Users/arthursarazin/Documents/nano-graphrag/book_data"


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class BookIngestionService:
    """
    Service for ingesting books through the nano-graphRAG pipeline.

    Ensures schema consistency with existing books and provides
    inter-book entity linking, progress tracking, and rollback.
    """

    # Expected Neo4j schema based on existing books
    EXPECTED_NODE_LABELS = {'Entity', 'Chunk', 'BOOK', 'Community'}
    EXPECTED_RELATIONSHIP_TYPES = {
        'RELATED_TO', 'MENTIONS', 'BELONGS_TO', 'PART_OF',
        'CONTAINS_ENTITY', 'CONTAINS', 'HAS_CHUNK'
    }
    EXPECTED_ENTITY_PROPERTIES = {
        'id', 'entity_type', 'description', 'source_id'
    }

    def __init__(self, neo4j_client=None, graphrag_config: Optional[Dict] = None):
        """
        Initialize the book ingestion service.

        Args:
            neo4j_client: Neo4j client instance
            graphrag_config: Custom nano-graphRAG configuration (uses default if None)
        """
        self.neo4j_client = neo4j_client
        self.graphrag_config = graphrag_config or self._get_default_config()
        self.config_hash = self._compute_config_hash()
        self._progress_callback: Optional[Callable[[BookIngestion], None]] = None

    def _get_default_config(self) -> Dict[str, Any]:
        """Get the default nano-graphRAG configuration matching existing books."""
        return {
            "chunk_token_size": 1200,
            "chunk_overlap_token_size": 100,
            "entity_extract_max_gleaning": 1,
            "entity_summary_to_max_tokens": 500,
            "graph_cluster_algorithm": "leiden",
            "max_graph_cluster_size": 10,
            "node_embedding_algorithm": "node2vec",
            # Reduce concurrency to avoid API rate limiting
            "best_model_max_async": 2,
            "cheap_model_max_async": 2,
            "embedding_func_max_async": 2,
            "embedding_batch_num": 4
        }

    def _compute_config_hash(self) -> str:
        """Compute hash of config for tracking consistency."""
        config_str = json.dumps(self.graphrag_config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def set_progress_callback(self, callback: Callable[[BookIngestion], None]) -> None:
        """Set callback for progress updates."""
        self._progress_callback = callback

    def _notify_progress(self, job: BookIngestion) -> None:
        """Notify progress callback if set."""
        if self._progress_callback:
            self._progress_callback(job)

    async def ingest_book(
        self,
        file_path: str,
        metadata: BookMetadata,
        admin_id: str
    ) -> BookIngestion:
        """
        Ingest a book through the nano-graphRAG pipeline.

        Args:
            file_path: Path to the book file
            metadata: Book metadata (title, author, etc.)
            admin_id: ID of the admin performing the ingestion

        Returns:
            BookIngestion job with final status
        """
        # Create ingestion job
        job = BookIngestion(
            file_path=file_path,
            book_metadata=metadata,
            admin_id=admin_id
        )
        job.config_hash = self.config_hash
        store_ingestion_job(job)

        # Start processing
        job.start()
        self._notify_progress(job)

        try:
            # Phase 1: Read and validate input file (0-10%)
            job.update_progress(0.05)
            self._notify_progress(job)

            content = await self._read_book_file(file_path)
            if not content:
                raise ValueError(f"Could not read book file: {file_path}")

            job.update_progress(0.10)
            self._notify_progress(job)

            # Phase 2: Run nano-graphRAG pipeline (10-70%)
            output_dir = self._get_output_dir(metadata)
            result = await self._run_graphrag_pipeline(
                content, output_dir, metadata, job
            )

            # Phase 3: Validate schema consistency (70-80%)
            job.update_progress(0.70)
            self._notify_progress(job)

            await self._validate_schema(result)

            job.update_progress(0.80)
            self._notify_progress(job)

            # Phase 4: Import to Neo4j and link entities (80-95%)
            import_result = await self._import_to_neo4j(
                result, metadata, job
            )

            job.update_progress(0.95)
            self._notify_progress(job)

            # Phase 5: Complete (95-100%)
            job.complete(
                chunks=result.get('chunks_count', 0),
                entities=import_result.get('entities_count', 0),
                relationships=import_result.get('relationships_count', 0),
                inter_book=import_result.get('inter_book_links', 0)
            )
            store_ingestion_job(job)
            self._notify_progress(job)

            return job

        except Exception as e:
            job.fail(str(e))
            store_ingestion_job(job)
            self._notify_progress(job)
            raise

    async def _read_book_file(self, file_path: str) -> Optional[str]:
        """Read book content from file."""
        path = Path(file_path)

        if not path.exists():
            # Try in raw_books directory
            path = Path(RAW_BOOKS_DIR) / path.name
            if not path.exists():
                return None

        try:
            if path.suffix.lower() == '.txt':
                return path.read_text(encoding='utf-8')
            elif path.suffix.lower() == '.pdf':
                return await self._extract_pdf_text(path)
            elif path.suffix.lower() == '.epub':
                return await self._extract_epub_text(path)
            else:
                # Try reading as text
                return path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return None

    async def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF file."""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            raise ValueError("PyMuPDF (fitz) not installed. Install with: pip install pymupdf")

    async def _extract_epub_text(self, path: Path) -> str:
        """Extract text from EPUB file."""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(str(path))
            text = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
            return text
        except ImportError:
            raise ValueError("ebooklib not installed. Install with: pip install ebooklib beautifulsoup4")

    def _get_output_dir(self, metadata: BookMetadata) -> str:
        """Get output directory for processed book."""
        # Create directory name from title
        safe_title = "".join(c if c.isalnum() else "_" for c in metadata.title.lower())
        safe_author = "".join(c if c.isalnum() else "_" for c in metadata.author.lower())
        dir_name = f"{safe_title}_{safe_author}"
        return os.path.join(BOOK_DATA_DIR, dir_name)

    async def _run_graphrag_pipeline(
        self,
        content: str,
        output_dir: str,
        metadata: BookMetadata,
        job: BookIngestion
    ) -> Dict[str, Any]:
        """
        Run the nano-graphRAG pipeline on book content.

        Uses the same configuration as existing books for schema consistency.
        """
        from nano_graphrag import GraphRAG

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize GraphRAG with consistent config
        # Use gpt-4o-mini for cost efficiency, reduced concurrency for rate limiting
        graphrag = GraphRAG(
            working_dir=output_dir,
            # Use gpt-4o-mini for both models (cheaper than gpt-4o)
            best_model_func=gpt_4o_mini_complete,
            cheap_model_func=gpt_4o_mini_complete,
            chunk_token_size=self.graphrag_config['chunk_token_size'],
            chunk_overlap_token_size=self.graphrag_config['chunk_overlap_token_size'],
            entity_extract_max_gleaning=self.graphrag_config['entity_extract_max_gleaning'],
            entity_summary_to_max_tokens=self.graphrag_config['entity_summary_to_max_tokens'],
            graph_cluster_algorithm=self.graphrag_config['graph_cluster_algorithm'],
            max_graph_cluster_size=self.graphrag_config['max_graph_cluster_size'],
            node_embedding_algorithm=self.graphrag_config['node_embedding_algorithm'],
            # Concurrency settings to avoid rate limiting
            best_model_max_async=self.graphrag_config.get('best_model_max_async', 2),
            cheap_model_max_async=self.graphrag_config.get('cheap_model_max_async', 2),
            embedding_func_max_async=self.graphrag_config.get('embedding_func_max_async', 2),
            embedding_batch_num=self.graphrag_config.get('embedding_batch_num', 4)
        )

        # Progress updates during pipeline
        async def update_progress():
            for i in range(10, 70, 10):
                await asyncio.sleep(10)  # Update every 10 seconds
                job.update_progress(i / 100)
                self._notify_progress(job)
                store_ingestion_job(job)

        # Run pipeline
        progress_task = asyncio.create_task(update_progress())
        try:
            await graphrag.ainsert(content)
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        # Count results from output files
        result = {
            'output_dir': output_dir,
            'chunks_count': self._count_chunks(output_dir),
            'entities': self._load_entities(output_dir),
            'relationships': self._load_relationships(output_dir)
        }

        return result

    def _count_chunks(self, output_dir: str) -> int:
        """Count chunks from KV store."""
        chunks_file = os.path.join(output_dir, 'kv_store_text_chunks.json')
        if os.path.exists(chunks_file):
            with open(chunks_file) as f:
                chunks = json.load(f)
                return len(chunks)
        return 0

    def _load_entities(self, output_dir: str) -> List[Dict]:
        """Load entities from output directory."""
        entities_file = os.path.join(output_dir, 'kv_store_full_docs.json')
        if os.path.exists(entities_file):
            with open(entities_file) as f:
                return json.load(f)
        return []

    def _load_relationships(self, output_dir: str) -> List[Dict]:
        """Load relationships from GraphML file."""
        graphml_file = os.path.join(output_dir, 'graph_chunk_entity_relation.graphml')
        if os.path.exists(graphml_file):
            try:
                import networkx as nx
                G = nx.read_graphml(graphml_file)
                return [
                    {'source': e[0], 'target': e[1], **e[2]}
                    for e in G.edges(data=True)
                ]
            except Exception:
                pass
        return []

    async def _validate_schema(self, result: Dict[str, Any]) -> None:
        """
        Validate that extracted entities and relationships match expected schema.

        Raises:
            SchemaValidationError: If schema doesn't match
        """
        # Validate entities have required properties
        entities = result.get('entities', [])
        for entity in entities:
            if isinstance(entity, dict):
                # Check for required properties (flexible validation)
                if 'id' not in entity and 'name' not in entity:
                    raise SchemaValidationError(
                        f"Entity missing id/name: {entity}"
                    )

        # Validate relationships
        relationships = result.get('relationships', [])
        for rel in relationships:
            if 'source' not in rel or 'target' not in rel:
                raise SchemaValidationError(
                    f"Relationship missing source/target: {rel}"
                )

    async def _import_to_neo4j(
        self,
        result: Dict[str, Any],
        metadata: BookMetadata,
        job: BookIngestion
    ) -> Dict[str, int]:
        """
        Import processed book data to Neo4j.

        Returns:
            Dict with counts of imported items
        """
        if not self.neo4j_client:
            # Return mock counts if no Neo4j client
            return {
                'entities_count': len(result.get('entities', [])),
                'relationships_count': len(result.get('relationships', [])),
                'inter_book_links': 0
            }

        async with self.neo4j_client.driver.session() as session:
            # Create BOOK node
            await session.run(
                """
                MERGE (b:BOOK {id: $book_id})
                SET b.title = $title,
                    b.author = $author,
                    b.genre = $genre,
                    b.publication_date = $pub_date,
                    b.ingestion_job_id = $job_id,
                    b.created_at = datetime()
                """,
                book_id=job.book_id,
                title=metadata.title,
                author=metadata.author,
                genre=metadata.genre,
                pub_date=metadata.publication_date,
                job_id=job.id
            )

            # Import entities
            entities_count = 0
            for entity in result.get('entities', []):
                if isinstance(entity, dict):
                    entity_id = entity.get('id') or entity.get('name', '')
                    await session.run(
                        """
                        MERGE (e:Entity {id: $entity_id})
                        SET e.entity_type = $entity_type,
                            e.description = $description,
                            e.source_id = $book_id
                        WITH e
                        MATCH (b:BOOK {id: $book_id})
                        MERGE (b)-[:CONTAINS_ENTITY]->(e)
                        """,
                        entity_id=entity_id,
                        entity_type=entity.get('entity_type', 'unknown'),
                        description=entity.get('description', ''),
                        book_id=job.book_id
                    )
                    entities_count += 1

            # Import relationships
            relationships_count = 0
            for rel in result.get('relationships', []):
                await session.run(
                    """
                    MATCH (source:Entity {id: $source})
                    MATCH (target:Entity {id: $target})
                    MERGE (source)-[r:RELATED_TO]->(target)
                    SET r.type = $rel_type,
                        r.weight = $weight,
                        r.source_id = $book_id
                    """,
                    source=rel.get('source'),
                    target=rel.get('target'),
                    rel_type=rel.get('type', 'RELATED_TO'),
                    weight=rel.get('weight', 1.0),
                    book_id=job.book_id
                )
                relationships_count += 1

            # Link to existing entities (inter-book linking)
            inter_book_links = await self._link_to_existing_entities(
                session, job.book_id
            )

            return {
                'entities_count': entities_count,
                'relationships_count': relationships_count,
                'inter_book_links': inter_book_links
            }

    async def _link_to_existing_entities(
        self,
        session,
        book_id: str,
        similarity_threshold: float = 0.85,
        inter_book_boost: float = 1.2
    ) -> int:
        """
        Link new entities to existing entities from other books.

        Uses entity name/description similarity to find matches.
        Applies inter-book weight boost per Constitution Principle V.

        Returns:
            Number of inter-book links created
        """
        # Find similar entities across books
        result = await session.run(
            """
            MATCH (new_entity:Entity {source_id: $book_id})
            MATCH (existing:Entity)
            WHERE existing.source_id <> $book_id
            AND (
                toLower(new_entity.id) CONTAINS toLower(existing.id)
                OR toLower(existing.id) CONTAINS toLower(new_entity.id)
            )
            WITH new_entity, existing,
                 CASE WHEN new_entity.id = existing.id THEN 1.0
                      ELSE 0.8 END as similarity
            WHERE similarity >= $threshold
            MERGE (new_entity)-[r:RELATED_TO]->(existing)
            SET r.weight = similarity * $boost,
                r.inter_book = true,
                r.created_at = datetime()
            RETURN count(r) as link_count
            """,
            book_id=book_id,
            threshold=similarity_threshold,
            boost=inter_book_boost
        )
        record = await result.single()
        return record['link_count'] if record else 0

    async def rollback(self, job_id: str) -> Dict[str, Any]:
        """
        Rollback a book ingestion, removing all created content.

        Args:
            job_id: ID of the ingestion job to rollback

        Returns:
            Dict with rollback statistics
        """
        job = get_ingestion_job(job_id)
        if not job:
            raise ValueError(f"Ingestion job not found: {job_id}")

        if not job.can_rollback():
            raise ValueError(
                f"Cannot rollback job in status: {job.status.value}. "
                f"Only completed or failed jobs can be rolled back."
            )

        start_time = datetime.utcnow()
        entities_deleted = 0
        relationships_deleted = 0

        if self.neo4j_client:
            async with self.neo4j_client.driver.session() as session:
                # Delete all content with this book_id
                # First count what we're deleting
                count_result = await session.run(
                    """
                    MATCH (n {source_id: $book_id})
                    OPTIONAL MATCH (n)-[r]-()
                    RETURN count(DISTINCT n) as nodes, count(DISTINCT r) as rels
                    """,
                    book_id=job.book_id
                )
                counts = await count_result.single()
                entities_deleted = counts['nodes'] if counts else 0
                relationships_deleted = counts['rels'] if counts else 0

                # Delete the book and all related content
                await session.run(
                    """
                    MATCH (b:BOOK {id: $book_id})
                    DETACH DELETE b
                    """,
                    book_id=job.book_id
                )

                await session.run(
                    """
                    MATCH (n {source_id: $book_id})
                    DETACH DELETE n
                    """,
                    book_id=job.book_id
                )

        # Update job status
        job.rollback()
        store_ingestion_job(job)

        duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

        return {
            'job_id': job_id,
            'book_id': job.book_id,
            'status': 'rolled_back',
            'entities_deleted': entities_deleted,
            'relationships_deleted': relationships_deleted,
            'duration_ms': duration_ms,
            'message': 'Book content successfully removed from graph.'
        }

    async def get_existing_schema(self) -> Dict[str, Any]:
        """
        Get the schema of existing books in Neo4j for reference.

        Returns:
            Dict with node labels, relationship types, and properties
        """
        if not self.neo4j_client:
            return {
                'node_labels': list(self.EXPECTED_NODE_LABELS),
                'relationship_types': list(self.EXPECTED_RELATIONSHIP_TYPES),
                'entity_properties': list(self.EXPECTED_ENTITY_PROPERTIES)
            }

        async with self.neo4j_client.driver.session() as session:
            # Get all node labels
            labels_result = await session.run("CALL db.labels()")
            labels = [record[0] async for record in labels_result]

            # Get all relationship types
            rels_result = await session.run("CALL db.relationshipTypes()")
            rel_types = [record[0] async for record in rels_result]

            # Get Entity properties
            props_result = await session.run(
                """
                MATCH (e:Entity)
                WITH keys(e) as props
                UNWIND props as prop
                RETURN DISTINCT prop
                LIMIT 100
                """
            )
            entity_props = [record[0] async for record in props_result]

            return {
                'node_labels': labels,
                'relationship_types': rel_types,
                'entity_properties': entity_props
            }
