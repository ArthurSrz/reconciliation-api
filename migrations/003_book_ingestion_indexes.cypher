// Migration: 003_book_ingestion_indexes
// Feature: 001-interactive-graphrag-refinement (US2 - Book Ingestion)
// Date: 2025-11-23
//
// Creates indexes for BookIngestion tracking and book data management.
// These indexes optimize queries for:
// - Looking up ingestion jobs by status
// - Finding books by ingestion job
// - Admin audit trail queries

// Index for querying ingestion jobs by status
// Used by: GET /admin/ingest/jobs?status=processing
CREATE INDEX ingestion_status_idx IF NOT EXISTS FOR (i:BookIngestion) ON (i.status);

// Index for finding ingestion job by book_id
// Used by: Rollback operations, book lookup
CREATE INDEX ingestion_book_idx IF NOT EXISTS FOR (i:BookIngestion) ON (i.book_id);

// Index for admin audit queries
// Used by: Filtering jobs by admin who created them
CREATE INDEX ingestion_admin_idx IF NOT EXISTS FOR (i:BookIngestion) ON (i.admin_id);

// Index for book source_id (used by all content linked to a book)
// Used by: Rollback queries (MATCH (n {source_id: $book_id}) DETACH DELETE n)
CREATE INDEX entity_source_idx IF NOT EXISTS FOR (e:Entity) ON (e.source_id);

// Index for book node lookup by id
// Used by: All book-related queries
CREATE INDEX book_id_idx IF NOT EXISTS FOR (b:BOOK) ON (b.id);

// Index for book ingestion job id
// Used by: Status polling, job lookup
CREATE INDEX book_ingestion_id_idx IF NOT EXISTS FOR (i:BookIngestion) ON (i.id);

// Composite index for book ingestion status + timestamp
// Used by: Listing recent jobs by status
CREATE INDEX ingestion_status_time_idx IF NOT EXISTS FOR (i:BookIngestion) ON (i.status, i.started_at);
