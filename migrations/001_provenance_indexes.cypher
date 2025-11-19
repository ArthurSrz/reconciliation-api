// Migration: 001 - Provenance Indexes
// Feature: Interactive GraphRAG Refinement
// Date: 2025-11-19
// Purpose: Create indexes for Query nodes to enable fast provenance retrieval

// Query node indexes for fast lookups
CREATE INDEX query_id_idx IF NOT EXISTS FOR (q:Query) ON (q.id);
CREATE INDEX query_timestamp_idx IF NOT EXISTS FOR (q:Query) ON (q.timestamp);
CREATE INDEX query_user_idx IF NOT EXISTS FOR (q:Query) ON (q.user_id);
CREATE INDEX query_version_idx IF NOT EXISTS FOR (q:Query) ON (q.version);
CREATE INDEX query_status_idx IF NOT EXISTS FOR (q:Query) ON (q.status);

// QueryResult node indexes
CREATE INDEX result_query_idx IF NOT EXISTS FOR (qr:QueryResult) ON (qr.query_id);
CREATE INDEX result_timestamp_idx IF NOT EXISTS FOR (qr:QueryResult) ON (qr.timestamp);

// GraphEdit node indexes for edit history and validation
CREATE INDEX edit_timestamp_idx IF NOT EXISTS FOR (ge:GraphEdit) ON (ge.timestamp);
CREATE INDEX edit_editor_idx IF NOT EXISTS FOR (ge:GraphEdit) ON (ge.editor_id);
CREATE INDEX edit_applied_idx IF NOT EXISTS FOR (ge:GraphEdit) ON (ge.applied);
CREATE INDEX edit_target_idx IF NOT EXISTS FOR (ge:GraphEdit) ON (ge.target_id);

// OntologicalPattern indexes for pattern discovery
CREATE INDEX pattern_frequency_idx IF NOT EXISTS FOR (op:OntologicalPattern) ON (op.frequency);
CREATE INDEX pattern_significance_idx IF NOT EXISTS FOR (op:OntologicalPattern) ON (op.significance_score);
CREATE INDEX pattern_cross_domain_idx IF NOT EXISTS FOR (op:OntologicalPattern) ON (op.cross_domain_count);

// PatternInstance indexes
CREATE INDEX instance_pattern_idx IF NOT EXISTS FOR (pi:PatternInstance) ON (pi.pattern_id);
CREATE INDEX instance_book_idx IF NOT EXISTS FOR (pi:PatternInstance) ON (pi.book_id);

// Composite indexes for common query patterns
CREATE INDEX query_user_timestamp_idx IF NOT EXISTS FOR (q:Query) ON (q.user_id, q.timestamp);
CREATE INDEX edit_editor_timestamp_idx IF NOT EXISTS FOR (ge:GraphEdit) ON (ge.editor_id, ge.timestamp);

// Primary key constraints for data integrity
CREATE CONSTRAINT query_id_unique IF NOT EXISTS FOR (q:Query) REQUIRE q.id IS UNIQUE;
CREATE CONSTRAINT query_result_id_unique IF NOT EXISTS FOR (qr:QueryResult) REQUIRE qr.id IS UNIQUE;
CREATE CONSTRAINT graph_edit_id_unique IF NOT EXISTS FOR (ge:GraphEdit) REQUIRE ge.id IS UNIQUE;
CREATE CONSTRAINT pattern_id_unique IF NOT EXISTS FOR (op:OntologicalPattern) REQUIRE op.id IS UNIQUE;
CREATE CONSTRAINT pattern_instance_id_unique IF NOT EXISTS FOR (pi:PatternInstance) REQUIRE pi.id IS UNIQUE;
