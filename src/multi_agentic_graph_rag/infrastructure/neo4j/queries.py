"""Cypher queries for deterministic graph projections."""

PROJECT_DOCUMENT_HIERARCHY = """
MERGE (p:Project {project_id: $project.project_id})
ON CREATE SET
    p.project_key = $project.project_key,
    p.name = $project.name,
    p.created_at = datetime()
ON MATCH SET
    p.project_key = $project.project_key,
    p.name = $project.name,
    p.updated_at = datetime()

MERGE (d:Document {document_id: $document.document_id})
ON CREATE SET
    d.logical_document_name = $document.logical_document_name,
    d.created_at = datetime()
ON MATCH SET
    d.logical_document_name = $document.logical_document_name,
    d.updated_at = datetime()

MERGE (p)-[:OWNS]->(d)

MERGE (v:DocumentVersion {document_version_id: $document_version.document_version_id})
ON CREATE SET
    v.supplied_version = $document_version.supplied_version,
    v.normalized_version = $document_version.normalized_version,
    v.source_checksum = $document_version.source_checksum,
    v.created_at = datetime()
ON MATCH SET
    v.supplied_version = $document_version.supplied_version,
    v.normalized_version = $document_version.normalized_version,
    v.source_checksum = $document_version.source_checksum,
    v.updated_at = datetime()

MERGE (d)-[:HAS_VERSION]->(v)

MERGE (r:IngestionRun {run_id: $ingestion_run.run_id})
ON CREATE SET
    r.status = $ingestion_run.status,
    r.created_at = datetime()
ON MATCH SET
    r.status = $ingestion_run.status,
    r.updated_at = datetime()
"""

PROJECT_CHUNK = """
MATCH (v:DocumentVersion {document_version_id: $document_version_id})
MERGE (c:Chunk {chunk_id: $chunk.chunk_id})
ON CREATE SET
    c.chunk_ordinal = $chunk.chunk_ordinal,
    c.content_hash = $chunk.content_hash,
    c.page_start = $chunk.page_start,
    c.page_end = $chunk.page_end,
    c.created_at = datetime()
ON MATCH SET
    c.chunk_ordinal = $chunk.chunk_ordinal,
    c.content_hash = $chunk.content_hash,
    c.page_start = $chunk.page_start,
    c.page_end = $chunk.page_end,
    c.updated_at = datetime()

MERGE (v)-[:CONTAINS]->(c)
"""

PROJECT_FACT_REQUIREMENT_TRACE = """
MERGE (run:IngestionRun {run_id: $ingestion_run.run_id})
ON CREATE SET run.status = $ingestion_run.status, run.created_at = datetime()
ON MATCH SET run.status = $ingestion_run.status, run.updated_at = datetime()

MERGE (f:Fact {fact_id: $fact.fact_id})
ON CREATE SET
    f.statement = $fact.statement,
    f.fact_type = $fact.fact_type,
    f.validation_status = $fact.validation_status,
    f.created_at = datetime()
ON MATCH SET
    f.statement = $fact.statement,
    f.fact_type = $fact.fact_type,
    f.validation_status = $fact.validation_status,
    f.updated_at = datetime()

MERGE (run)-[:PRODUCED]->(f)
"""

PROJECT_FACT_EVIDENCE = """
MATCH (f:Fact {fact_id: $fact_id})
MATCH (c:Chunk {chunk_id: $chunk_id})
MERGE (f)-[rel:EVIDENCED_BY]->(c)
ON CREATE SET
    rel.exact_quote = $exact_quote,
    rel.character_start = $character_start,
    rel.character_end = $character_end,
    rel.created_at = datetime()
ON MATCH SET
    rel.exact_quote = $exact_quote,
    rel.character_start = $character_start,
    rel.character_end = $character_end,
    rel.updated_at = datetime()
"""

PROJECT_REQUIREMENT = """
MERGE (run:IngestionRun {run_id: $ingestion_run.run_id})
ON CREATE SET run.status = $ingestion_run.status, run.created_at = datetime()
ON MATCH SET run.status = $ingestion_run.status, run.updated_at = datetime()

MERGE (req:Requirement {requirement_id: $requirement.requirement_id})
ON CREATE SET
    req.statement = $requirement.statement,
    req.requirement_type = $requirement.requirement_type,
    req.derivation_type = $requirement.derivation_type,
    req.validation_status = $requirement.validation_status,
    req.created_at = datetime()
ON MATCH SET
    req.statement = $requirement.statement,
    req.requirement_type = $requirement.requirement_type,
    req.derivation_type = $requirement.derivation_type,
    req.validation_status = $requirement.validation_status,
    req.updated_at = datetime()

MERGE (run)-[:PRODUCED]->(req)
"""

PROJECT_REQUIREMENT_FACT_LINK = """
MATCH (req:Requirement {requirement_id: $requirement_id})
MATCH (f:Fact {fact_id: $fact_id})
MERGE (req)-[:SUPPORTED_BY]->(f)
"""

PROJECT_DOCUMENT_VERSION_LINEAGE = """
MATCH (current:DocumentVersion {document_version_id: $document_version_id})
MATCH (previous:DocumentVersion {document_version_id: $supersedes_document_version_id})
MERGE (current)-[:SUPERSEDES]->(previous)
"""

PROJECT_REQUIREMENT_CONFLICT = """
MATCH (source:Requirement {requirement_id: $source_requirement_id})
MATCH (target:Requirement {requirement_id: $target_requirement_id})
MERGE (source)-[rel:CONFLICTS_WITH]->(target)
ON CREATE SET
    rel.created_at = datetime()
ON MATCH SET
    rel.updated_at = datetime()
SET
    rel.relation_source = $relation_source,
    rel.run_id = $run_id,
    rel.reason = $reason
"""

PROJECT_REQUIREMENT_DUPLICATE = """
MATCH (source:Requirement {requirement_id: $source_requirement_id})
MATCH (target:Requirement {requirement_id: $target_requirement_id})
MERGE (source)-[rel:DUPLICATES]->(target)
ON CREATE SET
    rel.created_at = datetime()
ON MATCH SET
    rel.updated_at = datetime()
SET
    rel.relation_source = $relation_source,
    rel.run_id = $run_id,
    rel.reason = $reason
"""

PROJECT_REQUIREMENT_SUPERSEDES = """
MATCH (source:Requirement {requirement_id: $source_requirement_id})
MATCH (target:Requirement {requirement_id: $target_requirement_id})
MERGE (source)-[rel:SUPERSEDES]->(target)
ON CREATE SET
    rel.created_at = datetime()
ON MATCH SET
    rel.updated_at = datetime()
SET
    rel.relation_source = $relation_source,
    rel.run_id = $run_id,
    rel.reason = $reason
"""

PROJECT_FACT_CONFLICT = """
MATCH (source:Fact {fact_id: $source_fact_id})
MATCH (target:Fact {fact_id: $target_fact_id})
MERGE (source)-[rel:CONFLICTS_WITH]->(target)
ON CREATE SET
    rel.created_at = datetime()
ON MATCH SET
    rel.updated_at = datetime()
SET
    rel.relation_source = $relation_source,
    rel.run_id = $run_id,
    rel.reason = $reason
"""
