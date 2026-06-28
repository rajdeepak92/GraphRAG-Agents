"""Neo4j schema management."""

from __future__ import annotations

from neo4j import AsyncDriver, AsyncManagedTransaction

CONSTRAINT_QUERIES = (
    """
    CREATE CONSTRAINT project_project_id IF NOT EXISTS
    FOR (n:Project)
    REQUIRE n.project_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_document_id IF NOT EXISTS
    FOR (n:Document)
    REQUIRE n.document_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT document_version_id IF NOT EXISTS
    FOR (n:DocumentVersion)
    REQUIRE n.document_version_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS
    FOR (n:Chunk)
    REQUIRE n.chunk_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT fact_fact_id IF NOT EXISTS
    FOR (n:Fact)
    REQUIRE n.fact_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT requirement_requirement_id IF NOT EXISTS
    FOR (n:Requirement)
    REQUIRE n.requirement_id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT ingestion_run_run_id IF NOT EXISTS
    FOR (n:IngestionRun)
    REQUIRE n.run_id IS UNIQUE
    """,
)


async def ensure_neo4j_schema(driver: AsyncDriver, *, database: str) -> None:
    """Create required Neo4j uniqueness constraints."""

    async with driver.session(database=database) as session:
        for query in CONSTRAINT_QUERIES:
            await session.execute_write(_run_schema_query, query)


async def _run_schema_query(
    tx: AsyncManagedTransaction,
    query: str,
) -> None:
    result = await tx.run(query)
    await result.consume()
