from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver

from multi_agentic_graph_rag.config.settings import load_settings
from multi_agentic_graph_rag.infrastructure.neo4j.client import neo4j_driver_scope
from multi_agentic_graph_rag.infrastructure.neo4j.projections import Neo4jGraphStore


def _secret_value(value: Any) -> str:
    """Return plain string from pydantic SecretStr or normal string."""

    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())

    return str(value)


async def _count_node(
    driver: AsyncDriver,
    *,
    database: str,
    label: str,
    key: str,
    value: str,
) -> int:
    query = f"""
    MATCH (n:{label} {{{key}: $value}})
    RETURN count(n) AS node_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(query, value=value)
        record = await result.single(strict=True)
        return int(record["node_count"])


async def _count_requirement_trace(
    driver: AsyncDriver,
    *,
    database: str,
    requirement_id: str,
    fact_id: str,
    chunk_id: str,
) -> int:
    query = """
    MATCH (:Requirement {requirement_id: $requirement_id})
        -[:SUPPORTED_BY]->
        (:Fact {fact_id: $fact_id})
        -[:EVIDENCED_BY]->
        (:Chunk {chunk_id: $chunk_id})
    RETURN count(*) AS trace_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(
            query,
            requirement_id=requirement_id,
            fact_id=fact_id,
            chunk_id=chunk_id,
        )
        record = await result.single(strict=True)
        return int(record["trace_count"])


async def _count_document_trace(
    driver: AsyncDriver,
    *,
    database: str,
    document_id: str,
    document_version_id: str,
    chunk_id: str,
) -> int:
    query = """
    MATCH (:Document {document_id: $document_id})
        -[:HAS_VERSION]->
        (:DocumentVersion {document_version_id: $document_version_id})
        -[:CONTAINS]->
        (:Chunk {chunk_id: $chunk_id})
    RETURN count(*) AS trace_count
    """

    async with driver.session(database=database) as session:
        result = await session.run(
            query,
            document_id=document_id,
            document_version_id=document_version_id,
            chunk_id=chunk_id,
        )
        record = await result.single(strict=True)
        return int(record["trace_count"])


async def main() -> None:
    settings = load_settings()

    neo4j_settings = settings.neo4j

    uri = str(neo4j_settings.uri)
    username = str(neo4j_settings.username)
    password = _secret_value(neo4j_settings.password)
    database = str(neo4j_settings.database)

    suffix = uuid4().hex.upper()[:10]

    project = {
        "project_id": f"PROJECT-SMOKE-{suffix}",
        "project_key": f"PHASE5_SMOKE_{suffix}",
        "name": "Phase 5 Smoke Project",
    }

    document = {
        "document_id": f"DOC-SMOKE-{suffix}",
        "logical_document_name": "phase5-smoke-requirements.md",
    }

    document_version = {
        "document_version_id": f"DOCVER-SMOKE-{suffix}",
        "supplied_version": "1.0",
        "normalized_version": "1.0",
        "source_checksum": f"sha256-smoke-{suffix.lower()}",
    }

    chunk = {
        "chunk_id": f"CHUNK-SMOKE-{suffix}",
        "chunk_ordinal": 1,
        "content_hash": f"chunk-hash-{suffix.lower()}",
        "page_start": 1,
        "page_end": 1,
    }

    ingestion_run = {
        "run_id": f"RUN-SMOKE-{suffix}",
        "status": "graph_projected",
    }

    fact = {
        "fact_id": f"FACT-SMOKE-{suffix}",
        "statement": "Authenticated sessions expire after 30 minutes of inactivity.",
        "fact_type": "security_constraint",
        "validation_status": "validated",
    }

    requirement = {
        "requirement_id": f"REQ-SMOKE-{suffix}",
        "statement": (
            "The system shall terminate an authenticated session after 30 minutes of inactivity."
        ),
        "requirement_type": "security",
        "derivation_type": "normalized",
        "validation_status": "validated",
    }

    fact_evidence = {
        "fact_id": fact["fact_id"],
        "chunk_id": chunk["chunk_id"],
        "exact_quote": "sessions expire after 30 minutes of inactivity",
        "character_start": 25,
        "character_end": 77,
    }

    requirement_fact_link = {
        "requirement_id": requirement["requirement_id"],
        "fact_id": fact["fact_id"],
    }

    async with neo4j_driver_scope(
        uri=uri,
        username=username,
        password=password,
        database=database,
    ) as driver:
        await driver.verify_connectivity()

        graph_store = Neo4jGraphStore(driver=driver, database=database)

        await graph_store.verify_schema()

        # First projection.
        await graph_store.project_document_hierarchy(
            project=project,
            document=document,
            document_version=document_version,
            chunks=[chunk],
            ingestion_run=ingestion_run,
        )

        await graph_store.project_requirement_trace(
            facts=[fact],
            requirements=[requirement],
            fact_evidence=[fact_evidence],
            requirement_fact_links=[requirement_fact_link],
            ingestion_run=ingestion_run,
        )

        # Second projection. This must not create duplicates.
        await graph_store.project_document_hierarchy(
            project=project,
            document=document,
            document_version=document_version,
            chunks=[chunk],
            ingestion_run=ingestion_run,
        )

        await graph_store.project_requirement_trace(
            facts=[fact],
            requirements=[requirement],
            fact_evidence=[fact_evidence],
            requirement_fact_links=[requirement_fact_link],
            ingestion_run=ingestion_run,
        )

        expected_nodes = (
            ("Project", "project_id", project["project_id"]),
            ("Document", "document_id", document["document_id"]),
            (
                "DocumentVersion",
                "document_version_id",
                document_version["document_version_id"],
            ),
            ("Chunk", "chunk_id", chunk["chunk_id"]),
            ("Fact", "fact_id", fact["fact_id"]),
            ("Requirement", "requirement_id", requirement["requirement_id"]),
            ("IngestionRun", "run_id", ingestion_run["run_id"]),
        )

        duplicate_failures: list[str] = []

        for label, key, value in expected_nodes:
            count = await _count_node(
                driver,
                database=database,
                label=label,
                key=key,
                value=str(value),
            )

            if count != 1:
                duplicate_failures.append(f"{label}.{key}={value} count={count}")

        if duplicate_failures:
            failure_text = "\n".join(f"  - {item}" for item in duplicate_failures)
            raise RuntimeError(
                f"FAIL Neo4j projection smoke test. Duplicate or missing nodes:\n{failure_text}"
            )

        requirement_trace_count = await _count_requirement_trace(
            driver,
            database=database,
            requirement_id=str(requirement["requirement_id"]),
            fact_id=str(fact["fact_id"]),
            chunk_id=str(chunk["chunk_id"]),
        )

        if requirement_trace_count != 1:
            raise RuntimeError(
                "FAIL Neo4j projection smoke test. "
                "Requirement -> Fact -> Chunk traceability is missing."
            )

        document_trace_count = await _count_document_trace(
            driver,
            database=database,
            document_id=str(document["document_id"]),
            document_version_id=str(document_version["document_version_id"]),
            chunk_id=str(chunk["chunk_id"]),
        )

        if document_trace_count != 1:
            raise RuntimeError(
                "FAIL Neo4j projection smoke test. "
                "Document -> DocumentVersion -> Chunk traceability is missing."
            )

    print("PASS Neo4j projection smoke test.")
    print("Verified idempotent nodes:")
    for label, key, value in expected_nodes:
        print(f"  - {label}.{key}={value}")

    print("Verified traceability:")
    print("  - Requirement -> Fact -> Chunk")
    print("  - Document -> DocumentVersion -> Chunk")


if __name__ == "__main__":
    asyncio.run(main())
