"""Neo4j projection adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from neo4j import AsyncDriver, AsyncManagedTransaction

from multi_agentic_graph_rag.application.ports.graph_store import GraphStorePort
from multi_agentic_graph_rag.infrastructure.neo4j.queries import (
    PROJECT_CHUNK,
    PROJECT_DOCUMENT_HIERARCHY,
    PROJECT_FACT_EVIDENCE,
    PROJECT_FACT_REQUIREMENT_TRACE,
    PROJECT_REQUIREMENT,
    PROJECT_REQUIREMENT_FACT_LINK,
)
from multi_agentic_graph_rag.infrastructure.neo4j.schema import ensure_neo4j_schema

GraphPayload = Mapping[str, object]


def _as_neo4j_dict(payload: GraphPayload) -> dict[str, object]:
    """Convert a read-only mapping into a Neo4j parameter dictionary."""

    return dict(payload)


class Neo4jGraphStore(GraphStorePort):
    """Neo4j-backed graph projection adapter.

    This adapter consumes canonical application/PostgreSQL records and projects
    them into Neo4j using stable IDs and idempotent Cypher MERGE operations.
    """

    def __init__(self, *, driver: AsyncDriver, database: str) -> None:
        self._driver = driver
        self._database = database

    async def verify_schema(self) -> None:
        """Ensure required Neo4j uniqueness constraints exist."""

        await ensure_neo4j_schema(self._driver, database=self._database)

    async def project_document_hierarchy(
        self,
        *,
        project: GraphPayload,
        document: GraphPayload,
        document_version: GraphPayload,
        chunks: Sequence[GraphPayload],
        ingestion_run: GraphPayload,
    ) -> None:
        """Project Project -> Document -> DocumentVersion -> Chunk hierarchy."""

        project_payload = _as_neo4j_dict(project)
        document_payload = _as_neo4j_dict(document)
        document_version_payload = _as_neo4j_dict(document_version)
        ingestion_run_payload = _as_neo4j_dict(ingestion_run)

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._project_document_hierarchy_tx,
                project_payload,
                document_payload,
                document_version_payload,
                ingestion_run_payload,
            )

            document_version_id = str(document_version_payload["document_version_id"])

            for chunk in chunks:
                await session.execute_write(
                    self._project_chunk_tx,
                    document_version_id,
                    _as_neo4j_dict(chunk),
                )

    async def project_requirement_trace(
        self,
        *,
        facts: Sequence[GraphPayload],
        requirements: Sequence[GraphPayload],
        fact_evidence: Sequence[GraphPayload],
        requirement_fact_links: Sequence[GraphPayload],
        ingestion_run: GraphPayload,
    ) -> None:
        """Project Requirement -> Fact -> Chunk traceability graph."""

        ingestion_run_payload = _as_neo4j_dict(ingestion_run)

        async with self._driver.session(database=self._database) as session:
            for fact in facts:
                await session.execute_write(
                    self._project_fact_tx,
                    _as_neo4j_dict(fact),
                    ingestion_run_payload,
                )

            for evidence in fact_evidence:
                await session.execute_write(
                    self._project_fact_evidence_tx,
                    _as_neo4j_dict(evidence),
                )

            for requirement in requirements:
                await session.execute_write(
                    self._project_requirement_tx,
                    _as_neo4j_dict(requirement),
                    ingestion_run_payload,
                )

            for link in requirement_fact_links:
                await session.execute_write(
                    self._project_requirement_fact_link_tx,
                    _as_neo4j_dict(link),
                )

    @staticmethod
    async def _project_document_hierarchy_tx(
        tx: AsyncManagedTransaction,
        project: dict[str, object],
        document: dict[str, object],
        document_version: dict[str, object],
        ingestion_run: dict[str, object],
    ) -> None:
        result = await tx.run(
            PROJECT_DOCUMENT_HIERARCHY,
            project=project,
            document=document,
            document_version=document_version,
            ingestion_run=ingestion_run,
        )
        await result.consume()

    @staticmethod
    async def _project_chunk_tx(
        tx: AsyncManagedTransaction,
        document_version_id: str,
        chunk: dict[str, object],
    ) -> None:
        result = await tx.run(
            PROJECT_CHUNK,
            document_version_id=document_version_id,
            chunk=chunk,
        )
        await result.consume()

    @staticmethod
    async def _project_fact_tx(
        tx: AsyncManagedTransaction,
        fact: dict[str, object],
        ingestion_run: dict[str, object],
    ) -> None:
        result = await tx.run(
            PROJECT_FACT_REQUIREMENT_TRACE,
            fact=fact,
            ingestion_run=ingestion_run,
        )
        await result.consume()

    @staticmethod
    async def _project_fact_evidence_tx(
        tx: AsyncManagedTransaction,
        evidence: dict[str, object],
    ) -> None:
        result = await tx.run(
            PROJECT_FACT_EVIDENCE,
            fact_id=evidence["fact_id"],
            chunk_id=evidence["chunk_id"],
            exact_quote=evidence["exact_quote"],
            character_start=evidence["character_start"],
            character_end=evidence["character_end"],
        )
        await result.consume()

    @staticmethod
    async def _project_requirement_tx(
        tx: AsyncManagedTransaction,
        requirement: dict[str, object],
        ingestion_run: dict[str, object],
    ) -> None:
        result = await tx.run(
            PROJECT_REQUIREMENT,
            requirement=requirement,
            ingestion_run=ingestion_run,
        )
        await result.consume()

    @staticmethod
    async def _project_requirement_fact_link_tx(
        tx: AsyncManagedTransaction,
        link: dict[str, object],
    ) -> None:
        result = await tx.run(
            PROJECT_REQUIREMENT_FACT_LINK,
            requirement_id=link["requirement_id"],
            fact_id=link["fact_id"],
        )
        await result.consume()
