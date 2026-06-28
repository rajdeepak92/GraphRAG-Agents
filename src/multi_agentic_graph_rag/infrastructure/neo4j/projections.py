"""Neo4j projection adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from neo4j import AsyncDriver, AsyncManagedTransaction

from multi_agentic_graph_rag.application.ports.graph_store import (
    FactRelationType,
    GraphPayload,
    GraphStorePort,
    RequirementRelationType,
)
from multi_agentic_graph_rag.infrastructure.neo4j.queries import (
    PROJECT_CHUNK,
    PROJECT_DOCUMENT_HIERARCHY,
    PROJECT_DOCUMENT_VERSION_LINEAGE,
    PROJECT_FACT_CONFLICT,
    PROJECT_FACT_EVIDENCE,
    PROJECT_FACT_REQUIREMENT_TRACE,
    PROJECT_REQUIREMENT,
    PROJECT_REQUIREMENT_CONFLICT,
    PROJECT_REQUIREMENT_DUPLICATE,
    PROJECT_REQUIREMENT_FACT_LINK,
    PROJECT_REQUIREMENT_SUPERSEDES,
)
from multi_agentic_graph_rag.infrastructure.neo4j.schema import ensure_neo4j_schema

_FACT_RELATION_QUERIES: dict[FactRelationType, str] = {
    "conflicts_with": PROJECT_FACT_CONFLICT,
}

_REQUIREMENT_RELATION_QUERIES: dict[RequirementRelationType, str] = {
    "conflicts_with": PROJECT_REQUIREMENT_CONFLICT,
    "duplicates": PROJECT_REQUIREMENT_DUPLICATE,
    "supersedes": PROJECT_REQUIREMENT_SUPERSEDES,
}


def _as_neo4j_dict(payload: GraphPayload) -> dict[str, object]:
    """Convert a read-only mapping into a Neo4j parameter dictionary."""

    return dict(payload)


def _relation_properties(payload: GraphPayload | None) -> dict[str, Any]:
    """Normalize optional relation metadata for Neo4j relationship properties."""

    properties: dict[str, Any] = dict(payload or {})

    return {
        "relation_source": properties.get("relation_source", "deterministic_projection"),
        "run_id": properties.get("run_id"),
        "reason": properties.get("reason"),
    }


def _reject_self_relation(
    source_id: str,
    target_id: str,
    *,
    relation_type: str,
) -> None:
    """Reject graph relations that point a node to itself."""

    if source_id == target_id:
        raise ValueError(f"{relation_type} relation cannot point to the same node.")


def _canonical_symmetric_pair(source_id: str, target_id: str) -> tuple[str, str]:
    """Return a stable direction for symmetric relations.

    CONFLICTS_WITH and DUPLICATES are semantically symmetric. Canonicalizing the
    pair prevents both A -> B and B -> A from being stored as duplicate edges.
    """

    first, second = sorted((source_id, target_id))
    return first, second


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

    async def project_document_version_lineage(
        self,
        *,
        document_version_id: str,
        supersedes_document_version_id: str,
    ) -> None:
        """Project DocumentVersion -> SUPERSEDES -> DocumentVersion lineage."""

        _reject_self_relation(
            document_version_id,
            supersedes_document_version_id,
            relation_type="document_version_supersedes",
        )

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._project_document_version_lineage_tx,
                document_version_id,
                supersedes_document_version_id,
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

    async def project_fact_relation(
        self,
        *,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: FactRelationType,
        relation_properties: GraphPayload | None = None,
    ) -> None:
        """Project Fact -> Fact deterministic relation.

        Currently supported:
        - conflicts_with
        """

        _reject_self_relation(
            source_fact_id,
            target_fact_id,
            relation_type=relation_type,
        )

        # Fact conflicts are symmetric, so store only one canonical direction.
        source_fact_id, target_fact_id = _canonical_symmetric_pair(
            source_fact_id,
            target_fact_id,
        )

        query = _FACT_RELATION_QUERIES[relation_type]
        properties = _relation_properties(relation_properties)

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._project_fact_relation_tx,
                query,
                source_fact_id,
                target_fact_id,
                properties,
            )

    async def project_requirement_relation(
        self,
        *,
        source_requirement_id: str,
        target_requirement_id: str,
        relation_type: RequirementRelationType,
        relation_properties: GraphPayload | None = None,
    ) -> None:
        """Project Requirement -> Requirement deterministic relation.

        Supported:
        - conflicts_with: symmetric
        - duplicates: symmetric
        - supersedes: directional
        """

        _reject_self_relation(
            source_requirement_id,
            target_requirement_id,
            relation_type=relation_type,
        )

        if relation_type in {"conflicts_with", "duplicates"}:
            source_requirement_id, target_requirement_id = _canonical_symmetric_pair(
                source_requirement_id,
                target_requirement_id,
            )

        query = _REQUIREMENT_RELATION_QUERIES[relation_type]
        properties = _relation_properties(relation_properties)

        async with self._driver.session(database=self._database) as session:
            await session.execute_write(
                self._project_requirement_relation_tx,
                query,
                source_requirement_id,
                target_requirement_id,
                properties,
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
    async def _project_document_version_lineage_tx(
        tx: AsyncManagedTransaction,
        document_version_id: str,
        supersedes_document_version_id: str,
    ) -> None:
        result = await tx.run(
            PROJECT_DOCUMENT_VERSION_LINEAGE,
            document_version_id=document_version_id,
            supersedes_document_version_id=supersedes_document_version_id,
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

    @staticmethod
    async def _project_fact_relation_tx(
        tx: AsyncManagedTransaction,
        query: str,
        source_fact_id: str,
        target_fact_id: str,
        properties: dict[str, Any],
    ) -> None:
        result = await tx.run(
            query,
            source_fact_id=source_fact_id,
            target_fact_id=target_fact_id,
            **properties,
        )
        await result.consume()

    @staticmethod
    async def _project_requirement_relation_tx(
        tx: AsyncManagedTransaction,
        query: str,
        source_requirement_id: str,
        target_requirement_id: str,
        properties: dict[str, Any],
    ) -> None:
        result = await tx.run(
            query,
            source_requirement_id=source_requirement_id,
            target_requirement_id=target_requirement_id,
            **properties,
        )
        await result.consume()
