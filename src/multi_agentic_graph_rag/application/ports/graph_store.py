"""Graph-store application port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

GraphPayload = Mapping[str, object]
FactRelationType = Literal["conflicts_with"]

RequirementRelationType = Literal[
    "conflicts_with",
    "duplicates",
    "supersedes",
]


class GraphStorePort(Protocol):
    async def verify_schema(self) -> None: ...

    async def project_document_hierarchy(
        self,
        *,
        project: GraphPayload,
        document: GraphPayload,
        document_version: GraphPayload,
        chunks: Sequence[GraphPayload],
        ingestion_run: GraphPayload,
    ) -> None: ...

    async def project_document_version_lineage(
        self,
        *,
        document_version_id: str,
        supersedes_document_version_id: str,
    ) -> None: ...

    async def project_requirement_trace(
        self,
        *,
        facts: Sequence[GraphPayload],
        requirements: Sequence[GraphPayload],
        fact_evidence: Sequence[GraphPayload],
        requirement_fact_links: Sequence[GraphPayload],
        ingestion_run: GraphPayload,
    ) -> None: ...

    async def project_fact_relation(
        self,
        *,
        source_fact_id: str,
        target_fact_id: str,
        relation_type: FactRelationType,
        relation_properties: GraphPayload | None = None,
    ) -> None: ...

    async def project_requirement_relation(
        self,
        *,
        source_requirement_id: str,
        target_requirement_id: str,
        relation_type: RequirementRelationType,
        relation_properties: GraphPayload | None = None,
    ) -> None: ...
