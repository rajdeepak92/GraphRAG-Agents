"""Graph-store application port."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

GraphPayload = Mapping[str, object]


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

    async def project_requirement_trace(
        self,
        *,
        facts: Sequence[GraphPayload],
        requirements: Sequence[GraphPayload],
        fact_evidence: Sequence[GraphPayload],
        requirement_fact_links: Sequence[GraphPayload],
        ingestion_run: GraphPayload,
    ) -> None: ...
