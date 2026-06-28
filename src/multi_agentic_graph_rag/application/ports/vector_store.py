"""Vector-store application port."""

from __future__ import annotations

from typing import Protocol

from multi_agentic_graph_rag.domain.vectors import VectorRecord, VectorSearchResult


class EmbeddingPort(Protocol):
    def fingerprint(self) -> str: ...

    def embed_documents(self, texts: tuple[str, ...]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStorePort(Protocol):
    def verify_connection(self) -> None: ...

    def upsert_chunks(
        self,
        *,
        records: tuple[VectorRecord, ...],
        embeddings: list[list[float]],
    ) -> int: ...

    def search_chunks(
        self,
        *,
        query_embedding: list[float],
        n_results: int,
        document_version_id: str | None = None,
    ) -> tuple[VectorSearchResult, ...]: ...
