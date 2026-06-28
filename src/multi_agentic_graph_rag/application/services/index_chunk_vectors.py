"""Index canonical chunks into the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from multi_agentic_graph_rag.application.ports.repositories import ChunkRepository
from multi_agentic_graph_rag.application.ports.vector_store import EmbeddingPort, VectorStorePort
from multi_agentic_graph_rag.domain.vectors import VectorRecord


@dataclass(frozen=True, slots=True)
class IndexChunkVectorsResult:
    """Result returned after indexing chunks for one document version."""

    document_version_id: UUID
    indexed_count: int
    embedding_fingerprint: str


class IndexChunkVectorsService:
    """Application service for indexing canonical PostgreSQL chunks into a vector store."""

    def __init__(
        self,
        *,
        chunk_repository: ChunkRepository,
        embedding_provider: EmbeddingPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._chunk_repository = chunk_repository
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    async def index_document_version(
        self,
        *,
        document_version_id: UUID,
    ) -> IndexChunkVectorsResult:
        """Index all active chunks for a single document version."""

        chunks = await self._chunk_repository.list_active_by_document_version(
            document_version_id=document_version_id,
        )

        embedding_fingerprint = self._embedding_provider.fingerprint()

        records = tuple(
            VectorRecord(
                chunk_id=chunk.chunk_id,
                document_version_id=str(chunk.document_version_id),
                normalized_text=chunk.normalized_text,
                content_hash=chunk.content_hash,
                ordinal=chunk.ordinal,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=chunk.section_path,
                embedding_fingerprint=embedding_fingerprint,
            )
            for chunk in chunks
        )

        embeddings = self._embedding_provider.embed_documents(
            tuple(record.normalized_text for record in records)
        )

        indexed_count = self._vector_store.upsert_chunks(
            records=records,
            embeddings=embeddings,
        )

        return IndexChunkVectorsResult(
            document_version_id=document_version_id,
            indexed_count=indexed_count,
            embedding_fingerprint=embedding_fingerprint,
        )
