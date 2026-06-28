"""Chroma vector-store adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from chromadb.api import ClientAPI

from multi_agentic_graph_rag.application.ports.vector_store import VectorStorePort
from multi_agentic_graph_rag.domain.vectors import VectorRecord, VectorSearchResult

MetadataValue = str | int | float | bool


class ChromaVectorStore(VectorStorePort):
    def __init__(
        self,
        *,
        client: ClientAPI,
        collection_name: str,
    ) -> None:
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"owner": "multi-agentic-graph-rag", "record_type": "chunk"},
        )

    def verify_connection(self) -> None:
        self._client.heartbeat()
        self._collection.count()

    def upsert_chunks(
        self,
        *,
        records: tuple[VectorRecord, ...],
        embeddings: list[list[float]],
    ) -> int:
        if len(records) != len(embeddings):
            msg = "records and embeddings length mismatch."
            raise ValueError(msg)

        if not records:
            return 0

        collection = cast(Any, self._collection)
        collection.upsert(
            ids=[record.chunk_id for record in records],
            embeddings=embeddings,
            documents=[record.normalized_text for record in records],
            metadatas=[_metadata(record) for record in records],
        )

        return len(records)

    def search_chunks(
        self,
        *,
        query_embedding: list[float],
        n_results: int,
        document_version_id: str | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        where = _where_document_version(document_version_id)

        collection = cast(Any, self._collection)
        result = cast(
            Mapping[str, Any],
            collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            ),
        )

        ids = cast(list[list[str]], result.get("ids") or [[]])[0]
        documents = cast(list[list[str]], result.get("documents") or [[]])[0]
        metadatas = cast(
            list[list[Mapping[str, object]]],
            result.get("metadatas") or [[]],
        )[0]
        distances = cast(list[list[float]], result.get("distances") or [[]])[0]

        output: list[VectorSearchResult] = []

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            safe_metadata = {
                str(key): _safe_metadata_value(value) for key, value in metadata.items()
            }

            output.append(
                VectorSearchResult(
                    chunk_id=str(chunk_id),
                    document_version_id=str(safe_metadata["document_version_id"]),
                    text=str(document),
                    distance=float(distance),
                    metadata=safe_metadata,
                )
            )

        return tuple(output)


def _where_document_version(document_version_id: str | None) -> dict[str, str] | None:
    if document_version_id is None:
        return None

    return {"document_version_id": document_version_id}


def _metadata(record: VectorRecord) -> dict[str, MetadataValue]:
    metadata: dict[str, MetadataValue] = {
        "chunk_id": record.chunk_id,
        "document_version_id": record.document_version_id,
        "content_hash": record.content_hash,
        "ordinal": record.ordinal,
        "section_path": " > ".join(record.section_path),
        "embedding_fingerprint": record.embedding_fingerprint,
    }

    if record.page_start is not None:
        metadata["page_start"] = record.page_start

    if record.page_end is not None:
        metadata["page_end"] = record.page_end

    return metadata


def _safe_metadata_value(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return ", ".join(str(item) for item in value)

    return str(value)
