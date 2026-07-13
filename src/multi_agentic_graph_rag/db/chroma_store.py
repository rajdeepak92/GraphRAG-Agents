"""Chroma vector index adapter."""

from __future__ import annotations

from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import DocumentManifest
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel


class ChromaStore:
    """Coordinate chroma store behavior within the db boundary."""

    def __init__(self, settings: AppSettings) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (AppSettings): Validated settings that control this operation.
        """
        self.settings = settings
        self.collection_name = settings.chroma.collection_name

    def check(self) -> str:
        """Check check.

        Returns:
            str: The typed result produced by the operation.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
        self.settings.paths.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        client = self._client()
        collection = client.get_or_create_collection(self.collection_name)
        return f"PASS chroma collection={collection.name}"

    def index_chunks(self, manifest: DocumentManifest, embedding_model: EmbeddingModel) -> None:
        """Index chunks through the owning storage boundary.

        Args:
            manifest (DocumentManifest): Manifest required by the operation's typed contract.
            embedding_model (EmbeddingModel): Provider-neutral model adapter used by the operation.

        Side Effects:
            May write transactional or derivative state through the configured store.
            May invoke configured model or workflow providers.
        """
        client = self._client()
        collection = client.get_or_create_collection(self.collection_name)
        ids = [chunk.chunk_id for chunk in manifest.chunks]
        embeddings = embedding_model.embed_documents([chunk.text for chunk in manifest.chunks])
        metadatas = [
            {
                "project": manifest.project,
                "document_id": manifest.document_id,
                "document_version_id": manifest.document_version_id,
                "version": manifest.version,
                "page": chunk.page or 0,
                "section": chunk.section or "",
                "checksum": manifest.source_checksum,
                "embedding_fingerprint": embedding_model.embedding_fingerprint,
            }
            for chunk in manifest.chunks
        ]
        collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in manifest.chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query_chunks(
        self,
        query_embedding: list[float],
        document_version_id: str,
        n_results: int,
    ) -> list[tuple[str, str, float]]:
        """Return (chunk_id, document_text, distance) for the nearest chunks.

        Scoped to a single document version via metadata filtering.
        """
        if n_results <= 0:
            return []
        collection = self._client().get_or_create_collection(self.collection_name)
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"document_version_id": document_version_id},
        )
        ids = _first_row(result.get("ids"))
        documents = _first_row(result.get("documents"))
        distances = _first_row(result.get("distances"))
        matches: list[tuple[str, str, float]] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            distance = distances[index] if index < len(distances) else 0.0
            matches.append(
                (
                    str(chunk_id),
                    str(document) if document is not None else "",
                    float(distance) if distance is not None else 0.0,
                )
            )
        return matches

    def count_ids(self, ids: list[str]) -> int:
        """Count ids.

        Args:
            ids (list[str]): Ids required by the operation's typed contract.

        Returns:
            int: The typed result produced by the operation.
        """
        collection = self._client().get_or_create_collection(self.collection_name)
        result = collection.get(ids=ids)
        return len(result.get("ids", []))

    def _client(self) -> Any:
        """Execute the client operation within its declared architectural boundary.

        Returns:
            Any: The typed result produced by the operation.
        """
        import chromadb

        return chromadb.PersistentClient(path=str(self.settings.paths.chroma_persist_dir))


def _first_row(value: Any) -> list[Any]:
    """Execute the first row operation within its declared architectural boundary.

    Args:
        value (Any): Value required by the operation's typed contract.

    Returns:
        list[Any]: The typed result produced by the operation.
    """
    if not value:
        return []
    first = value[0]
    return list(first) if first is not None else []
