"""Project-collection Chroma adapter for chunk embeddings."""

from __future__ import annotations

import math
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.identifiers import normalize_project
from multi_agentic_graph_rag.domain.schemas import ManifestChunk


class ChromaStore:
    """Persist and retrieve only project-scoped chunk embeddings."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def collection_name(self, project: str) -> str:
        """Return the one collection name assigned to a project."""
        return f"{self.settings.chroma.collection_prefix}-{normalize_project(project)}"[:63]

    def check(self, project: str) -> str:
        """Validate connectivity and collection compatibility."""
        collection = self._client().get_or_create_collection(self.collection_name(project))
        return f"PASS chroma collection={collection.name}"

    def embedding_metadata(self, project: str) -> dict[str, Any] | None:
        """Return the persisted embedding contract for a non-empty collection."""
        result = (
            self._client()
            .get_or_create_collection(self.collection_name(project))
            .get(limit=1, include=["metadatas"])
        )
        ids = _as_list(result.get("ids"))
        if not ids:
            return None
        metadatas = _as_list(result.get("metadatas"))
        if not metadatas or metadatas[0] is None:
            return {}
        return dict(metadatas[0])

    def upsert_chunk(
        self,
        *,
        project: str,
        run_id: str,
        chunk: ManifestChunk,
        embedding: list[float],
        embedding_fingerprint: str,
    ) -> None:
        """Idempotently write one validated embedding record."""
        if not embedding or any(not math.isfinite(value) for value in embedding):
            raise ValueError("embedding must contain finite numeric values")
        self._client().get_or_create_collection(self.collection_name(project)).upsert(
            ids=[chunk.chunk_id],
            documents=[chunk.chunk_text],
            embeddings=[embedding],
            metadatas=[
                {
                    "project": project,
                    "chunk_id": chunk.chunk_id,
                    "content_hash": chunk.content_hash,
                    "run_id": run_id,
                    "embedding_dimension": len(embedding),
                    "embedding_fingerprint": embedding_fingerprint,
                }
            ],
        )

    def read_chunk(self, project: str, chunk_id: str) -> dict[str, Any] | None:
        """Read one record back for persistence validation."""
        result = (
            self._client()
            .get_or_create_collection(self.collection_name(project))
            .get(
                ids=[chunk_id],
                include=["documents", "metadatas", "embeddings"],
            )
        )
        ids = _as_list(result.get("ids"))
        if not ids:
            return None
        documents = _as_list(result.get("documents"))
        metadatas = _as_list(result.get("metadatas"))
        embeddings = _as_list(result.get("embeddings"))
        return {
            "id": str(ids[0]),
            "document": str(documents[0]) if documents and documents[0] is not None else "",
            "metadata": dict(metadatas[0]) if metadatas and metadatas[0] is not None else {},
            "embedding": list(embeddings[0]) if len(embeddings) else [],
        }

    def query_chunks(
        self,
        *,
        project: str,
        query_embedding: list[float],
        allowed_chunk_ids: set[str],
        n_results: int,
    ) -> list[tuple[str, str, float]]:
        """Query the project collection and enforce the manifest allowlist."""
        if not allowed_chunk_ids or n_results <= 0:
            return []
        collection = self._client().get_or_create_collection(self.collection_name(project))
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, len(allowed_chunk_ids), max(collection.count(), 1)),
            where=build_allowlist_where(project, allowed_chunk_ids),
            include=["documents", "distances", "metadatas"],
        )
        ids = _first(result.get("ids"))
        documents = _first(result.get("documents"))
        distances = _first(result.get("distances"))
        selected: list[tuple[str, str, float]] = []
        for index, value in enumerate(ids):
            chunk_id = str(value)
            if chunk_id not in allowed_chunk_ids:
                continue
            text = str(documents[index] or "") if index < len(documents) else ""
            distance = float(distances[index] or 0.0) if index < len(distances) else 0.0
            selected.append((chunk_id, text, distance))
            if len(selected) >= n_results:
                break
        return selected

    def delete_project(self, project: str) -> bool:
        """Delete the project's collection; return False when it did not exist."""
        try:
            self._client().delete_collection(self.collection_name(project))
        except Exception:
            return False
        return True

    def _client(self) -> Any:
        import chromadb

        self.settings.paths.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self.settings.paths.chroma_persist_dir))


def build_allowlist_where(project: str, allowed_chunk_ids: set[str]) -> dict[str, Any]:
    """Build a Chroma metadata filter scoping to the project and manifest allowlist.

    Pushes the current-run manifest chunk-ID allowlist server-side via ``$in`` so
    out-of-manifest candidates are never returned; the caller retains a post-hoc
    intersection as the final deterministic scope gate.
    """
    return {
        "$and": [
            {"project": project},
            {"chunk_id": {"$in": sorted(allowed_chunk_ids)}},
        ]
    }


def _as_list(value: Any) -> list[Any]:
    """Coerce a Chroma result field (possibly a numpy array or None) to a list."""
    if value is None:
        return []
    return list(value)


def _first(value: Any) -> list[Any]:
    if value is None or len(value) == 0:
        return []
    first = value[0]
    return list(first) if first is not None else []


__all__ = ["ChromaStore", "build_allowlist_where"]
