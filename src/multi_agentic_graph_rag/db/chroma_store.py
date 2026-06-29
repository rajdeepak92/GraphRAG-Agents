"""Chroma vector index adapter."""

from __future__ import annotations

from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import DocumentManifest
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel


class ChromaStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.collection_name = settings.chroma.collection_name

    def check(self) -> str:
        self.settings.paths.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        client = self._client()
        collection = client.get_or_create_collection(self.collection_name)
        return f"PASS chroma collection={collection.name}"

    def index_chunks(self, manifest: DocumentManifest, embedding_model: EmbeddingModel) -> None:
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

    def count_ids(self, ids: list[str]) -> int:
        collection = self._client().get_or_create_collection(self.collection_name)
        result = collection.get(ids=ids)
        return len(result.get("ids", []))

    def _client(self) -> Any:
        import chromadb

        return chromadb.PersistentClient(path=str(self.settings.paths.chroma_persist_dir))
