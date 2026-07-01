"""Hybrid per-requirement context retrieval reused by generation stages.

Fuses dense (Chroma), sparse/BM25 (Neo4j full-text) and graph-neighbour
(Neo4j multi-hop) candidate chunks, reranks them with the cross-encoder, and
degrades gracefully to the requirement's own evidence chunks (or just the
requirement text) so generation is never blocked by an empty retrieval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from multi_agentic_graph_rag.config.settings import UserStorySettings
from multi_agentic_graph_rag.db.chroma_store import ChromaStore
from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, RerankerModel
from multi_agentic_graph_rag.observability.logging import RunLogger

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str


@dataclass
class RetrievedContext:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    source: str = "hybrid"


class RetrievalService:
    def __init__(
        self,
        *,
        chroma: ChromaStore,
        neo4j: Neo4jStore,
        embedding_model: EmbeddingModel,
        reranker_model: RerankerModel | None,
        settings: UserStorySettings,
        logger: RunLogger | None = None,
    ) -> None:
        self.chroma = chroma
        self.neo4j = neo4j
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.settings = settings
        self.logger = logger

    def retrieve_context(
        self,
        *,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
    ) -> RetrievedContext:
        fused: dict[str, str] = {}
        evidence_ids: list[str] = []
        empty_pairs: list[tuple[str, str]] = []
        empty_scored: list[tuple[str, str, float]] = []

        if evidence_chunk_ids:
            for chunk_id, text in self._safe(
                lambda: self.neo4j.fetch_chunks(evidence_chunk_ids),
                "evidence",
                empty_pairs,
            ):
                if chunk_id not in fused:
                    fused[chunk_id] = text
                    evidence_ids.append(chunk_id)

        retrieved = 0
        for chunk_id, text, _ in self._safe(
            lambda: self._dense(requirement_text, document_version_id),
            "dense",
            empty_scored,
        ):
            if chunk_id not in fused:
                fused[chunk_id] = text
                retrieved += 1

        for chunk_id, text, _ in self._safe(
            lambda: self.neo4j.fulltext_search_chunks(
                requirement_text, document_version_id, self.settings.sparse_k
            ),
            "sparse",
            empty_scored,
        ):
            if chunk_id not in fused:
                fused[chunk_id] = text
                retrieved += 1

        if evidence_chunk_ids:
            for chunk_id, text in self._safe(
                lambda: self.neo4j.neighbor_chunks(
                    evidence_chunk_ids, document_version_id, self.settings.neighbor_window
                ),
                "graph",
                empty_pairs,
            ):
                if chunk_id not in fused:
                    fused[chunk_id] = text
                    retrieved += 1

        if not fused:
            self._log_fallback(document_version_id, source="requirement_text_fallback")
            return RetrievedContext(chunks=[], source="requirement_text_fallback")

        ranked = self._rerank(requirement_text, fused)
        top_k = max(self.settings.top_k, 0)
        selected = ranked[:top_k]
        selected_ids = {chunk.chunk_id for chunk in selected}
        for chunk_id in evidence_ids:
            if chunk_id not in selected_ids:
                selected.append(RetrievedChunk(chunk_id=chunk_id, text=fused[chunk_id]))
                selected_ids.add(chunk_id)

        source = "hybrid" if retrieved > 0 else "evidence"
        if source == "evidence":
            self._log_fallback(document_version_id, source=source)
        return RetrievedContext(chunks=selected, source=source)

    def _dense(
        self,
        requirement_text: str,
        document_version_id: str,
    ) -> list[tuple[str, str, float]]:
        embeddings = self.embedding_model.embed_documents([requirement_text])
        if not embeddings:
            return []
        return self.chroma.query_chunks(embeddings[0], document_version_id, self.settings.dense_k)

    def _rerank(self, query: str, fused: dict[str, str]) -> list[RetrievedChunk]:
        ids = list(fused)
        texts = [fused[chunk_id] for chunk_id in ids]
        order = list(range(len(ids)))
        reranker = self.reranker_model
        if reranker is not None and len(texts) > 1:
            ranked = self._safe(lambda: reranker.rerank(query, texts), "rerank", order)
            if sorted(ranked) == order:
                order = ranked
        return [RetrievedChunk(chunk_id=ids[index], text=texts[index]) for index in order]

    def _safe(self, operation: Callable[[], T], label: str, default: T) -> T:
        try:
            return operation()
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    "Hybrid retrieval step failed; continuing with partial context",
                    step="user_stories.retrieve",
                    retrieval_step=label,
                    error=str(exc),
                    status="degraded",
                )
            return default

    def _log_fallback(self, document_version_id: str, *, source: str) -> None:
        if self.logger is None:
            return
        self.logger.warning(
            "Retrieval degraded for requirement; generating from limited context",
            step="user_stories.retrieve",
            document_version_id=document_version_id,
            source=source,
            status="degraded",
        )
