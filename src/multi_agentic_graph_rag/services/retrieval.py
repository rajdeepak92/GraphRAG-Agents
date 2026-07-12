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
from multi_agentic_graph_rag.domain.schemas import AssertionContextItem, SemanticContext
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, RerankerModel
from multi_agentic_graph_rag.observability.logging import RunLogger
from multi_agentic_graph_rag.services.knowledge_context import (
    assemble_semantic_context,
    config_for_stage,
)
from multi_agentic_graph_rag.services.knowledge_retrieval import (
    SOURCE_KNOWLEDGE,
    KnowledgeRetrievalConfig,
    build_context_snapshot,
    build_semantic_snapshot,
)

T = TypeVar("T")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str


@dataclass
class RetrievedContext:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    source: str = "hybrid"
    assertions: list[AssertionContextItem] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkProvenance:
    """Per-chunk retrieval provenance used to build the loop-1 context map.

    Carries identifiers and retrieval metadata only (never chunk text), so it can
    be checkpointed to ``context_map.json`` without duplicating source content.
    """

    chunk_id: str
    source: str
    rank: int
    score: float | None = None


# Retrieval-source labels recorded per fused chunk (stable identifiers used in
# the context-map ``retrieval_metadata`` records).
_SOURCE_EVIDENCE = "evidence"
_SOURCE_DENSE = "chroma_dense"
_SOURCE_SPARSE = "neo4j_fulltext"
_SOURCE_NEIGHBOR = "neo4j_neighbor"
_SOURCE_KNOWLEDGE = SOURCE_KNOWLEDGE


@dataclass(frozen=True)
class _SelectedChunk:
    chunk_id: str
    text: str
    source: str
    score: float | None


@dataclass(frozen=True)
class _Selection:
    items: list[_SelectedChunk]
    source: str


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
        knowledge: KnowledgeRetrievalConfig | None = None,
    ) -> None:
        self.chroma = chroma
        self.neo4j = neo4j
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.settings = settings
        self.logger = logger
        self.knowledge = knowledge

    def retrieve_context(
        self,
        *,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
        anchor_id: str = "",
    ) -> RetrievedContext:
        selection = self._retrieve(
            requirement_text=requirement_text,
            document_version_id=document_version_id,
            evidence_chunk_ids=evidence_chunk_ids,
            anchor_id=anchor_id,
        )
        return RetrievedContext(
            chunks=[
                RetrievedChunk(chunk_id=item.chunk_id, text=item.text) for item in selection.items
            ],
            source=selection.source,
        )

    def retrieve_context_map(
        self,
        *,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
        anchor_id: str = "",
    ) -> list[ChunkProvenance]:
        """Loop-1 retrieval: identifiers + provenance only, never chunk text.

        Runs the identical hybrid fusion + rerank + evidence-guarantee selection
        as :meth:`retrieve_context`, but returns only the selected chunk ids with
        their retrieval source, deterministic rank order, and (dense/sparse) score
        so the result can be checkpointed to ``context_map.json``.
        """
        selection = self._retrieve(
            requirement_text=requirement_text,
            document_version_id=document_version_id,
            evidence_chunk_ids=evidence_chunk_ids,
            anchor_id=anchor_id,
        )
        return [
            ChunkProvenance(
                chunk_id=item.chunk_id,
                source=item.source,
                rank=rank,
                score=item.score,
            )
            for rank, item in enumerate(selection.items, start=1)
        ]

    def assemble_primary_context(
        self,
        *,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
        anchor_id: str = "",
    ) -> SemanticContext | None:
        """Assemble structured assertion context when the stage is graph-primary.

        Returns ``None`` (legacy chunk fallback) when the knowledge channel is
        off, the stage is not graph-primary, or the graph fails the grounding
        gate — no mandatory anchor for this requirement, or fewer than
        ``min_assertions`` selected assertions (plan §17 fallback conditions).
        Never raises: any retrieval error degrades to the legacy path.
        """
        knowledge = self.knowledge
        if knowledge is None or not knowledge.include_in_context:
            return None
        context = self._safe(
            lambda: assemble_semantic_context(
                self.neo4j,
                anchor_id=anchor_id,
                query_text=requirement_text,
                document_version_id=document_version_id,
                evidence_chunk_ids=evidence_chunk_ids,
                config=config_for_stage(knowledge.stage),
            ),
            "semantic_primary",
            None,
        )
        if context is None or not context.mandatory_anchor_ids:
            return None
        if len(context.items) < max(knowledge.min_assertions, 1):
            return None
        return context

    def _retrieve(
        self,
        *,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
        anchor_id: str = "",
    ) -> _Selection:
        fused: dict[str, str] = {}
        source_by_id: dict[str, str] = {}
        score_by_id: dict[str, float | None] = {}
        evidence_ids: list[str] = []
        empty_pairs: list[tuple[str, str]] = []
        empty_scored: list[tuple[str, str, float]] = []

        def add(chunk_id: str, text: str, source: str, score: float | None) -> bool:
            if chunk_id in fused:
                return False
            fused[chunk_id] = text
            source_by_id[chunk_id] = source
            score_by_id[chunk_id] = score
            return True

        if evidence_chunk_ids:
            for chunk_id, text in self._safe(
                lambda: self.neo4j.fetch_chunks(evidence_chunk_ids),
                "evidence",
                empty_pairs,
            ):
                if add(chunk_id, text, _SOURCE_EVIDENCE, None):
                    evidence_ids.append(chunk_id)

        retrieved = 0
        for chunk_id, text, score in self._safe(
            lambda: self._dense(requirement_text, document_version_id),
            "dense",
            empty_scored,
        ):
            if add(chunk_id, text, _SOURCE_DENSE, score):
                retrieved += 1

        for chunk_id, text, score in self._safe(
            lambda: self.neo4j.fulltext_search_chunks(
                requirement_text, document_version_id, self.settings.sparse_k
            ),
            "sparse",
            empty_scored,
        ):
            if add(chunk_id, text, _SOURCE_SPARSE, score):
                retrieved += 1

        if evidence_chunk_ids:
            for chunk_id, text in self._safe(
                lambda: self.neo4j.neighbor_chunks(
                    evidence_chunk_ids, document_version_id, self.settings.neighbor_window
                ),
                "graph",
                empty_pairs,
            ):
                if add(chunk_id, text, _SOURCE_NEIGHBOR, None):
                    retrieved += 1

        knowledge_candidates: list[tuple[str, str, float]] = []
        if self.knowledge is not None and evidence_chunk_ids:
            expansion_k = self.knowledge.expansion_k
            knowledge_candidates = self._safe(
                lambda: self.neo4j.knowledge_related_chunks(
                    evidence_chunk_ids, document_version_id, expansion_k
                ),
                "knowledge",
                empty_scored,
            )
            if self.knowledge.include_in_context:
                for chunk_id, text, score in knowledge_candidates:
                    if add(chunk_id, text, _SOURCE_KNOWLEDGE, score):
                        retrieved += 1

        if not fused:
            self._log_fallback(document_version_id, source="requirement_text_fallback")
            self._record_snapshot(
                anchor_id=anchor_id,
                requirement_text=requirement_text,
                document_version_id=document_version_id,
                evidence_chunk_ids=evidence_chunk_ids,
                selection=_Selection(items=[], source="requirement_text_fallback"),
                knowledge_candidates=knowledge_candidates,
            )
            return _Selection(items=[], source="requirement_text_fallback")

        ranked_ids = self._rerank_ids(requirement_text, fused)
        top_k = max(self.settings.top_k, 0)
        selected_ids = ranked_ids[:top_k]
        selected_set = set(selected_ids)
        for chunk_id in evidence_ids:
            if chunk_id not in selected_set:
                selected_ids.append(chunk_id)
                selected_set.add(chunk_id)

        items = [
            _SelectedChunk(
                chunk_id=chunk_id,
                text=fused[chunk_id],
                source=source_by_id[chunk_id],
                score=score_by_id[chunk_id],
            )
            for chunk_id in selected_ids
        ]
        source = "hybrid" if retrieved > 0 else "evidence"
        if source == "evidence":
            self._log_fallback(document_version_id, source=source)
        selection = _Selection(items=items, source=source)
        self._record_snapshot(
            anchor_id=anchor_id,
            requirement_text=requirement_text,
            document_version_id=document_version_id,
            evidence_chunk_ids=evidence_chunk_ids,
            selection=selection,
            knowledge_candidates=knowledge_candidates,
        )
        return selection

    def _record_snapshot(
        self,
        *,
        anchor_id: str,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
        selection: _Selection,
        knowledge_candidates: list[tuple[str, str, float]],
    ) -> None:
        """Persist the retrieval snapshot(s) for shadow comparison; never blocks."""
        knowledge = self.knowledge
        recorder = knowledge.recorder if knowledge is not None else None
        if knowledge is None or recorder is None:
            return
        run, items = build_context_snapshot(
            config=knowledge,
            anchor_id=anchor_id,
            document_version_id=document_version_id,
            selection_source=selection.source,
            selected_items=[(item.chunk_id, item.source, item.score) for item in selection.items],
            knowledge_candidates=[(chunk_id, score) for chunk_id, _, score in knowledge_candidates],
        )
        self._safe(lambda: recorder(run, items), "snapshot", None)
        self._record_semantic_snapshot(
            anchor_id=anchor_id,
            requirement_text=requirement_text,
            document_version_id=document_version_id,
            evidence_chunk_ids=evidence_chunk_ids,
        )

    def _record_semantic_snapshot(
        self,
        *,
        anchor_id: str,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
    ) -> None:
        """Assemble and record the structured assertion snapshot (shadow §18).

        Increment 2 contract: this records the knowledge-graph assertion channel
        for shadow comparison only; it never feeds the generation prompt, so the
        selected chunk context above is unchanged.
        """
        knowledge = self.knowledge
        recorder = knowledge.recorder if knowledge is not None else None
        if knowledge is None or recorder is None:
            return
        context = self._safe(
            lambda: assemble_semantic_context(
                self.neo4j,
                anchor_id=anchor_id,
                query_text=requirement_text,
                document_version_id=document_version_id,
                evidence_chunk_ids=evidence_chunk_ids,
                config=config_for_stage(knowledge.stage),
            ),
            "semantic",
            None,
        )
        if context is None or not context.items:
            return
        run, items = build_semantic_snapshot(config=knowledge, context=context)
        self._safe(lambda: recorder(run, items), "semantic_snapshot", None)

    def _dense(
        self,
        requirement_text: str,
        document_version_id: str,
    ) -> list[tuple[str, str, float]]:
        embeddings = self.embedding_model.embed_documents([requirement_text])
        if not embeddings:
            return []
        return self.chroma.query_chunks(embeddings[0], document_version_id, self.settings.dense_k)

    def _rerank_ids(self, query: str, fused: dict[str, str]) -> list[str]:
        ids = list(fused)
        texts = [fused[chunk_id] for chunk_id in ids]
        order = list(range(len(ids)))
        reranker = self.reranker_model
        if reranker is not None and len(texts) > 1:
            ranked = self._safe(lambda: reranker.rerank(query, texts), "rerank", order)
            if sorted(ranked) == order:
                order = ranked
        return [ids[index] for index in order]

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
