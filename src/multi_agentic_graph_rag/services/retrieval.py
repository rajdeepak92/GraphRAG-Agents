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
    REASON_BELOW_MIN_ASSERTIONS,
    REASON_KNOWLEDGE_DISABLED,
    REASON_NO_MANDATORY_ANCHOR,
    REASON_RETRIEVAL_ERROR,
    REASON_SELECTED,
    REASON_STAGE_NOT_PRIMARY,
    SOURCE_KNOWLEDGE,
    STATUS_FALLBACK,
    STATUS_SELECTED,
    GraphPrimaryDecision,
    KnowledgeRetrievalConfig,
    build_context_snapshot,
    build_semantic_snapshot,
    new_context_run_id,
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

    def decide_primary_context(
        self,
        *,
        requirement_text: str,
        document_version_id: str,
        evidence_chunk_ids: list[str],
        anchor_id: str = "",
    ) -> GraphPrimaryDecision:
        """Assemble structured assertion context once and return a typed decision.

        This is the single semantic-retrieval entry point: the assertion set is
        assembled exactly once here, recorded to ``generation_context_*`` under one
        ``context_run_id``, and (when the gate passes) returned for freezing into
        the checkpoint and the generation prompt. The ``reason`` names precisely
        why a fallback happened; callers emit the structured ``graph_fallback`` log
        and degrade to the legacy chunk path. Never raises.
        """
        knowledge = self.knowledge
        if knowledge is None:
            return GraphPrimaryDecision(status=STATUS_FALLBACK, reason=REASON_KNOWLEDGE_DISABLED)
        min_assertions = max(knowledge.min_assertions, 1)
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
        if context is None:
            return GraphPrimaryDecision(
                status=STATUS_FALLBACK,
                reason=REASON_RETRIEVAL_ERROR,
                min_assertions=min_assertions,
            )
        # Record the semantic snapshot exactly once, under a single shared id, so
        # shadow comparison, checkpoint freeze, and record persistence all agree.
        context_run_id = self._record_semantic_snapshot(context)
        assertion_count = len(context.items)
        mandatory_count = len(context.mandatory_anchor_ids)

        def _decide(status: str, reason: str, keep: bool) -> GraphPrimaryDecision:
            return GraphPrimaryDecision(
                status=status,
                reason=reason,
                context=context if keep else None,
                assertion_count=assertion_count,
                mandatory_anchor_count=mandatory_count,
                min_assertions=min_assertions,
                context_run_id=context_run_id,
                metrics=dict(context.metrics),
            )

        if not knowledge.include_in_context:
            return _decide(STATUS_FALLBACK, REASON_STAGE_NOT_PRIMARY, keep=False)
        if not context.mandatory_anchor_ids:
            return _decide(STATUS_FALLBACK, REASON_NO_MANDATORY_ANCHOR, keep=False)
        if assertion_count < min_assertions:
            return _decide(STATUS_FALLBACK, REASON_BELOW_MIN_ASSERTIONS, keep=False)
        return _decide(STATUS_SELECTED, REASON_SELECTED, keep=True)

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
        """Persist the legacy chunk retrieval snapshot for shadow comparison.

        The structured assertion channel is recorded separately and exactly once
        by :meth:`decide_primary_context`, so this no longer re-assembles it.
        """
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

    def _record_semantic_snapshot(self, context: SemanticContext) -> str:
        """Record the already-assembled structured assertion snapshot (§18) once.

        Returns the ``context_run_id`` used for the snapshot so the exact same id
        is frozen into the checkpoint and persisted on the generated record. The
        assertion set is assembled by the caller; this method never re-assembles
        it, which is what guarantees a single production of the semantic context.
        """
        knowledge = self.knowledge
        recorder = knowledge.recorder if knowledge is not None else None
        context_run_id = new_context_run_id()
        if knowledge is None or recorder is None or not context.items:
            return context_run_id
        run, items = build_semantic_snapshot(
            config=knowledge, context=context, context_run_id=context_run_id
        )
        self._safe(lambda: recorder(run, items), "semantic_snapshot", None)
        return context_run_id

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

    @property
    def _log_step(self) -> str:
        """Stage-specific retrieval log step (so scenario logs are not labelled
        as user-story retrieval)."""
        stage = self.knowledge.stage if self.knowledge is not None else "generation"
        return f"{stage}.retrieve"

    def _safe(self, operation: Callable[[], T], label: str, default: T) -> T:
        try:
            return operation()
        except Exception as exc:
            if self.logger is not None:
                self.logger.warning(
                    "Hybrid retrieval step failed; continuing with partial context",
                    step=self._log_step,
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
            step=self._log_step,
            document_version_id=document_version_id,
            source=source,
            status="degraded",
        )
