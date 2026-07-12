"""Flag-gated knowledge-graph retrieval integration for generation stages.

Rollout contract (see ``KnowledgeGraphSettings``): with the graph-primary flag
off and shadow mode on, generation context is byte-identical to the legacy
hybrid retrieval and graph expansion only records comparison snapshots to
PostgreSQL. Flipping the stage's graph-primary flag adds assertion-hop
candidates to the fusion pool, where the reranker and the evidence-chunk
guarantee still apply.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import (
    GenerationContextItem,
    GenerationContextRun,
    SemanticContext,
)

SOURCE_KNOWLEDGE = "neo4j_knowledge"
SOURCE_SEMANTIC = "graph_semantic"

ContextRecorder = Callable[[GenerationContextRun, list[GenerationContextItem]], None]


@dataclass(frozen=True)
class KnowledgeRetrievalConfig:
    stage: str
    project: str
    include_in_context: bool
    shadow: bool
    expansion_k: int
    recorder: ContextRecorder | None
    min_assertions: int = 0


def build_knowledge_retrieval_config(
    settings: AppSettings,
    *,
    stage: str,
    project: str,
    primary: bool,
    recorder: ContextRecorder | None,
) -> KnowledgeRetrievalConfig | None:
    """Resolve the stage's knowledge-retrieval mode; ``None`` disables it all."""
    knowledge_graph = settings.knowledge_graph
    if not knowledge_graph.enabled:
        return None
    if not primary and not knowledge_graph.shadow_mode:
        return None
    return KnowledgeRetrievalConfig(
        stage=stage,
        project=project,
        include_in_context=primary,
        shadow=knowledge_graph.shadow_mode,
        expansion_k=knowledge_graph.expansion_k,
        recorder=recorder,
        min_assertions=knowledge_graph.graph_min_assertions,
    )


def build_context_snapshot(
    *,
    config: KnowledgeRetrievalConfig,
    anchor_id: str,
    document_version_id: str,
    selection_source: str,
    selected_items: list[tuple[str, str, float | None]],
    knowledge_candidates: list[tuple[str, float]],
) -> tuple[GenerationContextRun, list[GenerationContextItem]]:
    """Assemble one retrieval snapshot from the final selection and shadow pool.

    ``selected_items`` is the final context in rank order as (chunk_id, source,
    score); ``knowledge_candidates`` is the graph expansion pool as (chunk_id,
    score) — candidates that did not make the selection are recorded with
    ``selected=False`` for shadow comparison.
    """
    context_run_id = f"CTX-{uuid4().hex[:16].upper()}"
    items: list[GenerationContextItem] = []
    selected_ids: set[str] = set()
    for rank, (chunk_id, source, score) in enumerate(selected_items, start=1):
        selected_ids.add(chunk_id)
        items.append(
            GenerationContextItem(
                context_run_id=context_run_id,
                rank=rank,
                item_id=chunk_id,
                source=source,
                score=score,
                selected=True,
            )
        )
    rank = len(items)
    for chunk_id, score in knowledge_candidates:
        if chunk_id in selected_ids:
            continue
        rank += 1
        items.append(
            GenerationContextItem(
                context_run_id=context_run_id,
                rank=rank,
                item_id=chunk_id,
                source=SOURCE_KNOWLEDGE,
                score=score,
                selected=False,
            )
        )

    source_counts: dict[str, int] = {}
    for item in items:
        if item.selected:
            source_counts[item.source] = source_counts.get(item.source, 0) + 1
    run = GenerationContextRun(
        context_run_id=context_run_id,
        stage=config.stage,
        anchor_id=anchor_id,
        project=config.project,
        document_version_id=document_version_id,
        source=selection_source,
        metrics={
            "selected_count": len(selected_ids),
            "selected_by_source": source_counts,
            "knowledge_candidate_count": len(knowledge_candidates),
            "knowledge_included": config.include_in_context,
            "shadow": config.shadow,
            "expansion_k": config.expansion_k,
        },
    )
    return run, items


def build_semantic_snapshot(
    *,
    config: KnowledgeRetrievalConfig,
    context: SemanticContext,
) -> tuple[GenerationContextRun, list[GenerationContextItem]]:
    """Assemble a structured assertion snapshot (schema §18) for shadow recording.

    Each selected :class:`AssertionContextItem` becomes one ``assertion``-typed
    context item carrying its assertion/entity/predicate/hop provenance and the
    representative TextUnit id, so shadow comparison can measure mandatory-anchor
    recall and assertion selection against the legacy chunk channel.
    """
    context_run_id = f"CTX-{uuid4().hex[:16].upper()}"
    items: list[GenerationContextItem] = []
    for rank, item in enumerate(context.items, start=1):
        text_unit_id = next(
            (evidence.text_unit_id for evidence in item.evidence if evidence.text_unit_id),
            None,
        )
        items.append(
            GenerationContextItem(
                context_run_id=context_run_id,
                rank=rank,
                item_type="assertion",
                item_id=item.assertion_id,
                source=item.source_channel,
                score=item.retrieval_score,
                selected=True,
                assertion_id=item.assertion_id,
                text_unit_id=text_unit_id,
                entity_id=item.subject_entity_id,
                predicate=item.predicate,
                hop_count=item.hop_count,
                normalized_score=item.retrieval_score,
                mandatory=item.mandatory,
                metadata={
                    "polarity": item.polarity,
                    "modality": item.modality,
                    "explicitness": item.explicitness,
                    "object_entity_id": item.object_entity_id,
                    "object_literal": item.object_literal,
                    "evidence_count": len(item.evidence),
                },
            )
        )
    run = GenerationContextRun(
        context_run_id=context_run_id,
        stage=config.stage,
        anchor_id=context.anchor_id,
        project=config.project,
        document_version_id=context.document_version_id,
        source=SOURCE_SEMANTIC,
        metrics={**context.metrics, "shadow": config.shadow, "included": config.include_in_context},
    )
    return run, items
