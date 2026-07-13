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
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import (
    GenerationContextItem,
    GenerationContextRun,
    SemanticContext,
)

SOURCE_KNOWLEDGE = "neo4j_knowledge"
SOURCE_SEMANTIC = "graph_semantic"

# Graph-primary decision statuses and machine-readable fallback reasons. These are
# stable identifiers emitted in structured logs and snapshot metrics so a run's
# effective generation mode (graph vs. legacy chunk) is auditable per anchor.
STATUS_SELECTED = "selected"
STATUS_FALLBACK = "fallback"

REASON_SELECTED = "selected"
REASON_KNOWLEDGE_DISABLED = "knowledge_disabled"
REASON_STAGE_NOT_PRIMARY = "stage_not_primary"
REASON_RETRIEVAL_ERROR = "retrieval_error"
REASON_NO_MANDATORY_ANCHOR = "no_mandatory_anchor"
REASON_BELOW_MIN_ASSERTIONS = "below_min_assertions"

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


@dataclass(frozen=True)
class GraphPrimaryDecision:
    """Typed, auditable outcome of the graph-primary grounding gate.

    Replaces the ambiguous ``SemanticContext | None`` return: ``context`` is the
    selected structured context to freeze/generate from (only when
    ``status == STATUS_SELECTED``), while ``reason`` names exactly why a fallback
    happened. ``context_run_id`` is the single id shared by the recorded semantic
    snapshot, the frozen checkpoint entry, and the persisted generated record.
    """

    status: str
    reason: str
    context: SemanticContext | None = None
    assertion_count: int = 0
    mandatory_anchor_count: int = 0
    min_assertions: int = 0
    context_run_id: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def selected(self) -> bool:
        return self.status == STATUS_SELECTED and self.context is not None


# Reasons that mean "the graph-primary gate was actually attempted and missed"
# (as opposed to the graph being intentionally off / shadow-only). Only these
# produce a ``graph_fallback`` warning.
_GATE_MISS_REASONS = frozenset(
    {REASON_RETRIEVAL_ERROR, REASON_NO_MANDATORY_ANCHOR, REASON_BELOW_MIN_ASSERTIONS}
)


def log_graph_primary_decision(
    logger: Any,
    *,
    stage: str,
    project: str,
    document_version_id: str,
    anchor_id: str,
    decision: GraphPrimaryDecision,
) -> None:
    """Emit a stage-specific structured log for one graph-primary decision.

    Gate misses (retrieval error / no mandatory anchor / below minimum) log at
    ``status="graph_fallback"`` with project, document version, anchor id, reason,
    counts, and the configured threshold so the effective mode is auditable. A
    disabled or shadow-only channel is expected operation and is not logged as a
    fallback.
    """
    if logger is None:
        return
    if decision.selected:
        logger.info(
            "Graph-primary context selected",
            step=f"{stage}.context_map",
            project=project,
            document_version_id=document_version_id,
            anchor_id=anchor_id,
            reason=decision.reason,
            assertion_count=decision.assertion_count,
            mandatory_anchor_count=decision.mandatory_anchor_count,
            min_assertions=decision.min_assertions,
            status="graph_primary",
        )
        return
    if decision.reason not in _GATE_MISS_REASONS:
        return
    logger.warning(
        "Graph-primary grounding gate missed; using legacy chunk retrieval",
        step=f"{stage}.context_map",
        project=project,
        document_version_id=document_version_id,
        anchor_id=anchor_id,
        reason=decision.reason,
        assertion_count=decision.assertion_count,
        mandatory_anchor_count=decision.mandatory_anchor_count,
        min_assertions=decision.min_assertions,
        status="graph_fallback",
    )


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


def require_knowledge_graph_when_primary(
    *,
    settings: AppSettings,
    neo4j: Any,
    document_version_id: str,
    primary: bool,
    stage: str,
    postgres: Any | None = None,
    project: str = "",
) -> None:
    """Block graph-primary generation unless the version's KG is ``ready``.

    Readiness is authoritative: a persisted state row proves the KG completed the
    full build path, so a partial/failed/in-progress build (which never reaches
    ``ready``) blocks fail-closed. Versions ingested before the readiness state
    machine existed have no state row; those are grandfathered on the presence of
    projected assertions so their generation is not retroactively broken.

    Legacy fallback stays valid only when the knowledge channel or the
    stage-primary flag was explicitly disabled — never as a silent reaction to a
    KG failure.
    """
    from multi_agentic_graph_rag.domain.errors import ConfigurationError
    from multi_agentic_graph_rag.services.knowledge_graph_state import (
        knowledge_graph_rebuild_command,
    )

    knowledge_graph = settings.knowledge_graph
    if not (knowledge_graph.enabled and primary):
        return

    rebuild = knowledge_graph_rebuild_command(project, document_version_id)
    state = (
        postgres.get_knowledge_graph_state(document_version_id) if postgres is not None else None
    )

    if state is not None:
        if state.status == "ready":
            return
        reason = f" (reason: {state.failure_reason})" if state.failure_reason else ""
        raise ConfigurationError(
            f"graph-primary {stage} generation is blocked: the semantic knowledge "
            f"graph for document version {document_version_id} is '{state.status}', "
            f"not 'ready'{reason}. Rebuild it with `{rebuild}`, or disable "
            "graph-primary generation to use the legacy chunk path"
        )

    # No state row: grandfather pre-readiness-machine versions on assertion presence.
    if neo4j.has_knowledge_assertions(document_version_id):
        return
    raise ConfigurationError(
        f"graph-primary {stage} generation is enabled but document version "
        f"{document_version_id} has no semantic knowledge graph; run "
        f"`{rebuild}` first (or ingest with KNOWLEDGE_GRAPH_ENABLED=true), "
        "or disable graph-primary generation to use the legacy chunk path"
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


def new_context_run_id() -> str:
    return f"CTX-{uuid4().hex[:16].upper()}"


def build_semantic_snapshot(
    *,
    config: KnowledgeRetrievalConfig,
    context: SemanticContext,
    context_run_id: str | None = None,
) -> tuple[GenerationContextRun, list[GenerationContextItem]]:
    """Assemble a structured assertion snapshot (schema §18) for shadow recording.

    Each selected :class:`AssertionContextItem` becomes one ``assertion``-typed
    context item carrying its assertion/entity/predicate/hop provenance and the
    representative TextUnit id, so shadow comparison can measure mandatory-anchor
    recall and assertion selection against the legacy chunk channel.

    ``context_run_id`` may be supplied so the recorded snapshot shares the exact id
    frozen into the checkpoint and persisted on the generated record (produced
    once, never re-derived).
    """
    context_run_id = context_run_id or new_context_run_id()
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
