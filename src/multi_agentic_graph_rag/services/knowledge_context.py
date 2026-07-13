"""Assemble bounded, evidence-backed assertion context for generation stages.

This is the structured-retrieval core of the knowledge-graph rollout: instead of
handing whole chunks to a generation prompt, it selects a small set of grounded
:class:`AssertionContextItem`s — mandatory source anchors, full-text seeds, and a
degree-bounded one-hop entity expansion — scored by predicate relevance,
confidence, explicitness and hop distance, then hydrated with exact TextUnit
evidence.

Contract (Increment 2 / shadow): the assembler is pure retrieval. It does not
alter generation prompts; its :class:`SemanticContext` is recorded as a retrieval
snapshot for shadow comparison until a stage's graph-primary flag is set.
Weights here are provisional and meant to be tuned against shadow metrics
(plan §14 step 5: do not freeze final weights without evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from multi_agentic_graph_rag.domain.schemas import (
    AssertionContextItem,
    AssertionEvidenceContext,
    SemanticContext,
)

# Predicate priority profiles (plan §14 / §15). Order = descending priority; the
# assembler both bounds one-hop expansion to these predicates and rewards them in
# scoring. Anchors and full-text seeds are never predicate-filtered so recall of
# directly-grounded assertions stays complete.
STORY_PREDICATES: tuple[str, ...] = (
    "ACTS_AS",
    "COLLECTS",
    "PRESENTS",
    "PRODUCES",
    "CONSUMES",
    "EXPOSES",
    "EXPOSES_THROUGH",
    "USES",
    "USES_INTERFACE",
    "USES_PROTOCOL",
    "HAS_INTERFACE",
    "REQUIRES",
    "DEPENDS_ON",
    "ENABLES",
    "APPLIES_WHEN",
    "HAS_PRECONDITION",
    "HAS_POSTCONDITION",
    "HAS_FREQUENCY",
)
SCENARIO_PREDICATES: tuple[str, ...] = (
    "HAS_PRECONDITION",
    "HAS_POSTCONDITION",
    "APPLIES_WHEN",
    "HAS_THRESHOLD",
    "HAS_RANGE",
    "HAS_TIMEOUT",
    "HAS_FREQUENCY",
    "HAS_CAPACITY",
    "TRIGGERS",
    "FAILS_WHEN",
    "REJECTS",
    "MUST_NOT",
    "HAS_EXCEPTION",
    "GENERATES_ALARM",
    "HAS_STATE",
    "TRANSITIONS_TO",
    "REQUIRES",
    "DEPENDS_ON",
    "SECURITY_CONSTRAINT",
    "PERFORMANCE_CONSTRAINT",
    "USES_INTERFACE",
    "USES_PROTOCOL",
    "HAS_INTERFACE",
)

# Retrieval channels recorded per assertion (stable identifiers used in metrics
# and the structured snapshot rows).
CHANNEL_ANCHOR = "anchor"
CHANNEL_FULLTEXT = "fulltext"
CHANNEL_EXPANSION = "expansion"

_EXPLICIT_BONUS = 0.25
_ANCHOR_SORT_BONUS = 100.0
_MAX_EVIDENCE_PER_ITEM = 4


class AssertionQueryStore(Protocol):
    """Structural interface over the Neo4j knowledge read methods used here."""

    def fetch_anchor_assertions(
        self, evidence_chunk_ids: list[str], document_version_id: str
    ) -> list[dict[str, Any]]: ...

    def search_assertions_fulltext(
        self,
        query: str,
        document_version_id: str,
        limit: int,
        allowed_predicates: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def expand_entity_assertions(
        self,
        entity_ids: list[str],
        document_version_id: str,
        per_entity_limit: int,
        allowed_predicates: list[str] | None = None,
    ) -> list[dict[str, Any]]: ...

    def hydrate_assertion_evidence(
        self, assertion_ids: list[str], document_version_id: str
    ) -> dict[str, list[dict[str, Any]]]: ...


@dataclass(frozen=True)
class ContextAssemblyConfig:
    """Bounded retrieval limits + predicate profile for one generation stage."""

    stage: str
    predicate_priorities: tuple[str, ...]
    seed_assertions: int = 20
    seed_entities: int = 8
    per_entity_expansion: int = 5
    selected: int = 14
    max_per_entity: int = 5


def story_config() -> ContextAssemblyConfig:
    """Execute the story config operation within its declared architectural boundary.

    Returns:
        ContextAssemblyConfig: The typed result produced by the operation.
    """
    return ContextAssemblyConfig(stage="user_story", predicate_priorities=STORY_PREDICATES)


def scenario_config() -> ContextAssemblyConfig:
    """Execute the scenario config operation within its declared architectural boundary.

    Returns:
        ContextAssemblyConfig: The typed result produced by the operation.
    """
    return ContextAssemblyConfig(stage="test_scenario", predicate_priorities=SCENARIO_PREDICATES)


def config_for_stage(stage: str) -> ContextAssemblyConfig:
    """Pick the predicate profile by stage, preserving the exact stage label."""
    if "scenario" in stage.lower():
        return ContextAssemblyConfig(stage=stage, predicate_priorities=SCENARIO_PREDICATES)
    return ContextAssemblyConfig(stage=stage, predicate_priorities=STORY_PREDICATES)


def authoritative_related(
    assertions: list[AssertionContextItem],
) -> tuple[list[AssertionContextItem], list[AssertionContextItem]]:
    """Split assertions into authoritative (mandatory/hop-0) and related (hop >= 1).

    Authoritative facts are directly evidenced for the anchor and are the
    grounding for generation; related facts are adjacent context that must not
    be treated as independently sufficient (plan §14/§15).
    """
    authoritative = [item for item in assertions if item.mandatory or item.hop_count == 0]
    related = [item for item in assertions if not (item.mandatory or item.hop_count == 0)]
    return authoritative, related


def render_assertion_lines(items: list[AssertionContextItem]) -> str:
    """Render assertions as grounded one-line facts with their evidence quote."""
    if not items:
        return "(none)"
    lines: list[str] = []
    for item in items:
        obj = item.object_name or item.object_literal or ""
        subject = item.subject_name + (f" [{item.subject_type}]" if item.subject_type else "")
        qualifiers = ", ".join(
            part for part in (item.modality, item.polarity, item.condition) if part
        )
        qualifier_text = f" ({qualifiers})" if qualifiers else ""
        quote = item.evidence[0].quote if item.evidence else ""
        evidence_text = f' — evidence: "{quote}"' if quote else ""
        lines.append(f"- {subject} {item.predicate} {obj}{qualifier_text}{evidence_text}".rstrip())
    return "\n".join(lines)


@dataclass
class _Candidate:
    """Coordinate candidate behavior within the services boundary."""

    row: dict[str, Any]
    channel: str
    hop_count: int
    mandatory: bool
    search_score: float | None = field(default=None)


def assemble_semantic_context(
    store: AssertionQueryStore,
    *,
    anchor_id: str,
    query_text: str,
    document_version_id: str,
    evidence_chunk_ids: list[str],
    config: ContextAssemblyConfig,
) -> SemanticContext:
    """Retrieve, score and select bounded assertion context for one anchor."""
    allowed = list(config.predicate_priorities)

    anchors = store.fetch_anchor_assertions(evidence_chunk_ids, document_version_id)
    seeds = store.search_assertions_fulltext(
        query_text, document_version_id, config.seed_assertions
    )

    candidates: dict[str, _Candidate] = {}
    mandatory_ids: list[str] = []
    for row in anchors:
        candidate = _merge(candidates, row, CHANNEL_ANCHOR, hop_count=0, mandatory=True)
        if candidate.row["assertion_id"] not in mandatory_ids:
            mandatory_ids.append(candidate.row["assertion_id"])
    for row in seeds:
        _merge(
            candidates,
            row,
            CHANNEL_FULLTEXT,
            hop_count=0,
            mandatory=False,
            search_score=_as_float(row.get("search_score")),
        )

    seed_entities = _seed_entities(anchors, seeds, limit=config.seed_entities)
    if seed_entities:
        expanded = store.expand_entity_assertions(
            seed_entities, document_version_id, config.per_entity_expansion, allowed
        )
        for row in expanded:
            _merge(candidates, row, CHANNEL_EXPANSION, hop_count=1, mandatory=False)

    max_search = max(
        (candidate.search_score or 0.0 for candidate in candidates.values()),
        default=0.0,
    )
    scored: dict[str, float] = {
        assertion_id: _score(candidate, config, max_search)
        for assertion_id, candidate in candidates.items()
    }

    selected_ids = _select(candidates, scored, config, mandatory_ids)
    evidence_by_assertion = store.hydrate_assertion_evidence(selected_ids, document_version_id)

    ordered = sorted(
        selected_ids,
        key=lambda assertion_id: (
            not candidates[assertion_id].mandatory,
            -scored[assertion_id],
            assertion_id,
        ),
    )
    max_score = max((scored[assertion_id] for assertion_id in ordered), default=0.0)
    items = [
        _build_item(
            candidates[assertion_id],
            score=scored[assertion_id],
            normalized=_normalize(scored[assertion_id], max_score),
            evidence=evidence_by_assertion.get(assertion_id, []),
        )
        for assertion_id in ordered
    ]

    channel_counts: dict[str, int] = {}
    for candidate in candidates.values():
        channel_counts[candidate.channel] = channel_counts.get(candidate.channel, 0) + 1
    metrics: dict[str, Any] = {
        "stage": config.stage,
        "pool_size": len(candidates),
        "pool_by_channel": channel_counts,
        "seed_entity_count": len(seed_entities),
        "selected_count": len(items),
        "mandatory_anchor_count": len(mandatory_ids),
        "mandatory_recalled": sum(1 for item in items if item.mandatory),
        "predicate_profile": config.stage,
    }
    return SemanticContext(
        stage=config.stage,
        anchor_id=anchor_id,
        document_version_id=document_version_id,
        items=items,
        mandatory_anchor_ids=mandatory_ids,
        metrics=metrics,
    )


def _merge(
    candidates: dict[str, _Candidate],
    row: dict[str, Any],
    channel: str,
    *,
    hop_count: int,
    mandatory: bool,
    search_score: float | None = None,
) -> _Candidate:
    """Insert or merge a row, keeping the strongest provenance (anchor > seed)."""
    assertion_id = str(row.get("assertion_id", ""))
    existing = candidates.get(assertion_id)
    if existing is None:
        candidate = _Candidate(
            row=row,
            channel=channel,
            hop_count=hop_count,
            mandatory=mandatory,
            search_score=search_score,
        )
        candidates[assertion_id] = candidate
        return candidate
    existing.mandatory = existing.mandatory or mandatory
    existing.hop_count = min(existing.hop_count, hop_count)
    if search_score is not None:
        existing.search_score = max(existing.search_score or 0.0, search_score)
    if mandatory:
        existing.channel = CHANNEL_ANCHOR
    return existing


def _seed_entities(
    anchors: list[dict[str, Any]],
    seeds: list[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    """Ordered, deduped subject/object entity ids from anchors then seeds."""
    ordered: list[str] = []
    seen: set[str] = set()
    for row in [*anchors, *seeds]:
        for key in ("subject_entity_id", "object_entity_id"):
            entity_id = row.get(key)
            if entity_id and entity_id not in seen:
                seen.add(str(entity_id))
                ordered.append(str(entity_id))
    return ordered[:limit]


def _score(candidate: _Candidate, config: ContextAssemblyConfig, max_search: float) -> float:
    """Execute the score operation within its declared architectural boundary.

    Args:
        candidate (_Candidate): Candidate required by the operation's typed contract.
        config (ContextAssemblyConfig): Validated settings that control this operation.
        max_search (float): Max search required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    row = candidate.row
    priority = _predicate_priority(str(row.get("predicate", "")), config.predicate_priorities)
    confidence = _as_float(row.get("confidence"))
    explicit = _EXPLICIT_BONUS if str(row.get("explicitness", "")) == "explicit" else 0.0
    normalized_search = (candidate.search_score or 0.0) / max_search if max_search > 0 else 0.0
    score = 2.0 * priority + confidence + explicit + normalized_search - 0.5 * candidate.hop_count
    if candidate.mandatory:
        score += _ANCHOR_SORT_BONUS
    return score


def _predicate_priority(predicate: str, priorities: tuple[str, ...]) -> float:
    """Execute the predicate priority operation within its declared architectural boundary.

    Args:
        predicate (str): Predicate required by the operation's typed contract.
        priorities (tuple[str, ...]): Priorities required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    if not priorities:
        return 0.0
    try:
        index = priorities.index(predicate.upper())
    except ValueError:
        return 0.1
    return (len(priorities) - index) / len(priorities)


def _select(
    candidates: dict[str, _Candidate],
    scored: dict[str, float],
    config: ContextAssemblyConfig,
    mandatory_ids: list[str],
) -> list[str]:
    """All anchors (never budget-dropped) plus top non-anchors under bounds."""
    selected = list(mandatory_ids)
    per_entity: dict[str, int] = {}
    for assertion_id in selected:
        subject = str(candidates[assertion_id].row.get("subject_entity_id", ""))
        per_entity[subject] = per_entity.get(subject, 0) + 1

    remaining = max(config.selected - len(selected), 0)
    non_anchor = sorted(
        (aid for aid in candidates if aid not in set(mandatory_ids)),
        key=lambda aid: (-scored[aid], aid),
    )
    for assertion_id in non_anchor:
        if remaining <= 0:
            break
        subject = str(candidates[assertion_id].row.get("subject_entity_id", ""))
        if per_entity.get(subject, 0) >= config.max_per_entity:
            continue
        selected.append(assertion_id)
        per_entity[subject] = per_entity.get(subject, 0) + 1
        remaining -= 1
    return selected


def _build_item(
    candidate: _Candidate,
    *,
    score: float,
    normalized: float,
    evidence: list[dict[str, Any]],
) -> AssertionContextItem:
    """Build item.

    Args:
        candidate (_Candidate): Candidate required by the operation's typed contract.
        score (float): Score required by the operation's typed contract.
        normalized (float): Normalized required by the operation's typed contract.
        evidence (list[dict[str, Any]]): Evidence required by the operation's typed contract.

    Returns:
        AssertionContextItem: The typed result produced by the operation.
    """
    row = candidate.row
    return AssertionContextItem(
        assertion_id=str(row.get("assertion_id", "")),
        subject_entity_id=str(row.get("subject_entity_id", "")),
        subject_name=str(row.get("subject_name", "")),
        subject_type=str(row.get("subject_type", "")),
        predicate=str(row.get("predicate", "")),
        object_entity_id=_as_optional_str(row.get("object_entity_id")),
        object_name=_as_optional_str(row.get("object_name")),
        object_literal=_as_optional_str(row.get("object_literal")),
        modality=str(row.get("modality", "")),
        polarity=str(row.get("polarity", "")),
        explicitness=str(row.get("explicitness", "")),
        condition=_as_optional_str(row.get("condition")),
        confidence=_as_float(row.get("confidence")),
        hop_count=candidate.hop_count,
        source_channel=candidate.channel,
        mandatory=candidate.mandatory,
        retrieval_score=round(normalized, 6),
        evidence=_build_evidence(evidence),
    )


def _build_evidence(evidence: list[dict[str, Any]]) -> list[AssertionEvidenceContext]:
    """Build evidence.

    Args:
        evidence (list[dict[str, Any]]): Evidence required by the operation's typed contract.

    Returns:
        list[AssertionEvidenceContext]: The typed result produced by the operation.
    """
    items: list[AssertionEvidenceContext] = []
    for row in evidence:
        text_unit_ids = [str(value) for value in row.get("text_unit_ids", []) if value]
        targets: list[str | None] = list(text_unit_ids) if text_unit_ids else [None]
        for text_unit_id in targets:
            if len(items) >= _MAX_EVIDENCE_PER_ITEM:
                return items
            items.append(
                AssertionEvidenceContext(
                    text_unit_id=text_unit_id,
                    chunk_id=str(row.get("chunk_id", "")),
                    quote=str(row.get("quote", "")),
                    page=_as_optional_int(row.get("page")),
                    section=_as_optional_str(row.get("section")),
                    start_char=_as_optional_int(row.get("start_char")),
                    end_char=_as_optional_int(row.get("end_char")),
                )
            )
    return items


def _normalize(score: float, max_score: float) -> float:
    """Normalize normalize deterministically within the active scope.

    Args:
        score (float): Score required by the operation's typed contract.
        max_score (float): Max score required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    if max_score <= 0:
        return 0.0
    return max(0.0, min(1.0, score / max_score))


def _as_float(value: Any) -> float:
    """Execute the as float operation within its declared architectural boundary.

    Args:
        value (Any): Value required by the operation's typed contract.

    Returns:
        float: The typed result produced by the operation.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_str(value: Any) -> str | None:
    """Execute the as optional str operation within its declared architectural boundary.

    Args:
        value (Any): Value required by the operation's typed contract.

    Returns:
        str | None: The typed result produced by the operation.
    """
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _as_optional_int(value: Any) -> int | None:
    """Execute the as optional int operation within its declared architectural boundary.

    Args:
        value (Any): Value required by the operation's typed contract.

    Returns:
        int | None: The typed result produced by the operation.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
