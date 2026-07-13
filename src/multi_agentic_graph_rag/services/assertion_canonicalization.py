"""Canonical assertion identity and occurrence merging for the knowledge graph.

The semantic identity (``assertion_key``) is project-scoped and built from the
resolved subject/object, normalized predicate, normalized literal, modality,
polarity, and normalized condition — ``shall X`` and ``may X`` are different
obligations and never merge. Explicitness stays out of the key: an explicit and
an inferred statement of the same claim merge into one assertion with explicit
winning. Each occurrence keeps its own evidence row with the exact quote/span.
"""

from __future__ import annotations

from multi_agentic_graph_rag.domain.errors import TraceValidationError
from multi_agentic_graph_rag.domain.identifiers import (
    assertion_evidence_id,
    assertion_id,
    assertion_key,
    assertion_lineage_key,
)
from multi_agentic_graph_rag.domain.schemas import (
    AssertionCandidate,
    AssertionEvidenceRecord,
    AssertionRecord,
    KnowledgeExtractionOutput,
)
from multi_agentic_graph_rag.services.entity_resolution import EntityResolutionResult
from multi_agentic_graph_rag.services.ontology import normalize_condition, normalize_literal


def canonicalize_assertions(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    extraction: KnowledgeExtractionOutput,
    resolution: EntityResolutionResult,
) -> tuple[list[AssertionRecord], list[AssertionEvidenceRecord]]:
    """Execute the canonicalize assertions operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        extraction (KnowledgeExtractionOutput): Extraction required by the operation's typed
                                                contract.
        resolution (EntityResolutionResult): Resolution required by the operation's typed contract.

    Returns:
        tuple[list[AssertionRecord], list[AssertionEvidenceRecord]]: The typed result produced
        by the operation.
    """
    entity_names = resolution.entity_names_by_id()
    records_by_key: dict[str, AssertionRecord] = {}
    evidence_by_id: dict[str, AssertionEvidenceRecord] = {}

    for chunk in extraction.chunks:
        for candidate in chunk.assertions:
            record = _to_record(
                project=project,
                document_id=document_id,
                document_version_id=document_version_id,
                candidate=candidate,
                resolution=resolution,
                entity_names=entity_names,
            )
            existing = records_by_key.get(record.assertion_key)
            if existing is None:
                records_by_key[record.assertion_key] = record
            else:
                records_by_key[record.assertion_key] = _merge_occurrence(existing, record)
            evidence = _to_evidence(record.assertion_id, candidate)
            evidence_by_id.setdefault(evidence.evidence_id, evidence)

    return list(records_by_key.values()), list(evidence_by_id.values())


def _to_record(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    candidate: AssertionCandidate,
    resolution: EntityResolutionResult,
    entity_names: dict[str, str],
) -> AssertionRecord:
    """Convert the value to record without mutating its source.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        candidate (AssertionCandidate): Candidate required by the operation's typed contract.
        resolution (EntityResolutionResult): Resolution required by the operation's typed contract.
        entity_names (dict[str, str]): Entity names required by the operation's typed contract.

    Returns:
        AssertionRecord: The typed result produced by the operation.

    Raises:
        TraceValidationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    subject_entity_id = resolution.resolve_reference(candidate.chunk_id, candidate.subject_name)
    if subject_entity_id is None:
        raise TraceValidationError(
            f"assertion subject {candidate.subject_name!r} in {candidate.chunk_id} "
            "does not resolve to a known entity"
        )
    object_entity_id: str | None = None
    if candidate.object_name is not None:
        object_entity_id = resolution.resolve_reference(candidate.chunk_id, candidate.object_name)
        if object_entity_id is None:
            raise TraceValidationError(
                f"assertion object {candidate.object_name!r} in {candidate.chunk_id} "
                "does not resolve to a known entity"
            )

    literal = candidate.object_literal
    normalized_literal = normalize_literal(literal) if literal is not None else None
    condition = candidate.condition
    normalized_cond = normalize_condition(condition) if condition is not None else None

    key = assertion_key(
        project=project,
        subject_entity_identifier=subject_entity_id,
        predicate=candidate.predicate,
        object_entity_identifier=object_entity_id,
        object_literal=normalized_literal,
        modality=candidate.modality,
        polarity=candidate.polarity,
        condition=normalized_cond,
    )
    lineage_key = assertion_lineage_key(
        project=project,
        subject_entity_identifier=subject_entity_id,
        predicate=candidate.predicate,
        object_entity_identifier=object_entity_id,
    )
    return AssertionRecord(
        assertion_id=assertion_id(key, document_version_id),
        assertion_key=key,
        assertion_lineage_key=lineage_key,
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        subject_entity_id=subject_entity_id,
        predicate=candidate.predicate,
        object_entity_id=object_entity_id,
        object_literal=literal,
        modality=candidate.modality,
        polarity=candidate.polarity,
        explicitness=candidate.explicitness,
        condition=condition,
        confidence=candidate.confidence,
        display_text=_display_text(
            subject_name=entity_names.get(subject_entity_id, candidate.subject_name),
            predicate=candidate.predicate,
            object_name=entity_names.get(object_entity_id or "", candidate.object_name or ""),
            object_literal=literal,
            polarity=candidate.polarity,
            condition=condition,
        ),
    )


def _merge_occurrence(existing: AssertionRecord, incoming: AssertionRecord) -> AssertionRecord:
    """Merge a repeated occurrence of the same semantic assertion.

    Modality is part of the key, so same-key occurrences already agree on it.
    The merged record is explicit if any occurrence is explicit and keeps the
    highest confidence seen.
    """
    explicitness = existing.explicitness
    if "explicit" in (existing.explicitness, incoming.explicitness):
        explicitness = "explicit"
    return existing.model_copy(
        update={
            "explicitness": explicitness,
            "confidence": max(existing.confidence, incoming.confidence),
        }
    )


def _to_evidence(
    assertion_identifier: str,
    candidate: AssertionCandidate,
) -> AssertionEvidenceRecord:
    """Convert the value to evidence without mutating its source.

    Args:
        assertion_identifier (str): Canonical assertion identifier used as a safe operational
                                    anchor.
        candidate (AssertionCandidate): Candidate required by the operation's typed contract.

    Returns:
        AssertionEvidenceRecord: The typed result produced by the operation.
    """
    trace = candidate.source_trace
    return AssertionEvidenceRecord(
        evidence_id=assertion_evidence_id(
            assertion_identifier=assertion_identifier,
            chunk_identifier=trace.chunk_id,
            quote=trace.quote,
            start_char=trace.start_char,
            end_char=trace.end_char,
        ),
        assertion_id=assertion_identifier,
        source_trace=trace.model_copy(),
    )


def _display_text(
    *,
    subject_name: str,
    predicate: str,
    object_name: str,
    object_literal: str | None,
    polarity: str,
    condition: str | None,
) -> str:
    """Execute the display text operation within its declared architectural boundary.

    Args:
        subject_name (str): Subject name required by the operation's typed contract.
        predicate (str): Predicate required by the operation's typed contract.
        object_name (str): Object name required by the operation's typed contract.
        object_literal (str | None): Object literal required by the operation's typed contract.
        polarity (str): Polarity required by the operation's typed contract.
        condition (str | None): Condition required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    target = object_name or (object_literal or "")
    parts = [subject_name, predicate, target]
    if polarity == "negative":
        parts.insert(1, "NOT")
    text = " ".join(part for part in parts if part)
    if condition:
        text = f"{text} WHEN {condition}"
    return text
