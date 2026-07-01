"""Convert validated nested LLM output into ledger-aware artifact records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from multi_agentic_graph_rag.domain.identifiers import (
    canonical_fact_id,
    fact_occurrence_id,
    requirement_delta_event_id,
    requirement_evidence_id,
    requirement_lineage_id,
    requirement_revision_id,
)
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalFact,
    CompactRequirementArtifact,
    CompactRequirementOccurrence,
    RequirementArtifact,
    RequirementDeltaEvent,
    RequirementDiscoveryOutput,
    RequirementEvidence,
    RequirementRevisionSnapshot,
    SourceTrace,
    VerifiedFact,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.observability.logging import RunLogger

_WHITESPACE = re.compile(r"\s+")
_QUOTED_VALUE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
_TEMPERATURE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:degrees?|deg|°)?\s*[cf]\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_NON_KEY_CHARS = re.compile(r"[^a-z0-9{}]+")
_DOWNSTREAM_ARTIFACT_TYPES = ["user_story", "scenario", "test_case"]
_DeltaEventType = Literal["new", "duplicate", "changed", "superseded"]
_CompactPriority = Literal["High", "Medium", "Low"]
_CompactStatus = Literal["Active", "Superseded"]


@dataclass
class _RequirementAccumulator:
    requirement_id: str
    revision_id: str
    requirement_key: str
    statement: str
    normalized_statement: str
    requirement_type: str
    priority: str
    source_trace: SourceTrace
    first_ordinal: int
    fact_ids: set[str] = field(default_factory=set)
    evidence: dict[str, RequirementEvidence] = field(default_factory=dict)


def build_requirement_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    version: str,
    source_path: str,
    source_checksum: str,
    discovery: RequirementDiscoveryOutput,
    prior_revisions: Mapping[str, RequirementRevisionSnapshot] | None = None,
    logger: RunLogger | None = None,
) -> RequirementArtifact:
    if logger is not None:
        logger.debug(
            "Building requirement artifact for {document_version_id}",
            step="build_requirement_artifact",
            document_version_id=document_version_id,
            chunk_count=len(discovery.chunks),
        )
    canonical_facts: dict[str, CanonicalFact] = {}
    fact_occurrences: dict[str, VerifiedFact] = {}
    requirements: dict[tuple[str, str], _RequirementAccumulator] = {}
    requirement_ordinal = 0

    for chunk_output in discovery.chunks:
        for fact_candidate in chunk_output.facts:
            normalized_fact = normalize_fact_text(fact_candidate.text)
            canonical_id = canonical_fact_id(project, document_id, normalized_fact)
            canonical_facts.setdefault(
                canonical_id,
                CanonicalFact(
                    canonical_fact_id=canonical_id,
                    normalized_text=normalized_fact,
                    representative_text=fact_candidate.text.strip(),
                ),
            )
            fact_occurrence = VerifiedFact(
                fact_id=fact_occurrence_id(
                    project=project,
                    document_version_identifier=document_version_id,
                    chunk_identifier=fact_candidate.source_trace.chunk_id,
                    text=fact_candidate.text,
                    quote=fact_candidate.source_trace.quote,
                    start_char=fact_candidate.source_trace.start_char,
                    end_char=fact_candidate.source_trace.end_char,
                ),
                canonical_fact_id=canonical_id,
                text=fact_candidate.text.strip(),
                source_trace=fact_candidate.source_trace,
            )
            fact_occurrences.setdefault(fact_occurrence.fact_id, fact_occurrence)

            for requirement_candidate in fact_candidate.requirements:
                requirement_ordinal += 1
                requirement_key = derive_requirement_key(
                    requirement_candidate.statement,
                    requirement_candidate.requirement_key,
                )
                lineage_id = requirement_lineage_id(project, document_id, requirement_key)
                normalized_statement = normalize_requirement_statement(
                    requirement_candidate.statement
                )
                revision_id = requirement_revision_id(lineage_id, normalized_statement)
                accumulator_key = (lineage_id, revision_id)
                accumulator = requirements.get(accumulator_key)
                if accumulator is None:
                    accumulator = _RequirementAccumulator(
                        requirement_id=lineage_id,
                        revision_id=revision_id,
                        requirement_key=requirement_key,
                        statement=requirement_candidate.statement.strip(),
                        normalized_statement=normalized_statement,
                        requirement_type=requirement_candidate.requirement_type,
                        priority=requirement_candidate.priority,
                        source_trace=requirement_candidate.source_trace,
                        first_ordinal=requirement_ordinal,
                    )
                    requirements[accumulator_key] = accumulator

                accumulator.fact_ids.add(fact_occurrence.fact_id)
                evidence_id = requirement_evidence_id(
                    requirement_identifier=lineage_id,
                    revision_identifier=revision_id,
                    document_version_identifier=document_version_id,
                    chunk_identifier=requirement_candidate.source_trace.chunk_id,
                    quote=requirement_candidate.source_trace.quote,
                    start_char=requirement_candidate.source_trace.start_char,
                    end_char=requirement_candidate.source_trace.end_char,
                )
                evidence = accumulator.evidence.get(evidence_id)
                if evidence is None:
                    accumulator.evidence[evidence_id] = RequirementEvidence(
                        evidence_id=evidence_id,
                        fact_ids=[fact_occurrence.fact_id],
                        source_trace=requirement_candidate.source_trace,
                    )
                elif fact_occurrence.fact_id not in evidence.fact_ids:
                    evidence.fact_ids.append(fact_occurrence.fact_id)

    ordered_requirements = sorted(requirements.values(), key=lambda item: item.first_ordinal)
    delta_events = _build_delta_events(
        ordered_requirements,
        document_version_id,
        prior_revisions or {},
    )

    artifact = RequirementArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        version=version,
        source_path=source_path,
        source_checksum=source_checksum,
        canonical_facts=list(canonical_facts.values()),
        facts=list(fact_occurrences.values()),
        requirements=[
            _to_verified_requirement(accumulator) for accumulator in ordered_requirements
        ],
        delta_events=delta_events,
    )
    if logger is not None:
        logger.debug(
            "Built requirement artifact with {fact_count} facts and "
            "{requirement_count} requirements",
            step="build_requirement_artifact",
            document_version_id=document_version_id,
            fact_count=len(artifact.facts),
            requirement_count=len(artifact.requirements),
        )
    return artifact


def build_compact_requirement_artifact(
    artifact: RequirementArtifact,
) -> CompactRequirementArtifact:
    superseded_revision_ids = {
        event.revision_id
        for event in artifact.delta_events
        if event.event_type == "superseded" and event.revision_id
    }
    requirements: dict[str, list[CompactRequirementOccurrence]] = {}
    for requirement in artifact.requirements:
        status: _CompactStatus = (
            "Superseded" if requirement.revision_id in superseded_revision_ids else "Active"
        )
        requirements[requirement.requirement_id] = [
            CompactRequirementOccurrence(
                chunk_id=requirement.source_trace.chunk_id,
                fact_id=fact_id,
                requirement_text=requirement.statement,
                requirement_type=requirement.requirement_type,
                priority=_compact_priority(requirement.priority),
                status=status,
                doc_version=artifact.version,
            )
            for fact_id in requirement.fact_ids
        ]
    return CompactRequirementArtifact(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        doc_version=artifact.version,
        generated_at=artifact.generated_at,
        requirements=requirements,
    )


def _compact_priority(priority: str) -> _CompactPriority:
    normalized = priority.strip().lower()
    if normalized == "high":
        return "High"
    if normalized == "medium":
        return "Medium"
    if normalized == "low":
        return "Low"
    raise ValueError(f"Unsupported compact requirement priority: {priority!r}")


def normalize_fact_text(text: str) -> str:
    return _WHITESPACE.sub(" ", text.strip().lower())


def normalize_requirement_statement(statement: str) -> str:
    return _WHITESPACE.sub(" ", statement.strip().lower())


def derive_requirement_key(statement: str, provided_key: str | None) -> str:
    source = provided_key.strip() if provided_key and provided_key.strip() else statement
    key = source.lower()
    key = _QUOTED_VALUE.sub("{value}", key)
    key = _TEMPERATURE.sub("{temperature}", key)
    key = _NUMBER.sub("{number}", key)
    key = _NON_KEY_CHARS.sub(" ", key)
    return _WHITESPACE.sub(" ", key).strip() or normalize_requirement_statement(statement)


def _to_verified_requirement(accumulator: _RequirementAccumulator) -> VerifiedRequirement:
    fact_ids = sorted(accumulator.fact_ids)
    return VerifiedRequirement(
        requirement_id=accumulator.requirement_id,
        revision_id=accumulator.revision_id,
        requirement_key=accumulator.requirement_key,
        statement=accumulator.statement,
        normalized_statement=accumulator.normalized_statement,
        requirement_type=accumulator.requirement_type,
        priority=accumulator.priority,
        fact_ids=fact_ids,
        source_trace=accumulator.source_trace,
        evidence=list(accumulator.evidence.values()),
    )


def _build_delta_events(
    requirements: list[_RequirementAccumulator],
    document_version_id: str,
    prior_revisions: Mapping[str, RequirementRevisionSnapshot],
) -> list[RequirementDeltaEvent]:
    events: list[RequirementDeltaEvent] = []
    active_revisions = dict(prior_revisions)
    for requirement in requirements:
        active = active_revisions.get(requirement.requirement_id)
        evidence_ids = list(requirement.evidence)
        if active is None:
            events.append(
                _delta_event(
                    event_type="new",
                    requirement=requirement,
                    document_version_id=document_version_id,
                    evidence_ids=evidence_ids,
                )
            )
        elif active.revision_id == requirement.revision_id:
            events.append(
                _delta_event(
                    event_type="duplicate",
                    requirement=requirement,
                    document_version_id=document_version_id,
                    evidence_ids=evidence_ids,
                    previous_revision_id=active.revision_id,
                )
            )
        else:
            events.append(
                _delta_event(
                    event_type="changed",
                    requirement=requirement,
                    document_version_id=document_version_id,
                    evidence_ids=evidence_ids,
                    previous_revision_id=active.revision_id,
                )
            )
            events.append(
                _delta_event(
                    event_type="superseded",
                    requirement=requirement,
                    document_version_id=document_version_id,
                    evidence_ids=[],
                    revision_id=active.revision_id,
                    superseded_by_revision_id=requirement.revision_id,
                )
            )

        active_revisions[requirement.requirement_id] = RequirementRevisionSnapshot(
            requirement_id=requirement.requirement_id,
            revision_id=requirement.revision_id,
            statement=requirement.statement,
            normalized_statement=requirement.normalized_statement,
        )
    return events


def _delta_event(
    *,
    event_type: _DeltaEventType,
    requirement: _RequirementAccumulator,
    document_version_id: str,
    evidence_ids: list[str],
    revision_id: str | None = None,
    previous_revision_id: str | None = None,
    superseded_by_revision_id: str | None = None,
) -> RequirementDeltaEvent:
    target_revision_id = revision_id or requirement.revision_id
    return RequirementDeltaEvent(
        event_id=requirement_delta_event_id(
            event_type=event_type,
            requirement_identifier=requirement.requirement_id,
            revision_identifier=target_revision_id,
            previous_revision_identifier=previous_revision_id or superseded_by_revision_id,
            document_version_identifier=document_version_id,
        ),
        event_type=event_type,
        requirement_id=requirement.requirement_id,
        revision_id=target_revision_id,
        previous_revision_id=previous_revision_id,
        superseded_by_revision_id=superseded_by_revision_id,
        document_version_id=document_version_id,
        evidence_ids=evidence_ids,
        impacted_artifact_types=[] if event_type == "duplicate" else _DOWNSTREAM_ARTIFACT_TYPES,
    )
