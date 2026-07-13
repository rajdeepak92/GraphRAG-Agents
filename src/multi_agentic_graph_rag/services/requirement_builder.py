"""Convert validated nested LLM output into ledger-aware artifact records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from multi_agentic_graph_rag.domain.errors import IngestionError
from multi_agentic_graph_rag.domain.identifiers import (
    canonical_fact_id,
    fact_occurrence_id,
    new_requirement_evidence_id,
    new_requirement_id,
    new_requirement_revision_id,
    requirement_delta_event_id,
    requirement_evidence_occurrence_key,
)
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalFact,
    CanonicalRequirement,
    CanonicalRequirementEvidence,
    CanonicalRequirementsArtifact,
    RequirementArtifact,
    RequirementDeltaEvent,
    RequirementDiscoveryOutput,
    RequirementEvidence,
    RequirementIdentityResolutionRecord,
    RequirementRevisionSnapshot,
    SourceTrace,
    VerifiedFact,
    VerifiedRequirement,
    normalize_source_req_id,
)
from multi_agentic_graph_rag.observability.logging import RunLogger
from multi_agentic_graph_rag.services.requirement_identity_resolver import (
    requirement_lineage_signature,
    resolve_requirement_identity,
    structured_requirement_signature,
)
from multi_agentic_graph_rag.services.requirement_memory import MemoryEntry, RequirementMemory

_WHITESPACE = re.compile(r"\s+")
_QUOTED_VALUE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
_TEMPERATURE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:degrees?|deg|°)?\s*[cf]\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_NON_KEY_CHARS = re.compile(r"[^a-z0-9{}]+")
_DOWNSTREAM_ARTIFACT_TYPES = ["user_story", "scenario", "test_case"]
_DeltaEventType = Literal["new", "duplicate", "changed", "superseded"]
_CompactPriority = Literal["High", "Medium", "Low"]


@dataclass
class _RequirementAccumulator:
    """Coordinate requirement accumulator behavior within the services boundary."""

    requirement_id: str
    revision_id: str
    requirement_key: str
    source_req_id: str | None
    id_generation_type: Literal["source", "generated"]
    confidence: float
    statement: str
    normalized_statement: str
    requirement_type: str
    priority: str
    semantic_signature: str
    actor: str
    modality: str
    action: str
    object_text: str
    condition: str
    polarity: Literal["positive", "negative"]
    requirement_family: str
    entity_discriminators: list[str]
    mutable_parameters: list[str]
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
    requirement_memory: RequirementMemory | None = None,
    logger: RunLogger | None = None,
) -> RequirementArtifact:
    """Build requirement artifact.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        version (str): Document version label within the project scope.
        source_path (str): Filesystem location authorized for this operation.
        source_checksum (str): Source checksum required by the operation's typed contract.
        discovery (RequirementDiscoveryOutput): Discovery required by the operation's typed
                                                contract.
        prior_revisions (Mapping[str, RequirementRevisionSnapshot] | None): Prior revisions required
                                                                            by the operation's typed
                                                                            contract.
        requirement_memory (RequirementMemory | None): Requirement memory required by the
                                                       operation's typed contract.
        logger (RunLogger | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        RequirementArtifact: The typed result produced by the operation.

    Side Effects:
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
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
    allocated_lineages: dict[str, str] = {}
    allocated_revisions: dict[tuple[str, str], str] = {}
    evidence_occurrences: dict[tuple[object, ...], str] = {}
    identity_resolutions: list[RequirementIdentityResolutionRecord] = []
    identity_total = sum(
        len(fact.requirements) for chunk_output in discovery.chunks for fact in chunk_output.facts
    )
    prior_by_signature = {
        (
            snapshot.semantic_signature
            or requirement_lineage_signature(snapshot.statement, snapshot.requirement_type)
        ): snapshot
        for snapshot in (prior_revisions or {}).values()
        if snapshot.requirement_type
    }
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
                # ``requirement_key`` is retained only as a non-authoritative hint;
                # the permanent lineage is decided by the deterministic identity
                # resolver from a discriminator-preserving semantic signature.
                requirement_key = derive_requirement_key(
                    requirement_candidate.statement,
                    requirement_candidate.requirement_key,
                )
                normalized_statement = normalize_requirement_statement(
                    requirement_candidate.statement
                )
                source_req_id = _validated_source_req_id(
                    requirement_candidate.source_req_id,
                    requirement_candidate.source_trace,
                )
                identity = resolve_requirement_identity(
                    project=project,
                    document_id=document_id,
                    statement=requirement_candidate.statement,
                    requirement_type=requirement_candidate.requirement_type,
                    normalized_statement=normalized_statement,
                )
                signature = structured_requirement_signature(
                    statement=requirement_candidate.statement,
                    requirement_type=requirement_candidate.requirement_type,
                    actor=requirement_candidate.actor,
                    modality=requirement_candidate.modality,
                    action=requirement_candidate.action,
                    object_text=requirement_candidate.object,
                    condition=requirement_candidate.condition,
                    polarity=requirement_candidate.polarity,
                    requirement_family=requirement_candidate.requirement_family,
                    entity_discriminators=requirement_candidate.entity_discriminators,
                    mutable_parameters=requirement_candidate.mutable_parameters,
                )
                prior = prior_by_signature.get(signature)
                reconciliation = (
                    requirement_memory.reconcile(
                        statement=requirement_candidate.statement,
                        requirement_type=requirement_candidate.requirement_type,
                        normalized_statement=normalized_statement,
                        source_req_id=source_req_id,
                        identity_index=requirement_ordinal,
                        identity_total=identity_total,
                    )
                    if requirement_memory is not None
                    else None
                )
                if (
                    reconciliation is not None
                    and reconciliation.decision == "AMBIGUOUS"
                    and logger is not None
                ):
                    logger.warning(
                        "Requirement identity was ambiguous; allocated a distinct lineage",
                        step="resolve_requirement_identity",
                        candidate_ids=list(reconciliation.candidate_ids),
                        identity_index=requirement_ordinal,
                        identity_total=identity_total,
                        reason=reconciliation.reasons[0],
                    )
                if reconciliation is not None and reconciliation.requirement_id is not None:
                    lineage_id = reconciliation.requirement_id
                else:
                    # Compatibility bridge: existing hash-era rows can still be reused
                    # until the explicit repair migration rewrites them to UUIDv7.
                    if prior is None:
                        prior = (prior_revisions or {}).get(identity.requirement_id)
                    lineage_id = (
                        prior.requirement_id
                        if prior is not None
                        else allocated_lineages.setdefault(signature, new_requirement_id())
                    )
                revision_key = (lineage_id, normalized_statement)
                revision_id = (
                    reconciliation.revision_id
                    if reconciliation is not None and reconciliation.revision_id is not None
                    else prior.revision_id
                    if prior is not None and prior.normalized_statement == normalized_statement
                    else allocated_revisions.setdefault(revision_key, new_requirement_revision_id())
                )
                accumulator_key = (lineage_id, revision_id)
                accumulator = requirements.get(accumulator_key)
                if accumulator is None:
                    accumulator = _RequirementAccumulator(
                        requirement_id=lineage_id,
                        revision_id=revision_id,
                        requirement_key=requirement_key,
                        source_req_id=source_req_id,
                        id_generation_type="source" if source_req_id else "generated",
                        confidence=requirement_candidate.confidence,
                        statement=requirement_candidate.statement.strip(),
                        normalized_statement=normalized_statement,
                        requirement_type=requirement_candidate.requirement_type,
                        priority=requirement_candidate.priority,
                        semantic_signature=signature,
                        actor=requirement_candidate.actor,
                        modality=requirement_candidate.modality,
                        action=requirement_candidate.action,
                        object_text=requirement_candidate.object,
                        condition=requirement_candidate.condition,
                        polarity=requirement_candidate.polarity,
                        requirement_family=requirement_candidate.requirement_family,
                        entity_discriminators=list(requirement_candidate.entity_discriminators),
                        mutable_parameters=list(requirement_candidate.mutable_parameters),
                        source_trace=requirement_candidate.source_trace,
                        first_ordinal=requirement_ordinal,
                    )
                    requirements[accumulator_key] = accumulator
                    if requirement_memory is not None:
                        requirement_memory.add(
                            MemoryEntry(
                                requirement_id=lineage_id,
                                revision_id=revision_id,
                                statement=accumulator.statement,
                                normalized_statement=normalized_statement,
                                requirement_type=accumulator.requirement_type,
                                source_req_id=source_req_id,
                                signature=signature,
                                semantic_recall_enabled=False,
                            )
                        )
                else:
                    if accumulator.source_req_id is None and source_req_id is not None:
                        accumulator.source_req_id = source_req_id
                        accumulator.id_generation_type = "source"
                    accumulator.confidence = max(
                        accumulator.confidence,
                        requirement_candidate.confidence,
                    )

                if reconciliation is not None:
                    decision = reconciliation.decision
                    reasons = reconciliation.reasons
                    candidate_ids = list(reconciliation.candidate_ids)
                    candidate_scores = dict(reconciliation.candidate_scores)
                    reranker_order = list(reconciliation.reranker_order)
                    judge_result = reconciliation.judge_result
                elif prior is not None and prior.normalized_statement == normalized_statement:
                    decision = "EXACT"
                    reasons = ("prior_exact_normalized_statement",)
                    candidate_ids = [prior.requirement_id]
                    candidate_scores = {}
                    reranker_order = []
                    judge_result = None
                elif prior is not None:
                    decision = "SAME_LINEAGE_REVISION"
                    reasons = ("prior_structured_signature",)
                    candidate_ids = [prior.requirement_id]
                    candidate_scores = {}
                    reranker_order = []
                    judge_result = None
                else:
                    decision = "DISTINCT"
                    reasons = ("no_compatible_canonical_candidate",)
                    candidate_ids = []
                    candidate_scores = {}
                    reranker_order = []
                    judge_result = None
                identity_resolutions.append(
                    RequirementIdentityResolutionRecord(
                        incoming_fingerprint=signature,
                        document_version_id=document_version_id,
                        chunk_id=requirement_candidate.source_trace.chunk_id,
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        reranker_order=reranker_order,
                        deterministic_rule=reasons[0],
                        judge_result=judge_result,
                        decision=decision,
                        reason=";".join(reasons),
                        requirement_id=lineage_id,
                        revision_id=revision_id,
                    )
                )

                accumulator.fact_ids.add(fact_occurrence.fact_id)
                occurrence_lookup_key = requirement_evidence_occurrence_key(
                    document_version_identifier=document_version_id,
                    chunk_identifier=requirement_candidate.source_trace.chunk_id,
                    quote=requirement_candidate.source_trace.quote,
                    start_char=requirement_candidate.source_trace.start_char,
                    end_char=requirement_candidate.source_trace.end_char,
                )
                occurrence_key = (lineage_id, revision_id, occurrence_lookup_key)
                evidence_id = evidence_occurrences.setdefault(
                    occurrence_key,
                    (prior.evidence_ids.get(occurrence_lookup_key) if prior is not None else None)
                    or new_requirement_evidence_id(),
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

    verified_requirements = [
        _to_verified_requirement(accumulator) for accumulator in ordered_requirements
    ]
    superseded_revision_ids = {
        event.revision_id
        for event in delta_events
        if event.event_type == "superseded" and event.revision_id
    }
    verified_requirements = [
        requirement.model_copy(
            update={
                "status": (
                    "superseded" if requirement.revision_id in superseded_revision_ids else "active"
                )
            }
        )
        for requirement in verified_requirements
    ]
    validate_requirement_revision_lifecycle(verified_requirements)

    artifact = RequirementArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        version=version,
        source_path=source_path,
        source_checksum=source_checksum,
        canonical_facts=list(canonical_facts.values()),
        facts=list(fact_occurrences.values()),
        requirements=verified_requirements,
        delta_events=delta_events,
        identity_resolutions=identity_resolutions,
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


def build_canonical_requirements_artifact(
    artifact: RequirementArtifact,
) -> CanonicalRequirementsArtifact:
    """Project the audit ledger to the canonical, non-duplicating public contract."""
    requirements = [
        CanonicalRequirement(
            requirement_id=requirement.requirement_id,
            revision_id=requirement.revision_id,
            source_req_id=requirement.source_req_id,
            id_generation_type=requirement.id_generation_type,
            confidence=requirement.confidence,
            requirement_text=requirement.statement,
            semantic_signature=requirement.semantic_signature,
            requirement_type=requirement.requirement_type,
            actor=requirement.actor,
            modality=requirement.modality,
            action=requirement.action,
            object=requirement.object,
            condition=requirement.condition,
            polarity=requirement.polarity,
            requirement_family=requirement.requirement_family,
            entity_discriminators=list(requirement.entity_discriminators),
            mutable_parameters=list(requirement.mutable_parameters),
            priority=_compact_priority(requirement.priority),
            status="Superseded" if requirement.status == "superseded" else "Active",
            evidence=[
                CanonicalRequirementEvidence(
                    evidence_id=evidence.evidence_id,
                    document_version_id=artifact.document_version_id,
                    chunk_id=evidence.source_trace.chunk_id,
                    fact_ids=evidence.fact_ids,
                    quote=evidence.source_trace.quote,
                    start_char=evidence.source_trace.start_char,
                    end_char=evidence.source_trace.end_char,
                    page=evidence.source_trace.page,
                    section=evidence.source_trace.section,
                    source_path=artifact.source_path,
                )
                for evidence in requirement.evidence
            ],
        )
        for requirement in artifact.requirements
    ]
    return CanonicalRequirementsArtifact(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        doc_version=artifact.version,
        generated_at=artifact.generated_at,
        requirements=requirements,
    )


def validate_requirement_revision_lifecycle(
    requirements: list[VerifiedRequirement],
) -> dict[str, VerifiedRequirement]:
    """Validate and group active revisions at an artifact boundary.

    Args:
        requirements (list[VerifiedRequirement]): Revisions represented by one internal artifact.

    Returns:
        dict[str, VerifiedRequirement]: The sole active revision for each represented lineage.

    Raises:
        IngestionError: If a represented lineage has zero or multiple active revisions.
    """
    grouped: dict[str, list[VerifiedRequirement]] = {}
    for requirement in requirements:
        grouped.setdefault(requirement.requirement_id, []).append(requirement)

    active_by_requirement: dict[str, VerifiedRequirement] = {}
    for requirement_id, revisions in grouped.items():
        active = [revision for revision in revisions if revision.status == "active"]
        if len(active) != 1:
            raise IngestionError(
                "requirement revision lifecycle invariant failed: each represented "
                "lineage must contain exactly one active revision"
            )
        active_by_requirement[requirement_id] = active[0]
    return active_by_requirement


def _compact_priority(priority: str) -> _CompactPriority:
    """Execute the compact priority operation within its declared architectural boundary.

    Args:
        priority (str): Priority required by the operation's typed contract.

    Returns:
        _CompactPriority: The typed result produced by the operation.

    Raises:
        ValueError: If validated inputs or required dependencies cannot satisfy the contract.
    """
    normalized = priority.strip().lower()
    if normalized == "high":
        return "High"
    if normalized == "medium":
        return "Medium"
    if normalized == "low":
        return "Low"
    raise ValueError(f"Unsupported compact requirement priority: {priority!r}")


def normalize_fact_text(text: str) -> str:
    """Normalize fact text deterministically within the active scope.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
    return _WHITESPACE.sub(" ", text.strip().lower())


def normalize_requirement_statement(statement: str) -> str:
    """Normalize requirement statement deterministically within the active scope.

    Args:
        statement (str): Statement required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return _WHITESPACE.sub(" ", statement.strip().lower())


def derive_requirement_key(statement: str, provided_key: str | None) -> str:
    """Derive requirement key deterministically within the active scope.

    Args:
        statement (str): Statement required by the operation's typed contract.
        provided_key (str | None): Provided key required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    source = provided_key.strip() if provided_key and provided_key.strip() else statement
    key = source.lower()
    key = _QUOTED_VALUE.sub("{value}", key)
    key = _TEMPERATURE.sub("{temperature}", key)
    key = _NUMBER.sub("{number}", key)
    key = _NON_KEY_CHARS.sub(" ", key)
    return _WHITESPACE.sub(" ", key).strip() or normalize_requirement_statement(statement)


def _to_verified_requirement(accumulator: _RequirementAccumulator) -> VerifiedRequirement:
    """Convert the value to verified requirement without mutating its source.

    Args:
        accumulator (_RequirementAccumulator): Accumulator required by the operation's typed
                                               contract.

    Returns:
        VerifiedRequirement: The typed result produced by the operation.
    """
    fact_ids = sorted(accumulator.fact_ids)
    return VerifiedRequirement(
        requirement_id=accumulator.requirement_id,
        revision_id=accumulator.revision_id,
        requirement_key=accumulator.requirement_key,
        source_req_id=accumulator.source_req_id,
        id_generation_type=accumulator.id_generation_type,
        confidence=accumulator.confidence,
        statement=accumulator.statement,
        normalized_statement=accumulator.normalized_statement,
        semantic_signature=accumulator.semantic_signature,
        requirement_type=accumulator.requirement_type,
        actor=accumulator.actor,
        modality=accumulator.modality,
        action=accumulator.action,
        object=accumulator.object_text,
        condition=accumulator.condition,
        polarity=accumulator.polarity,
        requirement_family=accumulator.requirement_family,
        entity_discriminators=list(accumulator.entity_discriminators),
        mutable_parameters=list(accumulator.mutable_parameters),
        priority=accumulator.priority,
        fact_ids=fact_ids,
        source_trace=accumulator.source_trace,
        evidence=list(accumulator.evidence.values()),
    )


def _validated_source_req_id(source_req_id: str | None, trace: SourceTrace) -> str | None:
    """Execute the validated source req id operation within its declared architectural boundary.

    Args:
        source_req_id (str | None): Canonical source req id used as a safe operational anchor.
        trace (SourceTrace): Trace required by the operation's typed contract.

    Returns:
        str | None: The typed result produced by the operation.
    """
    normalized = normalize_source_req_id(source_req_id)
    if normalized is None:
        return None
    quote_source_id = normalize_source_req_id(trace.quote)
    return normalized if quote_source_id == normalized else None


def _fact_chunks_for_requirement(requirement: VerifiedRequirement) -> dict[str, str]:
    """Execute the fact chunks for requirement operation within its declared architectural boundary.

    Args:
        requirement (VerifiedRequirement): Requirement required by the operation's typed contract.

    Returns:
        dict[str, str]: The typed result produced by the operation.
    """
    chunks: dict[str, str] = {}
    for evidence in requirement.evidence:
        for fact_id in evidence.fact_ids:
            chunks.setdefault(fact_id, evidence.source_trace.chunk_id)
    for fact_id in requirement.fact_ids:
        chunks.setdefault(fact_id, requirement.source_trace.chunk_id)
    return chunks


def _build_delta_events(
    requirements: list[_RequirementAccumulator],
    document_version_id: str,
    prior_revisions: Mapping[str, RequirementRevisionSnapshot],
) -> list[RequirementDeltaEvent]:
    """Build delta events.

    Args:
        requirements (list[_RequirementAccumulator]): Ordered requirements processed without
                                                      changing their identities.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        prior_revisions (Mapping[str, RequirementRevisionSnapshot]): Prior revisions required by the
                                                                     operation's typed contract.

    Returns:
        list[RequirementDeltaEvent]: The typed result produced by the operation.
    """
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
    """Execute the delta event operation within its declared architectural boundary.

    Args:
        event_type (_DeltaEventType): Event type required by the operation's typed contract.
        requirement (_RequirementAccumulator): Requirement required by the operation's typed
                                               contract.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        evidence_ids (list[str]): Evidence ids required by the operation's typed contract.
        revision_id (str | None): Canonical revision id used as a safe operational anchor.
        previous_revision_id (str | None): Canonical previous revision id used as a safe operational
                                           anchor.
        superseded_by_revision_id (str | None): Canonical superseded by revision id used as a safe
                                                operational anchor.

    Returns:
        RequirementDeltaEvent: The typed result produced by the operation.
    """
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
