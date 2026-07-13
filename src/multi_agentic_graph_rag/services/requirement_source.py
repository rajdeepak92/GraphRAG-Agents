"""Shared requirement-artifact loaders for generation stages."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalRequirementsArtifact,
    RequirementArtifact,
    RequirementInput,
    normalize_priority_label,
)


def _select_active_revision_entries[Entry](
    requirement_id: str,
    entries: list[Entry],
    *,
    status_of: Callable[[Entry], str],
    revision_of: Callable[[Entry], str],
) -> list[Entry]:
    """Return the entries of the single active revision for one requirement.

    Fails loudly when a requirement has no active revision or more than one
    distinct active revision, so a superseded (or ambiguous) requirement can never
    be silently fed into generation.
    """
    active = [entry for entry in entries if status_of(entry).strip().lower() == "active"]
    if not active:
        raise ConfigurationError(
            f"requirement {requirement_id} has no active revision; "
            "run `marag reconcile` or repair identities"
        )
    revisions = {revision_of(entry) for entry in active}
    if len(revisions) > 1:
        raise ConfigurationError(
            f"requirement {requirement_id} has multiple active revisions "
            f"{sorted(revisions)}; exactly one revision must be active"
        )
    return active


@dataclass(frozen=True)
class RequirementSource:
    """Coordinate requirement source behavior within the services boundary."""

    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    requirements: list[RequirementInput]


def load_requirement_source_local(path: Path) -> RequirementSource:
    """Load requirement source local within the authorized project and version scope.

    Args:
        path (Path): Filesystem location authorized for this operation.

    Returns:
        RequirementSource: The typed result produced by the operation.

    Raises:
        ValueError: If validated inputs or required dependencies cannot satisfy the contract.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("requirement artifact payload must be an object")
    return load_requirement_source_from_canonical_payload(data)


def load_requirement_source_from_canonical_payload(
    payload: dict[str, Any],
) -> RequirementSource:
    """Load requirement source from canonical payload within the authorized project and version
    scope.

    Args:
        payload (dict[str, Any]): Validated structured data for the operation.

    Returns:
        RequirementSource: The typed result produced by the operation.
    """
    artifact = CanonicalRequirementsArtifact.model_validate(payload)
    by_requirement: dict[str, list[Any]] = {}
    for requirement in artifact.requirements:
        by_requirement.setdefault(requirement.requirement_id, []).append(requirement)
    requirements: list[RequirementInput] = []
    for requirement_id, group in by_requirement.items():
        active = _select_active_revision_entries(
            requirement_id,
            group,
            status_of=lambda requirement: str(requirement.status),
            revision_of=lambda requirement: requirement.revision_id,
        )
        head = active[0]
        requirements.append(
            RequirementInput(
                requirement_id=requirement_id,
                revision_id=head.revision_id,
                source_req_id=head.source_req_id,
                id_generation_type=head.id_generation_type,
                confidence=head.confidence,
                requirement_text=head.requirement_text,
                requirement_type=head.requirement_type,
                priority=head.priority,
                evidence_chunk_ids=unique_strings(
                    evidence.chunk_id for requirement in active for evidence in requirement.evidence
                ),
            )
        )
    return RequirementSource(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        doc_version=artifact.doc_version,
        requirements=requirements,
    )


def load_requirement_source_from_full_payload(payload: dict[str, Any]) -> RequirementSource:
    """Load requirement source from full payload within the authorized project and version scope.

    Args:
        payload (dict[str, Any]): Validated structured data for the operation.

    Returns:
        RequirementSource: The typed result produced by the operation.
    """
    artifact = RequirementArtifact.model_validate(payload)
    by_requirement: dict[str, list[Any]] = {}
    for requirement in artifact.requirements:
        by_requirement.setdefault(requirement.requirement_id, []).append(requirement)

    requirements: list[RequirementInput] = []
    for requirement_id, group in by_requirement.items():
        # Downstream generation must run against the active revision only, never
        # the first array entry (which may be superseded).
        active = _select_active_revision_entries(
            requirement_id,
            group,
            status_of=lambda requirement: str(requirement.status),
            revision_of=lambda requirement: requirement.revision_id,
        )
        head = active[0]
        chunk_ids: list[str] = []
        for requirement in active:
            chunk_ids.append(requirement.source_trace.chunk_id)
            chunk_ids.extend(evidence.source_trace.chunk_id for evidence in requirement.evidence)
        requirements.append(
            RequirementInput(
                requirement_id=requirement_id,
                revision_id=head.revision_id,
                source_req_id=head.source_req_id,
                id_generation_type=head.id_generation_type,
                confidence=head.confidence,
                requirement_text=head.statement,
                requirement_type=head.requirement_type,
                priority=normalize_priority_label(head.priority),
                evidence_chunk_ids=unique_strings(chunk_ids),
            )
        )
    return RequirementSource(
        project=artifact.project,
        document_id=artifact.document_id,
        document_version_id=artifact.document_version_id,
        doc_version=artifact.version,
        requirements=requirements,
    )


def unique_strings(values: Iterable[object]) -> list[str]:
    """Execute the unique strings operation within its declared architectural boundary.

    Args:
        values (Iterable[object]): Ordered values processed without changing their identities.

    Returns:
        list[str]: The typed result produced by the operation.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered
