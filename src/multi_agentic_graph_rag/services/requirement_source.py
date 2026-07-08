"""Shared requirement-artifact loaders for generation stages."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.domain.schemas import (
    CompactRequirementArtifact,
    RequirementArtifact,
    RequirementInput,
    RequirementsCatalogArtifact,
    normalize_priority_label,
)


@dataclass(frozen=True)
class RequirementSource:
    project: str
    document_id: str
    document_version_id: str
    doc_version: str
    requirements: list[RequirementInput]


def load_requirement_source_local(path: Path) -> RequirementSource:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("artifact_schema_version") == "4.0-catalog":
        return _load_requirement_source_from_catalog(data)
    compact = CompactRequirementArtifact.model_validate(data)
    requirements: list[RequirementInput] = []
    for requirement_id, occurrences in compact.requirements.items():
        if not occurrences:
            continue
        head = occurrences[0]
        requirements.append(
            RequirementInput(
                requirement_id=requirement_id,
                requirement_text=head.requirement_text,
                requirement_type=head.requirement_type,
                priority=head.priority,
                evidence_chunk_ids=unique_strings(
                    occurrence.chunk_id for occurrence in occurrences
                ),
            )
        )
    return RequirementSource(
        project=compact.project,
        document_id=compact.document_id,
        document_version_id=compact.document_version_id,
        doc_version=compact.doc_version,
        requirements=requirements,
    )


def _load_requirement_source_from_catalog(payload: dict[str, Any]) -> RequirementSource:
    catalog = RequirementsCatalogArtifact.model_validate(payload)
    by_requirement: dict[str, list[Any]] = {}
    for entry in catalog.requirements:
        by_requirement.setdefault(entry.requirement_uid, []).append(entry)
    traceability: dict[str, list[str]] = {}
    for row in catalog.traceability:
        traceability.setdefault(row.requirement_uid, []).append(row.chunk_id)

    requirements: list[RequirementInput] = []
    for requirement_uid, entries in by_requirement.items():
        head = entries[0]
        requirements.append(
            RequirementInput(
                requirement_id=requirement_uid,
                revision_id=head.revision_id,
                display_id=head.display_id,
                source_req_id=head.source_req_id,
                id_generation_type=head.id_generation_type,
                confidence=head.confidence,
                requirement_text=head.requirement_text,
                requirement_type=head.requirement_type,
                priority=head.priority,
                evidence_chunk_ids=unique_strings(
                    [*traceability.get(requirement_uid, []), *(entry.chunk_id for entry in entries)]
                ),
            )
        )
    return RequirementSource(
        project=catalog.project,
        document_id=catalog.document_id,
        document_version_id=catalog.document_version_id,
        doc_version=catalog.doc_version,
        requirements=requirements,
    )


def load_requirement_source_from_full_payload(payload: dict[str, Any]) -> RequirementSource:
    artifact = RequirementArtifact.model_validate(payload)
    requirements: list[RequirementInput] = []
    for requirement in artifact.requirements:
        chunk_ids = [requirement.source_trace.chunk_id]
        chunk_ids.extend(evidence.source_trace.chunk_id for evidence in requirement.evidence)
        requirements.append(
            RequirementInput(
                requirement_id=requirement.requirement_id,
                revision_id=requirement.revision_id,
                display_id=requirement.display_id,
                source_req_id=requirement.source_req_id,
                id_generation_type=requirement.id_generation_type,
                confidence=requirement.confidence,
                requirement_text=requirement.statement,
                requirement_type=requirement.requirement_type,
                priority=normalize_priority_label(requirement.priority),
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
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered
