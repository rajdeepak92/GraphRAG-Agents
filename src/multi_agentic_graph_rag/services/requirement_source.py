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
