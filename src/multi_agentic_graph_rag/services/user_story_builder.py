"""Convert validated user-story model output into permanent artifact records."""

from __future__ import annotations

import re
from collections.abc import Sequence

from multi_agentic_graph_rag.domain.identifiers import user_story_id
from multi_agentic_graph_rag.domain.schemas import (
    RequirementInput,
    UserStoryArtifact,
    UserStoryBuildResult,
    UserStoryModel,
    UserStoryProjection,
    UserStoryRecord,
    UserStoryTraceability,
)

_WHITESPACE = re.compile(r"\s+")


def build_user_story_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    generated: Sequence[tuple[RequirementInput, UserStoryModel]],
) -> UserStoryBuildResult:
    """Assign permanent ids/provenance and group by requirement.

    Pure and deterministic: identical input pairs always yield identical ids and
    coverage, so the stage is idempotent and unit-testable without any store.
    """
    stories: dict[str, UserStoryRecord] = {}
    coverage: dict[str, list[str]] = {}
    ordinals: dict[str, int] = {}

    for requirement, story in generated:
        ordinal = ordinals.get(requirement.requirement_id, 0)
        ordinals[requirement.requirement_id] = ordinal + 1
        story_id = user_story_id(
            project,
            requirement.requirement_id,
            _normalize_title(story.title),
            ordinal,
        )
        stories[story_id] = _to_record(
            project=project,
            document_id=document_id,
            document_version_id=document_version_id,
            doc_version=doc_version,
            requirement=requirement,
            story=story,
            story_id=story_id,
        )
        coverage.setdefault(requirement.requirement_id, []).append(story_id)

    artifact = project_user_story_artifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        records=stories,
        requirement_display_ids={},
        story_display_ids={},
    )
    return UserStoryBuildResult(artifact=artifact, records=stories, coverage=coverage)


def project_user_story_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    records: dict[str, UserStoryRecord],
    requirement_display_ids: dict[str, str],
    story_display_ids: dict[str, str],
) -> UserStoryArtifact:
    projections: list[UserStoryProjection] = []
    traceability: list[UserStoryTraceability] = []
    for story_id, record in records.items():
        display_id = story_display_ids.get(story_id, record.display_id or story_id)
        req_id = requirement_display_ids.get(
            record.requirement_id,
            record.requirement_display_id or record.requirement_id,
        )
        projections.append(
            UserStoryProjection(
                display_id=display_id,
                req_id=req_id,
                source_req_id=record.source_req_id,
                title=record.title,
                priority=record.priority,
                persona=record.persona,
                user_story=record.user_story,
                acceptance_criteria=list(record.acceptance_criteria),
                confidence=record.confidence,
            )
        )
        traceability.append(
            UserStoryTraceability(
                us_id=display_id,
                req_id=req_id,
                source_req_id=record.source_req_id,
                evidence_chunk_ids=list(record.evidence_chunk_ids),
            )
        )
    return UserStoryArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        stories=projections,
        traceability=traceability,
    )


def _to_record(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    requirement: RequirementInput,
    story: UserStoryModel,
    story_id: str,
) -> UserStoryRecord:
    return UserStoryRecord(
        story_id=story_id,
        requirement_id=requirement.requirement_id,
        requirement_display_id=requirement.display_id,
        requirement_revision_id=requirement.revision_id,
        source_req_id=requirement.source_req_id,
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        origin_version=doc_version,
        title=story.title,
        priority=story.priority,
        persona=story.persona,
        user_story=story.user_story,
        acceptance_criteria=list(story.acceptance_criteria),
        confidence=story.confidence,
        evidence_chunk_ids=list(requirement.evidence_chunk_ids),
    )


def _normalize_title(title: str) -> str:
    return _WHITESPACE.sub(" ", title.strip().lower())
