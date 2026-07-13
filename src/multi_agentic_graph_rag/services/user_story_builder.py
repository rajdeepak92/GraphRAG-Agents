"""Convert validated user-story model output into permanent artifact records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from multi_agentic_graph_rag.domain.identifiers import user_story_id
from multi_agentic_graph_rag.domain.schemas import (
    GenerationTrace,
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
    traces: Mapping[str, GenerationTrace] | None = None,
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
            trace=traces.get(requirement.requirement_id) if traces else None,
        )
        coverage.setdefault(requirement.requirement_id, []).append(story_id)

    artifact = project_user_story_artifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        records=stories,
    )
    return UserStoryBuildResult(artifact=artifact, records=stories, coverage=coverage)


def project_user_story_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    records: dict[str, UserStoryRecord],
    **_legacy_aliases: object,
) -> UserStoryArtifact:
    projections: list[UserStoryProjection] = []
    traceability: list[UserStoryTraceability] = []
    for story_id, record in records.items():
        projections.append(
            UserStoryProjection(
                story_id=story_id,
                requirement_id=record.requirement_id,
                revision_id=record.requirement_revision_id,
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
                story_id=story_id,
                requirement_id=record.requirement_id,
                revision_id=record.requirement_revision_id,
                source_req_id=record.source_req_id,
                evidence_chunk_ids=list(record.evidence_chunk_ids),
                generation_context_run_id=record.generation_context_run_id,
                retrieved_assertion_ids=list(record.retrieved_assertion_ids),
                retrieved_chunk_ids=list(record.retrieved_chunk_ids),
                context_mode=record.context_mode,
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
    trace: GenerationTrace | None = None,
) -> UserStoryRecord:
    trace = trace or GenerationTrace()
    return UserStoryRecord(
        story_id=story_id,
        requirement_id=requirement.requirement_id,
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
        generation_context_run_id=trace.generation_context_run_id,
        retrieved_assertion_ids=list(trace.retrieved_assertion_ids),
        retrieved_chunk_ids=list(trace.retrieved_chunk_ids),
        context_mode=trace.context_mode,
    )


def _normalize_title(title: str) -> str:
    return _WHITESPACE.sub(" ", title.strip().lower())
