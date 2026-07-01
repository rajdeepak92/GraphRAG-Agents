"""Convert validated user-story model output into permanent artifact records."""

from __future__ import annotations

import re
from collections.abc import Sequence

from multi_agentic_graph_rag.domain.identifiers import user_story_id
from multi_agentic_graph_rag.domain.schemas import (
    RequirementInput,
    UserStoryArtifact,
    UserStoryModel,
    UserStoryRecord,
)

_WHITESPACE = re.compile(r"\s+")


def build_user_story_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    generated: Sequence[tuple[RequirementInput, UserStoryModel]],
) -> UserStoryArtifact:
    """Assign permanent ids/provenance, renumber AC/BR/TS, and group by requirement.

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

    return UserStoryArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        stories=stories,
        coverage=coverage,
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
    acceptance_criteria = [
        criterion.model_copy(update={"id": f"AC-{index:03d}"})
        for index, criterion in enumerate(story.acceptance_criteria, start=1)
    ]
    business_rules = [
        rule.model_copy(update={"id": f"BR-{index:03d}"})
        for index, rule in enumerate(story.business_rules, start=1)
    ]
    test_scenarios = [
        scenario.model_copy(update={"id": f"TS-{index:03d}"})
        for index, scenario in enumerate(story.test_scenarios, start=1)
    ]
    return UserStoryRecord(
        story_id=story_id,
        requirement_id=requirement.requirement_id,
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        title=story.title,
        epic=story.epic,
        priority=story.priority,
        persona=story.persona,
        user_story=story.user_story,
        business_value=story.business_value,
        scope=story.scope,
        acceptance_criteria=acceptance_criteria,
        business_rules=business_rules,
        test_scenarios=test_scenarios,
        definition_of_done=list(story.definition_of_done),
    )


def _normalize_title(title: str) -> str:
    return _WHITESPACE.sub(" ", title.strip().lower())
