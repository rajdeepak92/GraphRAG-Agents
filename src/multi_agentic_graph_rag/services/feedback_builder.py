"""Build additive feedback records for generated user-story/test-scenario artifacts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from multi_agentic_graph_rag.domain.identifiers import test_scenario_id, user_story_id
from multi_agentic_graph_rag.domain.schemas import (
    RequirementInput,
    TestScenarioArtifact,
    TestScenarioModel,
    TestScenarioRecord,
    UserStoryArtifact,
    UserStoryModel,
    UserStoryRecord,
)

_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    return _WHITESPACE.sub(" ", title.strip().lower())


def existing_user_story_titles(artifact: UserStoryArtifact, requirement_id: str) -> set[str]:
    story_ids = artifact.coverage.get(requirement_id, [])
    return {
        normalize_title(artifact.stories[story_id].title)
        for story_id in story_ids
        if story_id in artifact.stories
    }


def find_duplicate_titles(
    existing_titles: Iterable[str],
    generated_titles: Iterable[str],
) -> list[str]:
    normalized_existing = {normalize_title(title) for title in existing_titles}
    return [title for title in generated_titles if normalize_title(title) in normalized_existing]


def build_user_story_feedback_records(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    requirement: RequirementInput,
    ordinal_start: int,
    generated: Sequence[UserStoryModel],
    feedback_id_value: str,
    evidence_chunk_ids: list[str],
) -> list[UserStoryRecord]:
    records: list[UserStoryRecord] = []
    for offset, story in enumerate(generated):
        story_identifier = user_story_id(
            project,
            requirement.requirement_id,
            normalize_title(story.title),
            ordinal_start + offset,
        )
        records.append(
            UserStoryRecord(
                story_id=story_identifier,
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
                acceptance_criteria=[
                    criterion.model_copy(update={"id": f"AC-{index:03d}"})
                    for index, criterion in enumerate(story.acceptance_criteria, start=1)
                ],
                business_rules=[
                    rule.model_copy(update={"id": f"BR-{index:03d}"})
                    for index, rule in enumerate(story.business_rules, start=1)
                ],
                test_scenarios=[
                    scenario.model_copy(update={"id": f"TS-{index:03d}"})
                    for index, scenario in enumerate(story.test_scenarios, start=1)
                ],
                definition_of_done=list(story.definition_of_done),
                origin="feedback",
                feedback_id=feedback_id_value,
                evidence_chunk_ids=list(evidence_chunk_ids),
            )
        )
    return records


def merge_user_stories(
    artifact: UserStoryArtifact,
    new_records: Sequence[UserStoryRecord],
) -> UserStoryArtifact:
    stories = dict(artifact.stories)
    coverage = {
        requirement_id: list(story_ids) for requirement_id, story_ids in artifact.coverage.items()
    }
    for record in new_records:
        stories[record.story_id] = record
        coverage.setdefault(record.requirement_id, [])
        if record.story_id not in coverage[record.requirement_id]:
            coverage[record.requirement_id].append(record.story_id)
    return artifact.model_copy(
        update={
            "artifact_schema_version": "1.1-user-stories",
            "updated_at": datetime.now(UTC),
            "stories": stories,
            "coverage": coverage,
        }
    )


def user_story_delta_artifact(
    source_artifact: UserStoryArtifact,
    new_records: Sequence[UserStoryRecord],
) -> UserStoryArtifact:
    coverage: dict[str, list[str]] = {}
    stories = {record.story_id: record for record in new_records}
    for record in new_records:
        coverage.setdefault(record.requirement_id, []).append(record.story_id)
    return UserStoryArtifact(
        project=source_artifact.project,
        document_id=source_artifact.document_id,
        document_version_id=source_artifact.document_version_id,
        doc_version=source_artifact.doc_version,
        generated_at=source_artifact.generated_at,
        updated_at=datetime.now(UTC),
        stories=stories,
        coverage=coverage,
    )


def existing_test_scenario_titles(artifact: TestScenarioArtifact, story_id: str) -> set[str]:
    scenario_ids = artifact.coverage.get(story_id, [])
    return {
        normalize_title(artifact.scenarios[scenario_id].title)
        for scenario_id in scenario_ids
        if scenario_id in artifact.scenarios
    }


def build_test_scenario_feedback_records(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    story: UserStoryRecord,
    ordinal_start: int,
    generated: Sequence[TestScenarioModel],
    feedback_id_value: str,
    evidence_chunk_ids: list[str],
) -> list[TestScenarioRecord]:
    records: list[TestScenarioRecord] = []
    for offset, scenario in enumerate(generated):
        scenario_identifier = test_scenario_id(
            project,
            story.story_id,
            normalize_title(scenario.title),
            ordinal_start + offset,
        )
        records.append(
            TestScenarioRecord(
                scenario_id=scenario_identifier,
                story_id=story.story_id,
                requirement_id=story.requirement_id,
                project=project,
                document_id=document_id,
                document_version_id=document_version_id,
                doc_version=doc_version,
                title=scenario.title,
                description=scenario.description,
                scenario_type=scenario.scenario_type,
                preconditions=list(scenario.preconditions),
                expected_result=scenario.expected_result,
                priority=scenario.priority,
                confidence=scenario.confidence,
                origin="feedback",
                feedback_id=feedback_id_value,
                evidence_chunk_ids=list(evidence_chunk_ids),
            )
        )
    return records


def merge_test_scenarios(
    artifact: TestScenarioArtifact,
    new_records: Sequence[TestScenarioRecord],
) -> TestScenarioArtifact:
    scenarios = dict(artifact.scenarios)
    coverage = {
        story_id: list(scenario_ids) for story_id, scenario_ids in artifact.coverage.items()
    }
    requirement_coverage = {
        requirement_id: list(scenario_ids)
        for requirement_id, scenario_ids in artifact.requirement_coverage.items()
    }
    for record in new_records:
        scenarios[record.scenario_id] = record
        coverage.setdefault(record.story_id, [])
        if record.scenario_id not in coverage[record.story_id]:
            coverage[record.story_id].append(record.scenario_id)
        requirement_coverage.setdefault(record.requirement_id, [])
        if record.scenario_id not in requirement_coverage[record.requirement_id]:
            requirement_coverage[record.requirement_id].append(record.scenario_id)
    return artifact.model_copy(
        update={
            "artifact_schema_version": "1.1-test-scenarios",
            "updated_at": datetime.now(UTC),
            "scenarios": scenarios,
            "coverage": coverage,
            "requirement_coverage": requirement_coverage,
        }
    )


def test_scenario_delta_artifact(
    source_artifact: TestScenarioArtifact,
    new_records: Sequence[TestScenarioRecord],
) -> TestScenarioArtifact:
    coverage: dict[str, list[str]] = {}
    requirement_coverage: dict[str, list[str]] = {}
    scenarios = {record.scenario_id: record for record in new_records}
    for record in new_records:
        coverage.setdefault(record.story_id, []).append(record.scenario_id)
        requirement_coverage.setdefault(record.requirement_id, []).append(record.scenario_id)
    return TestScenarioArtifact(
        project=source_artifact.project,
        document_id=source_artifact.document_id,
        document_version_id=source_artifact.document_version_id,
        doc_version=source_artifact.doc_version,
        generated_at=source_artifact.generated_at,
        updated_at=datetime.now(UTC),
        scenarios=scenarios,
        coverage=coverage,
        requirement_coverage=requirement_coverage,
    )


test_scenario_delta_artifact.__test__ = False  # type: ignore[attr-defined]


def atomic_rewrite_json(path: Path, payload: dict[str, object]) -> None:
    """Rewrite ``path`` atomically: temp file in the same directory then os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
