"""Convert validated test-scenario model output into permanent artifact records."""

from __future__ import annotations

import re
from collections.abc import Sequence

from multi_agentic_graph_rag.domain.identifiers import test_scenario_id
from multi_agentic_graph_rag.domain.schemas import (
    TestScenarioArtifact,
    TestScenarioBuildResult,
    TestScenarioModel,
    TestScenarioProjection,
    TestScenarioRecord,
    TestScenarioTraceability,
    UserStoryRecord,
)

_WHITESPACE = re.compile(r"\s+")


def build_test_scenario_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    generated: Sequence[tuple[UserStoryRecord, TestScenarioModel]],
) -> TestScenarioBuildResult:
    """Assign permanent ids/provenance and group coverage by story and requirement."""
    scenarios: dict[str, TestScenarioRecord] = {}
    coverage: dict[str, list[str]] = {}
    requirement_coverage: dict[str, list[str]] = {}
    ordinals: dict[str, int] = {}

    for story, scenario in generated:
        ordinal = ordinals.get(story.story_id, 0)
        ordinals[story.story_id] = ordinal + 1
        scenario_id = test_scenario_id(
            project,
            story.story_id,
            _normalize_title(scenario.title),
            ordinal,
        )
        scenarios[scenario_id] = _to_record(
            project=project,
            document_id=document_id,
            document_version_id=document_version_id,
            doc_version=doc_version,
            story=story,
            scenario=scenario,
            scenario_id=scenario_id,
        )
        coverage.setdefault(story.story_id, []).append(scenario_id)
        requirement_coverage.setdefault(story.requirement_id, []).append(scenario_id)

    artifact = project_test_scenario_artifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        records=scenarios,
        scenario_display_ids={},
    )
    return TestScenarioBuildResult(
        artifact=artifact,
        records=scenarios,
        coverage=coverage,
        requirement_coverage=requirement_coverage,
    )


def project_test_scenario_artifact(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    records: dict[str, TestScenarioRecord],
    scenario_display_ids: dict[str, str],
) -> TestScenarioArtifact:
    projections: list[TestScenarioProjection] = []
    traceability: list[TestScenarioTraceability] = []
    for scenario_id, record in records.items():
        display_id = scenario_display_ids.get(scenario_id, record.display_id or scenario_id)
        us_id = record.story_display_id or record.story_id
        req_id = record.requirement_display_id or record.requirement_id
        projections.append(
            TestScenarioProjection(
                display_id=display_id,
                us_id=us_id,
                req_id=req_id,
                source_req_id=record.source_req_id,
                title=record.title,
                description=record.description,
                scenario_type=record.scenario_type,
                preconditions=list(record.preconditions),
                expected_result=record.expected_result,
                priority=record.priority,
                confidence=record.confidence,
            )
        )
        traceability.append(
            TestScenarioTraceability(
                ts_id=display_id,
                us_id=us_id,
                req_id=req_id,
                source_req_id=record.source_req_id,
                evidence_chunk_ids=list(record.evidence_chunk_ids),
            )
        )
    return TestScenarioArtifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        scenarios=projections,
        traceability=traceability,
    )


def _to_record(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    doc_version: str,
    story: UserStoryRecord,
    scenario: TestScenarioModel,
    scenario_id: str,
) -> TestScenarioRecord:
    return TestScenarioRecord(
        scenario_id=scenario_id,
        story_id=story.story_id,
        story_display_id=story.display_id,
        requirement_id=story.requirement_id,
        requirement_display_id=story.requirement_display_id,
        requirement_revision_id=story.requirement_revision_id,
        source_req_id=story.source_req_id,
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        origin_version=story.origin_version or doc_version,
        title=scenario.title,
        description=scenario.description,
        scenario_type=scenario.scenario_type,
        preconditions=list(scenario.preconditions),
        expected_result=scenario.expected_result,
        priority=scenario.priority,
        confidence=scenario.confidence,
        evidence_chunk_ids=list(story.evidence_chunk_ids),
    )


def _normalize_title(title: str) -> str:
    return _WHITESPACE.sub(" ", title.strip().lower())
