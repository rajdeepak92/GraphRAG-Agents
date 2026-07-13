"""Convert validated test-scenario model output into permanent artifact records."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from multi_agentic_graph_rag.domain.identifiers import test_scenario_id
from multi_agentic_graph_rag.domain.schemas import (
    GenerationTrace,
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
    traces: Mapping[str, GenerationTrace] | None = None,
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
            normalize_scenario_title(scenario.title),
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
            trace=traces.get(story.story_id) if traces else None,
        )
        coverage.setdefault(story.story_id, []).append(scenario_id)
        requirement_coverage.setdefault(story.requirement_id, []).append(scenario_id)

    artifact = project_test_scenario_artifact(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        doc_version=doc_version,
        records=scenarios,
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
    **_legacy_aliases: object,
) -> TestScenarioArtifact:
    """Project test scenario artifact through the owning storage boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        doc_version (str): Document version label within the project scope.
        records (dict[str, TestScenarioRecord]): Ordered records processed without changing their
                                                 identities.
        _legacy_aliases (object): Legacy aliases required by the operation's typed contract.

    Returns:
        TestScenarioArtifact: The typed result produced by the operation.
    """
    projections: list[TestScenarioProjection] = []
    traceability: list[TestScenarioTraceability] = []
    for scenario_id, record in records.items():
        projections.append(
            TestScenarioProjection(
                scenario_id=scenario_id,
                story_id=record.story_id,
                requirement_id=record.requirement_id,
                revision_id=record.requirement_revision_id,
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
                scenario_id=scenario_id,
                story_id=record.story_id,
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
    trace: GenerationTrace | None = None,
) -> TestScenarioRecord:
    """Convert the value to record without mutating its source.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        doc_version (str): Document version label within the project scope.
        story (UserStoryRecord): Story required by the operation's typed contract.
        scenario (TestScenarioModel): Scenario required by the operation's typed contract.
        scenario_id (str): Canonical scenario id used as a safe operational anchor.
        trace (GenerationTrace | None): Trace required by the operation's typed contract.

    Returns:
        TestScenarioRecord: The typed result produced by the operation.
    """
    trace = trace or GenerationTrace()
    return TestScenarioRecord(
        scenario_id=scenario_id,
        story_id=story.story_id,
        requirement_id=story.requirement_id,
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
        generation_context_run_id=trace.generation_context_run_id,
        retrieved_assertion_ids=list(trace.retrieved_assertion_ids),
        retrieved_chunk_ids=list(trace.retrieved_chunk_ids),
        context_mode=trace.context_mode,
    )


def normalize_scenario_title(title: str) -> str:
    """Normalize a scenario title deterministically for id derivation and identity.

    This is the single content key used both to mint ``scenario_id`` and to match
    a regenerated scenario to its prior lineage, so the two can never disagree.

    Args:
        title (str): Title required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return _WHITESPACE.sub(" ", title.strip().lower())
