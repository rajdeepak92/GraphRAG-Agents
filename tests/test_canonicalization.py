"""Requirement, story, and scenario canonicalization tests."""

from __future__ import annotations

import pytest

from multi_agentic_graph_rag.domain.schemas import (
    Evidence,
    LLMAcceptanceCriterion,
    LLMTestScenarioCandidate,
    LLMUserStoryCandidate,
    RequirementChunkResult,
    RequirementEntityRelationshipMap,
    RequirementMapEntry,
    UserStoryStatement,
)
from multi_agentic_graph_rag.services.requirement_builder import (
    build_requirements_artifact,
)
from multi_agentic_graph_rag.services.test_scenario_builder import (
    build_test_scenarios_artifact,
)
from multi_agentic_graph_rag.services.user_story_builder import (
    build_user_stories_artifact,
)


def _entry(source_id: str, text: str) -> RequirementMapEntry:
    return RequirementMapEntry(
        requirement_ref="req_1",
        source_req_id=source_id,
        source_req_id_type="source",
        requirement_text=text,
        requirement_type="Functional Requirement",
        priority="Medium",
        constraints=[],
        evidence=[
            Evidence(
                evidence_id=f"EVD-{source_id}",
                chunk_id="CHK-1",
                quote=text,
                start_char=0,
                end_char=len(text),
            )
        ],
        entity_ids=[],
        relationship_ids=[],
        confidence=0.9,
    )


def test_conflicting_text_for_same_source_id_fails() -> None:
    requirement_map = RequirementEntityRelationshipMap(
        project="alpha",
        run_id="RUN-1",
        chunk_results=[
            RequirementChunkResult(
                chunk_id="CHK-1",
                sequence_index=0,
                status="completed",
                requirements=[_entry("BR-1", "The system shall log events.")],
                error=None,
            ),
            RequirementChunkResult(
                chunk_id="CHK-2",
                sequence_index=1,
                status="completed",
                requirements=[_entry("BR-1", "The system shall delete events.")],
                error=None,
            ),
        ],
    )
    with pytest.raises(ValueError, match="conflicting"):
        build_requirements_artifact(
            project="alpha",
            run_id="RUN-1",
            requirement_map=requirement_map,
            entities=[],
            relationships=[],
        )


def test_story_and_scenario_dedup_are_provenance_partitioned() -> None:
    criterion = LLMAcceptanceCriterion(
        title="Stored",
        given="An event exists",
        when="It is recorded",
        then="It remains available",
    )
    story = LLMUserStoryCandidate(
        story_ref="s1",
        source_req_id="BR-1",
        source_req_id_type="source",
        title="Retain events",
        priority="Medium",
        persona="Auditor",
        user_story=UserStoryStatement(
            as_a="auditor", i_want="events retained", so_that="I can investigate"
        ),
        acceptance_criteria=[criterion],
        business_rules=[],
        evidence_chunk_ids=["CHK-1"],
        supporting_entity_ids=[],
        supporting_relationship_ids=[],
        confidence=0.9,
    )
    stories = build_user_stories_artifact(
        project="alpha",
        run_id="RUN-1",
        candidates=[("REQ-1", story), ("REQ-2", story.model_copy(update={"story_ref": "s2"}))],
    )
    assert len(stories.stories) == 1
    assert stories.stories[0].requirement_ids == ["REQ-1", "REQ-2"]
    scenario = LLMTestScenarioCandidate(
        scenario_ref="t1",
        source_req_id="BR-1",
        source_req_id_type="source",
        title="Retain one event",
        description="Verify retention",
        scenario_type="Positive",
        priority="Medium",
        preconditions=["An event is available"],
        action="Record the event",
        expected_result="The event remains available",
        covered_acceptance_criterion_ids=[stories.stories[0].acceptance_criteria[0].criterion_id],
        evidence_chunk_ids=["CHK-1"],
        supporting_entity_ids=[],
        supporting_relationship_ids=[],
        confidence=0.9,
    )
    scenarios = build_test_scenarios_artifact(
        project="alpha",
        run_id="RUN-1",
        candidates=[
            ("US-1", ["REQ-1"], scenario),
            ("US-2", ["REQ-2"], scenario.model_copy(update={"scenario_ref": "t2"})),
        ],
    )
    assert len(scenarios.scenarios) == 1
    assert scenarios.scenarios[0].story_ids == ["US-1", "US-2"]

    repeated_stories = build_user_stories_artifact(
        project="alpha",
        run_id="RUN-2",
        candidates=[("REQ-3", story)],
        existing=stories,
    )
    assert repeated_stories.stories[0].story_id == stories.stories[0].story_id
    assert repeated_stories.stories[0].requirement_ids == ["REQ-3"]

    repeated_scenarios = build_test_scenarios_artifact(
        project="alpha",
        run_id="RUN-2",
        candidates=[("US-3", ["REQ-3"], scenario)],
        existing=scenarios,
    )
    assert repeated_scenarios.scenarios[0].scenario_id == scenarios.scenarios[0].scenario_id
    assert repeated_scenarios.scenarios[0].story_ids == ["US-3"]
