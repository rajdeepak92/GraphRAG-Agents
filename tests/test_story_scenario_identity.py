"""Acceptance tests: story/scenario identity is content-based, never ordinal.

These lock architectural decision #7 (never match story/scenario identities by
output ordinal). They prove that ``_preserve_existing_story_ids`` /
``_preserve_existing_scenario_ids`` reuse a prior permanent id only on a content
match (same parent + same normalized title) and never reassign an id to
unrelated content because of its position in the regenerated batch.
"""

from __future__ import annotations

import unittest

from multi_agentic_graph_rag.domain.schemas import (
    TestScenarioBuildResult,
    TestScenarioRecord,
    UserStoryBuildResult,
    UserStoryRecord,
    UserStoryStatement,
)
from multi_agentic_graph_rag.services.story_scenario_identity import (
    StoryScenarioIdentityResolver,
)
from multi_agentic_graph_rag.services.test_scenario_builder import project_test_scenario_artifact
from multi_agentic_graph_rag.services.user_story_builder import project_user_story_artifact
from multi_agentic_graph_rag.workflows.test_scenario_graph import _preserve_existing_scenario_ids
from multi_agentic_graph_rag.workflows.user_story_graph import _preserve_existing_story_ids


class _ConstantEmbedder:
    """Returns identical unit vectors so every candidate clears the recall floor,
    leaving the entailment judge as the sole decider."""

    provider_name = "fake"

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in documents]


class _KeywordJudge:
    """Bidirectional entailment stand-in: two texts are the 'same' iff they share
    the marker keyword. Deterministic; no real model call."""

    def __init__(self, keyword: str) -> None:
        self.keyword = keyword.lower()

    def equivalent(self, premise: str, hypothesis: str) -> bool:
        return self.keyword in premise.lower() and self.keyword in hypothesis.lower()


class StoryIdentityTests(unittest.TestCase):
    # Inserting a NEW story ahead of existing ones must NOT steal their ids.
    def test_inserted_story_does_not_steal_prior_ids(self) -> None:
        existing = [
            _story_record("US-A", "Alpha requirement behaviour"),
            _story_record("US-B", "Beta requirement behaviour"),
        ]
        # Regeneration emits a brand-new story first, then Alpha and Beta.
        new = _story_build(
            [
                _story_record("US-FRESH-G", "Gamma requirement behaviour"),
                _story_record("US-FRESH-A", "Alpha requirement behaviour"),
                _story_record("US-FRESH-B", "Beta requirement behaviour"),
            ]
        )

        result = _preserve_existing_story_ids(new, existing)
        by_title = {r.title: sid for sid, r in result.records.items()}

        self.assertEqual(by_title["Alpha requirement behaviour"], "US-A")
        self.assertEqual(by_title["Beta requirement behaviour"], "US-B")
        # New content keeps its freshly minted id — never a prior story's id.
        self.assertEqual(by_title["Gamma requirement behaviour"], "US-FRESH-G")
        self.assertEqual(result.coverage["REQ-1"].count("US-A"), 1)

    # Pure reordering preserves every id by content.
    def test_reordering_preserves_ids(self) -> None:
        existing = [
            _story_record("US-A", "Alpha requirement behaviour"),
            _story_record("US-B", "Beta requirement behaviour"),
        ]
        new = _story_build(
            [
                _story_record("US-FRESH-B", "Beta requirement behaviour"),
                _story_record("US-FRESH-A", "Alpha requirement behaviour"),
            ]
        )
        result = _preserve_existing_story_ids(new, existing)
        by_title = {r.title: sid for sid, r in result.records.items()}
        self.assertEqual(by_title["Alpha requirement behaviour"], "US-A")
        self.assertEqual(by_title["Beta requirement behaviour"], "US-B")

    # Genuinely new content (no prior story) keeps its minted id.
    def test_new_story_keeps_minted_id(self) -> None:
        new = _story_build([_story_record("US-FRESH-X", "Novel behaviour")])
        result = _preserve_existing_story_ids(new, [])
        self.assertEqual(list(result.records), ["US-FRESH-X"])

    # A reworded title with no exact match is reused via LLM-proposed entailment.
    def test_reworded_story_reuses_id_via_semantic_match(self) -> None:
        existing = [_story_record("US-A", "Alpha login capability")]
        new = _story_build([_story_record("US-FRESH", "Sign in for alpha users")])
        resolver = StoryScenarioIdentityResolver(
            embedder=_ConstantEmbedder(),
            judge=_KeywordJudge("alpha"),
        )
        result = _preserve_existing_story_ids(new, existing, resolver)
        self.assertEqual(list(result.records), ["US-A"])

    # A prior story the regeneration drops is preserved as 'outdated', never lost.
    def test_dropped_prior_story_marked_outdated(self) -> None:
        existing = [
            _story_record("US-A", "Alpha requirement behaviour"),
            _story_record("US-B", "Beta requirement behaviour"),
        ]
        # Only Alpha is regenerated; Beta disappears from the batch.
        new = _story_build([_story_record("US-FRESH-A", "Alpha requirement behaviour")])
        result = _preserve_existing_story_ids(new, existing)
        self.assertIn("US-B", result.records)
        self.assertEqual(result.records["US-B"].status, "outdated")
        self.assertEqual(result.records["US-A"].status, "active")
        # Outdated history is not counted as active coverage.
        self.assertNotIn("US-B", result.coverage.get("REQ-1", []))

    # A prior under a parent NOT in this batch (resume / subset run) must stay
    # active — "absent from this batch" is not "removed".
    def test_out_of_batch_parent_is_not_retired(self) -> None:
        existing = [
            _story_record("US-R1", "Alpha", requirement_id="REQ-1"),
            _story_record("US-R2", "Beta", requirement_id="REQ-2"),
        ]
        # This run only regenerated REQ-1's stories; REQ-2 was out of scope.
        new = _story_build([_story_record("US-FRESH-A", "Alpha", requirement_id="REQ-1")])
        result = _preserve_existing_story_ids(new, existing)
        self.assertEqual(result.records["US-R1"].story_id, "US-R1")
        # REQ-2 was never in this batch: its story is left untouched (not persisted
        # by this run, so its active DB row is preserved) and never marked outdated.
        self.assertNotIn("US-R2", result.records)

    def test_out_of_batch_scenario_parent_is_not_retired(self) -> None:
        existing = [
            _scenario_record("SC-S1", "Alpha path", story_id="US-1"),
            _scenario_record("SC-S2", "Beta path", story_id="US-2"),
        ]
        new = _scenario_build([_scenario_record("SC-FRESH", "Alpha path", story_id="US-1")])
        result = _preserve_existing_scenario_ids(new, existing)
        # US-2 was never in this batch: its scenario is left untouched, not retired.
        self.assertNotIn("SC-S2", result.records)

    def test_dropped_prior_scenario_marked_outdated(self) -> None:
        existing = [
            _scenario_record("SC-A", "Alpha path"),
            _scenario_record("SC-B", "Beta path"),
        ]
        new = _scenario_build([_scenario_record("SC-FRESH-A", "Alpha path")])
        result = _preserve_existing_scenario_ids(new, existing)
        self.assertEqual(result.records["SC-B"].status, "outdated")
        self.assertEqual(result.records["SC-A"].status, "active")

    # Two prior stories that share a title are re-paired stably (true duplicates).
    def test_duplicate_titles_pair_stably(self) -> None:
        existing = [
            _story_record("US-A1", "Same title"),
            _story_record("US-A2", "Same title"),
        ]
        new = _story_build(
            [
                _story_record("US-FRESH-1", "Same title"),
                _story_record("US-FRESH-2", "Same title"),
            ]
        )
        result = _preserve_existing_story_ids(new, existing)
        self.assertEqual(set(result.records), {"US-A1", "US-A2"})


class ScenarioIdentityTests(unittest.TestCase):
    def test_inserted_scenario_does_not_steal_prior_ids(self) -> None:
        existing = [
            _scenario_record("SC-A", "Alpha path"),
            _scenario_record("SC-B", "Beta path"),
        ]
        new = _scenario_build(
            [
                _scenario_record("SC-FRESH-G", "Gamma path"),
                _scenario_record("SC-FRESH-A", "Alpha path"),
                _scenario_record("SC-FRESH-B", "Beta path"),
            ]
        )
        result = _preserve_existing_scenario_ids(new, existing)
        by_title = {r.title: sid for sid, r in result.records.items()}
        self.assertEqual(by_title["Alpha path"], "SC-A")
        self.assertEqual(by_title["Beta path"], "SC-B")
        self.assertEqual(by_title["Gamma path"], "SC-FRESH-G")

    def test_new_scenario_keeps_minted_id(self) -> None:
        new = _scenario_build([_scenario_record("SC-FRESH-X", "Novel path")])
        result = _preserve_existing_scenario_ids(new, [])
        self.assertEqual(list(result.records), ["SC-FRESH-X"])


def _story_record(story_id: str, title: str, requirement_id: str = "REQ-1") -> UserStoryRecord:
    return UserStoryRecord(
        story_id=story_id,
        requirement_id=requirement_id,
        requirement_revision_id="REQREV-1",
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        title=title,
        priority="Medium",
        persona="Operator",
        user_story=UserStoryStatement(as_a="operator", i_want="capability", so_that="benefit"),
        acceptance_criteria=["Given a, when b, then c."],
        confidence=0.9,
    )


def _story_build(records: list[UserStoryRecord]) -> UserStoryBuildResult:
    mapping = {record.story_id: record for record in records}
    artifact = project_user_story_artifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        records=mapping,
    )
    coverage: dict[str, list[str]] = {}
    for record in records:
        coverage.setdefault(record.requirement_id, []).append(record.story_id)
    return UserStoryBuildResult(artifact=artifact, records=mapping, coverage=coverage)


def _scenario_record(scenario_id: str, title: str, story_id: str = "US-1") -> TestScenarioRecord:
    return TestScenarioRecord(
        scenario_id=scenario_id,
        story_id=story_id,
        requirement_id="REQ-1",
        requirement_revision_id="REQREV-1",
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        title=title,
        description="Verify the described behaviour occurs",
        scenario_type="Positive",
        preconditions=["a precondition"],
        expected_result="the expected result occurs",
        priority="Medium",
        confidence=0.9,
    )


def _scenario_build(records: list[TestScenarioRecord]) -> TestScenarioBuildResult:
    mapping = {record.scenario_id: record for record in records}
    artifact = project_test_scenario_artifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        records=mapping,
    )
    coverage: dict[str, list[str]] = {}
    requirement_coverage: dict[str, list[str]] = {}
    for record in records:
        coverage.setdefault(record.story_id, []).append(record.scenario_id)
        requirement_coverage.setdefault(record.requirement_id, []).append(record.scenario_id)
    return TestScenarioBuildResult(
        artifact=artifact,
        records=mapping,
        coverage=coverage,
        requirement_coverage=requirement_coverage,
    )


if __name__ == "__main__":
    unittest.main()
