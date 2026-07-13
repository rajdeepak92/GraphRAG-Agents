from __future__ import annotations

import unittest

from multi_agentic_graph_rag.domain.schemas import (
    RequirementInput,
    UserStoryModel,
    UserStoryStatement,
)
from multi_agentic_graph_rag.services.user_story_builder import build_user_story_artifact


class UserStoryBuilderTests(unittest.TestCase):
    def test_assigns_permanent_ids_and_groups_coverage(self) -> None:
        requirement = _requirement("REQ-1", "Users shall configure warning thresholds.")
        artifact = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=[(requirement, _story("Configure warning thresholds"))],
        )

        self.assertEqual(len(artifact.records), 1)
        story_id = next(iter(artifact.records))
        self.assertTrue(story_id.startswith("US-"))
        record = artifact.records[story_id]
        self.assertEqual(record.requirement_id, "REQ-1")
        self.assertEqual(record.project, "SIIMCS")
        self.assertEqual(record.document_id, "DOC-1")
        self.assertEqual(record.document_version_id, "DV-1")
        self.assertEqual(record.doc_version, "1.0")
        self.assertEqual(artifact.coverage, {"REQ-1": [story_id]})
        self.assertEqual(artifact.artifact.stories[0].story_id, story_id)

    def test_projects_flat_acceptance_criteria(self) -> None:
        requirement = _requirement("REQ-1", "Users shall configure warning thresholds.")
        story = _story("Configure warning thresholds", acceptance=3)
        artifact = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=[(requirement, story)],
        )

        record = next(iter(artifact.records.values()))
        self.assertEqual(len(record.acceptance_criteria), 3)
        self.assertEqual(
            artifact.artifact.stories[0].acceptance_criteria, record.acceptance_criteria
        )

    def test_multiple_stories_per_requirement_get_distinct_ids(self) -> None:
        requirement = _requirement("REQ-1", "Users shall configure warning thresholds.")
        artifact = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=[
                (requirement, _story("Configure warning thresholds")),
                (requirement, _story("Configure critical thresholds")),
            ],
        )

        self.assertEqual(len(artifact.records), 2)
        self.assertEqual(len(artifact.coverage["REQ-1"]), 2)
        self.assertEqual(len(set(artifact.coverage["REQ-1"])), 2)

    def test_build_is_deterministic_and_idempotent(self) -> None:
        requirement = _requirement("REQ-1", "Users shall configure warning thresholds.")
        pairs = [(requirement, _story("Configure warning thresholds"))]

        first = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=pairs,
        )
        second = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=pairs,
        )

        self.assertEqual(list(first.records), list(second.records))
        self.assertEqual(first.coverage, second.coverage)


def _requirement(requirement_id: str, text: str) -> RequirementInput:
    return RequirementInput(
        requirement_id=requirement_id,
        requirement_text=text,
        requirement_type="Functional Requirement",
        priority="Medium",
        evidence_chunk_ids=["CHUNK-0001"],
    )


def _story(
    title: str,
    *,
    priority: str = "Medium",
    acceptance: int = 1,
) -> UserStoryModel:
    return UserStoryModel(
        title=title,
        priority=priority,
        persona="Operations Engineer",
        user_story=UserStoryStatement(
            as_a="operations engineer",
            i_want="to configure warning thresholds",
            so_that="alerts fire before equipment is damaged",
        ),
        acceptance_criteria=[
            f"criterion {index}: alert is raised when a threshold is crossed"
            for index in range(1, acceptance + 1)
        ],
        confidence=0.85,
    )


if __name__ == "__main__":
    unittest.main()
