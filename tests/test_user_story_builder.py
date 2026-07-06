from __future__ import annotations

import unittest

from multi_agentic_graph_rag.domain.schemas import (
    AcceptanceCriterion,
    BusinessRule,
    RequirementInput,
    TestScenario,
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

        self.assertEqual(len(artifact.stories), 1)
        story_id = next(iter(artifact.stories))
        self.assertTrue(story_id.startswith("US-"))
        record = artifact.stories[story_id]
        self.assertEqual(record.requirement_id, "REQ-1")
        self.assertEqual(record.project, "SIIMCS")
        self.assertEqual(record.document_id, "DOC-1")
        self.assertEqual(record.document_version_id, "DV-1")
        self.assertEqual(record.doc_version, "1.0")
        self.assertEqual(artifact.coverage, {"REQ-1": [story_id]})

    def test_renumbers_acceptance_business_and_test_ids(self) -> None:
        requirement = _requirement("REQ-1", "Users shall configure warning thresholds.")
        story = _story(
            "Configure warning thresholds",
            acceptance=3,
            business_rules=2,
            test_scenarios=2,
        )
        artifact = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=[(requirement, story)],
        )

        record = next(iter(artifact.stories.values()))
        self.assertEqual(
            [ac.id for ac in record.acceptance_criteria], ["AC-001", "AC-002", "AC-003"]
        )
        self.assertEqual([br.id for br in record.business_rules], ["BR-001", "BR-002"])
        self.assertEqual([ts.id for ts in record.test_scenarios], ["TS-001", "TS-002"])

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

        self.assertEqual(len(artifact.stories), 2)
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

        self.assertEqual(list(first.stories), list(second.stories))
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
    business_rules: int = 1,
    test_scenarios: int = 1,
) -> UserStoryModel:
    return UserStoryModel(
        title=title,
        epic="Threshold Management",
        priority=priority,
        persona="Operations Engineer",
        user_story=UserStoryStatement(
            as_a="operations engineer",
            i_want="to configure warning thresholds",
            so_that="alerts fire before equipment is damaged",
        ),
        business_value="Reduces unplanned downtime through timely alerting",
        acceptance_criteria=[
            AcceptanceCriterion(
                id=f"ACX{index}",
                title=f"criterion {index}",
                given="a configured sensor",
                when="a threshold is crossed",
                then="an alert is raised",
            )
            for index in range(1, acceptance + 1)
        ],
        business_rules=[
            BusinessRule(id=f"BRX{index}", rule=f"rule {index}")
            for index in range(1, business_rules + 1)
        ],
        test_scenarios=[
            TestScenario(id=f"TSX{index}", scenario=f"scenario {index}")
            for index in range(1, test_scenarios + 1)
        ],
        definition_of_done=["code reviewed", "tests passing"],
    )


if __name__ == "__main__":
    unittest.main()
