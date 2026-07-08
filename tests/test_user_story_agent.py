from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from multi_agentic_graph_rag.agents.user_story_agent import UserStoryGenerationAgent
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import RequirementInput
from multi_agentic_graph_rag.services.retrieval import RetrievedChunk, RetrievedContext
from multi_agentic_graph_rag.services.user_story_builder import build_user_story_artifact

T = TypeVar("T", bound=BaseModel)


class UserStoryAgentTests(unittest.TestCase):
    def test_generate_returns_stories_and_prompt_includes_context(self) -> None:
        reasoner = _StoryReasoner([_story_payload("Configure warning thresholds")])
        context = RetrievedContext(
            chunks=[RetrievedChunk(chunk_id="CHUNK-0001", text="warning threshold context")],
            source="hybrid",
        )

        output = UserStoryGenerationAgent(reasoner).generate(_requirement(), context)

        self.assertEqual(reasoner.prompts, 1)
        self.assertEqual(len(output.user_stories), 1)
        self.assertIn("warning threshold context", reasoner.last_prompt)
        self.assertIn("Users shall configure warning thresholds", reasoner.last_prompt)

    def test_temp_ids_replaced_and_requirement_id_attached(self) -> None:
        reasoner = _StoryReasoner([_story_payload("Configure warning thresholds")])
        requirement = _requirement()

        output = UserStoryGenerationAgent(reasoner).generate(requirement, _empty_context())
        artifact = build_user_story_artifact(
            project="SIIMCS",
            document_id="DOC-1",
            document_version_id="DV-1",
            doc_version="1.0",
            generated=[(requirement, story) for story in output.user_stories],
        )

        story_id = next(iter(artifact.records))
        self.assertTrue(story_id.startswith("US-"))
        self.assertNotEqual(story_id, "US1")
        self.assertEqual(artifact.records[story_id].requirement_id, requirement.requirement_id)

    def test_non_descriptive_story_is_retried_once(self) -> None:
        reasoner = _RetryStoryReasoner(
            bad=_story_payload("Configure"),
            good=_story_payload("Configure warning thresholds"),
        )

        output = UserStoryGenerationAgent(reasoner).generate(_requirement(), _empty_context())

        self.assertEqual(reasoner.prompts, 2)
        self.assertEqual(output.user_stories[0].title, "Configure warning thresholds")

    def test_persistently_invalid_output_saves_raw_response_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            reasoner = _PersistingBadReasoner(Path(temp_dir), _story_payload("Configure"))

            with self.assertRaises(ModelOutputError):
                UserStoryGenerationAgent(reasoner).generate(
                    _requirement(), _empty_context(), requirement_index=1
                )

            first = Path(temp_dir) / "llm_response_us_1_1.txt"
            second = Path(temp_dir) / "llm_response_us_1_2.txt"
            self.assertEqual(first.read_text(encoding="utf-8"), "raw response 1")
            self.assertEqual(second.read_text(encoding="utf-8"), "raw response 2")


class _StoryReasoner:
    provider_name = "huggingface"

    def __init__(self, stories: list[dict[str, Any]]) -> None:
        self.stories = stories
        self.prompts = 0
        self.last_prompt = ""

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        self.prompts += 1
        self.last_prompt = prompt
        return schema.model_validate({"user_stories": self.stories})


class _RetryStoryReasoner:
    provider_name = "huggingface"

    def __init__(self, *, bad: dict[str, Any], good: dict[str, Any]) -> None:
        self.bad = bad
        self.good = good
        self.prompts = 0

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        self.prompts += 1
        story = self.good if self.prompts == 2 else self.bad
        return schema.model_validate({"user_stories": [story]})


class _PersistingBadReasoner:
    provider_name = "azure_openai"

    def __init__(self, run_dir: Path, story: dict[str, Any]) -> None:
        self.run_dir = run_dir
        self.story = story
        self.prompts = 0
        self.last_response_path: Path | None = None
        self._last_response = ""

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        self.prompts += 1
        self._last_response = f"raw response {self.prompts}"
        return schema.model_validate({"user_stories": [self.story]})

    def persist_last_response(self, *, filename: str) -> Path:
        path = self.run_dir / filename
        path.write_text(self._last_response, encoding="utf-8")
        self.last_response_path = path
        return path


def _requirement() -> RequirementInput:
    return RequirementInput(
        requirement_id="REQ-1",
        requirement_text="Users shall configure warning thresholds.",
        requirement_type="Functional Requirement",
        priority="Medium",
        evidence_chunk_ids=["CHUNK-0001"],
    )


def _empty_context() -> RetrievedContext:
    return RetrievedContext(chunks=[], source="requirement_text_fallback")


def _story_payload(title: str) -> dict[str, Any]:
    return {
        "title": title,
        "priority": "Medium",
        "persona": "Operations Engineer",
        "user_story": {
            "as_a": "operations engineer",
            "i_want": "to configure warning thresholds",
            "so_that": "alerts fire before equipment is damaged",
        },
        "acceptance_criteria": [
            "Given a configured sensor, when a threshold is crossed, then an alert is raised."
        ],
        "confidence": 0.85,
    }


if __name__ == "__main__":
    unittest.main()
