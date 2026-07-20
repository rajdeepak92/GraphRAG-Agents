"""Grounded Stage 3 scenario generation with one targeted repair."""

from __future__ import annotations

import json

from multi_agentic_graph_rag.common_prompt_defs import PromptTestScenarioGeneration
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalUserStory,
    LLMTestScenarioCandidate,
    ScenarioContext,
    TestScenarioGenerationDraft,
    TestScenarioGenerationResponse,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel


class TestScenarioGenerationAgent:
    """Generate and validate behavioral scenarios for one story."""

    __test__ = False

    def __init__(self, reasoning_model: ReasoningModel) -> None:
        self.reasoning_model = reasoning_model

    def generate(
        self,
        story: CanonicalUserStory,
        context: ScenarioContext,
    ) -> TestScenarioGenerationResponse:
        """Allow one targeted repair while reusing the frozen context."""
        base = {
            "story": story.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        feedback: str | None = None
        for attempt in (1, 2):
            prompt_payload: dict[str, object] = dict(base)
            if feedback is not None:
                prompt_payload["validation_feedback"] = feedback
                prompt_payload["repair_instruction"] = (
                    "Return a corrected object for the same story and frozen context."
                )
            try:
                draft = self.reasoning_model.generate_structured(
                    prompt=json.dumps(prompt_payload, ensure_ascii=False),
                    schema=TestScenarioGenerationDraft,
                    system_message=PromptTestScenarioGeneration.SYSTEM.value,
                    operation=(
                        "test_scenario.generate" if attempt == 1 else "test_scenario.repair"
                    ),
                    request_id=story.story_id,
                    max_attempts=1,
                )
                response = self._stamp_provenance(story, draft)
                self._validate(story, context, response)
                return response
            except Exception as exc:
                if attempt == 2:
                    raise
                feedback = str(exc)
        raise AssertionError("unreachable")

    @staticmethod
    def _stamp_provenance(
        story: CanonicalUserStory,
        draft: TestScenarioGenerationDraft,
    ) -> TestScenarioGenerationResponse:
        """Inherit immutable source provenance instead of asking the model to copy it."""
        return TestScenarioGenerationResponse(
            story_id=draft.story_id,
            requirement_ids=draft.requirement_ids,
            test_scenarios=[
                LLMTestScenarioCandidate(
                    source_req_id=story.source_req_id,
                    source_req_id_type=story.source_req_id_type,
                    **scenario.model_dump(),
                )
                for scenario in draft.test_scenarios
            ],
        )

    @staticmethod
    def _validate(
        story: CanonicalUserStory,
        context: ScenarioContext,
        response: TestScenarioGenerationResponse,
    ) -> None:
        if response.story_id != story.story_id:
            raise ValueError("response story_id does not match the supplied story")
        if response.requirement_ids != story.requirement_ids:
            raise ValueError("response requirement_ids do not match the story mappings")
        criteria = {criterion.criterion_id for criterion in story.acceptance_criteria}
        chunks = {item.chunk_id for item in context.ranked_evidence}
        chunks.update(context.authoritative_evidence_chunk_ids)
        entities = set(context.supporting_entity_ids)
        relationships = set(context.supporting_relationship_ids)
        for scenario in response.test_scenarios:
            if (
                scenario.source_req_id != story.source_req_id
                or scenario.source_req_id_type != story.source_req_id_type
            ):
                raise ValueError("scenario source provenance was not inherited unchanged")
            if not set(scenario.covered_acceptance_criterion_ids) <= criteria:
                raise ValueError("scenario covers an unknown acceptance criterion")
            if not set(scenario.evidence_chunk_ids) <= chunks:
                raise ValueError("scenario references a chunk outside the supplied context")
            if not set(scenario.supporting_entity_ids) <= entities:
                raise ValueError("scenario references an entity outside the supplied context")
            if not set(scenario.supporting_relationship_ids) <= relationships:
                raise ValueError("scenario references a relationship outside the supplied context")


__all__ = ["TestScenarioGenerationAgent"]
