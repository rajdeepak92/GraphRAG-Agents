"""Grounded Stage 2 user-story generation with one targeted repair."""

from __future__ import annotations

import json

from multi_agentic_graph_rag.common_prompt_defs import PromptUserStoryGeneration
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalRequirement,
    LLMUserStoryCandidate,
    StoryContext,
    UserStoryGenerationDraft,
    UserStoryGenerationResponse,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel


class UserStoryGenerationAgent:
    """Generate and semantically validate stories for one requirement."""

    def __init__(self, reasoning_model: ReasoningModel) -> None:
        self.reasoning_model = reasoning_model

    def generate(
        self,
        requirement: CanonicalRequirement,
        context: StoryContext,
    ) -> UserStoryGenerationResponse:
        """Allow one targeted output repair without repeating retrieval."""
        base = {
            "requirement": requirement.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
        }
        feedback: str | None = None
        for attempt in (1, 2):
            prompt_payload: dict[str, object] = dict(base)
            if feedback is not None:
                prompt_payload["validation_feedback"] = feedback
                prompt_payload["repair_instruction"] = (
                    "Return a corrected object for the same requirement and frozen context."
                )
            try:
                draft = self.reasoning_model.generate_structured(
                    prompt=json.dumps(prompt_payload, ensure_ascii=False),
                    schema=UserStoryGenerationDraft,
                    system_message=PromptUserStoryGeneration.SYSTEM.value,
                    operation="user_story.generate" if attempt == 1 else "user_story.repair",
                    request_id=requirement.requirement_id,
                    max_attempts=2,
                )
                response = self._stamp_provenance(requirement, draft)
                self._validate(requirement, context, response)
                return response
            except Exception as exc:
                if attempt == 2:
                    raise
                feedback = str(exc)
        raise AssertionError("unreachable")

    @staticmethod
    def _stamp_provenance(
        requirement: CanonicalRequirement,
        draft: UserStoryGenerationDraft,
    ) -> UserStoryGenerationResponse:
        """Inherit provenance from the requirement instead of trusting the model.

        Provenance is deterministic (it must equal the requirement's, see
        :meth:`_validate`), so it is stamped here rather than reproduced by the
        model, which otherwise conflates ``requirement_id`` with ``source_req_id``.
        """
        return UserStoryGenerationResponse(
            requirement_id=draft.requirement_id,
            user_stories=[
                LLMUserStoryCandidate(
                    source_req_id=requirement.source_req_id,
                    source_req_id_type=requirement.source_req_id_type,
                    **story.model_dump(),
                )
                for story in draft.user_stories
            ],
        )

    @staticmethod
    def _validate(
        requirement: CanonicalRequirement,
        context: StoryContext,
        response: UserStoryGenerationResponse,
    ) -> None:
        if response.requirement_id != requirement.requirement_id:
            raise ValueError("response requirement_id does not match the supplied requirement")
        allowed_chunks = {item.chunk_id for item in context.ranked_evidence}
        allowed_chunks.update(context.authoritative_evidence_chunk_ids)
        allowed_entities = set(context.mapped_entity_ids)
        allowed_relationships = set(context.mapped_relationship_ids)
        for story in response.user_stories:
            if (
                story.source_req_id != requirement.source_req_id
                or story.source_req_id_type != requirement.source_req_id_type
            ):
                raise ValueError("story source provenance was not inherited unchanged")
            if not set(story.evidence_chunk_ids) <= allowed_chunks:
                raise ValueError("story references a chunk outside the supplied context")
            if not set(story.supporting_entity_ids) <= allowed_entities:
                raise ValueError("story references an entity outside the supplied context")
            if not set(story.supporting_relationship_ids) <= allowed_relationships:
                raise ValueError("story references a relationship outside the supplied context")


__all__ = ["UserStoryGenerationAgent"]
