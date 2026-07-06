"""Enterprise user-story generation agent (stage 3)."""

from __future__ import annotations

import json
import re
from typing import Any

from multi_agentic_graph_rag.common_prompt_defs import (
    PromptSharedFragments,
    PromptUserStoryGeneration,
)
from multi_agentic_graph_rag.domain.errors import ModelOutputError, UserStoryValidationError
from multi_agentic_graph_rag.domain.schemas import (
    RequirementInput,
    UserStoryGenerationOutput,
    UserStoryRequest,
    UserStoryResult,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.observability.session import RunSession
from multi_agentic_graph_rag.services.retrieval import RetrievedContext

_WORD = re.compile(r"[A-Za-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


class UserStoryGenerationAgent:
    """One strict prompt per requirement, validated with a single fed-back retry.

    Mirrors :class:`RequirementDiscoveryAgent`: the provider adapter already does
    JSON extraction + strict Pydantic + one internal retry inside
    ``generate_structured``; this agent adds a meaningfulness pass on top and a
    second fed-back retry, persisting the raw response on terminal failure.
    """

    def __init__(self, reasoning_model: ReasoningModel, *, logger: Any | None = None) -> None:
        self.reasoning_model = reasoning_model
        self.logger = logger

    def generate(
        self,
        requirement: RequirementInput,
        context: RetrievedContext,
        *,
        requirement_index: int = 1,
        reviewer_directive: str | None = None,
    ) -> UserStoryGenerationOutput:
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
                prompt = _build_user_story_prompt(
                    requirement,
                    context,
                    validation_error=validation_error,
                    reviewer_directive=reviewer_directive,
                )
                _set_response_context(
                    self.reasoning_model,
                    batch_index=requirement_index,
                    attempt=attempt,
                    chunk_ids=[requirement.requirement_id],
                )
                output = self.reasoning_model.generate_structured(
                    prompt=prompt,
                    schema=UserStoryGenerationOutput,
                )
                try:
                    _verify_user_stories(requirement, output)
                except UserStoryValidationError as error:
                    response_path = _persist_last_response(
                        self.reasoning_model,
                        requirement_index=requirement_index,
                        attempt=attempt,
                    )
                    self._log_validation_failure(
                        requirement=requirement,
                        requirement_index=requirement_index,
                        attempt=attempt,
                        error=error,
                        response_path=response_path,
                    )
                    if attempt == 2:
                        raise ModelOutputError(
                            "User-story generation for "
                            f"{requirement.requirement_id} failed validation after retry: "
                            f"{error}; raw_response_path={response_path}"
                        ) from error
                    validation_error = str(error)
                    continue

                self._log_requirement_completed(
                    requirement=requirement,
                    requirement_index=requirement_index,
                    story_count=len(output.user_stories),
                    retry_count=attempt - 1,
                    context_source=context.source,
                    response_path=_last_response_path(self.reasoning_model),
                )
                return output
        finally:
            _clear_response_context(self.reasoning_model)

        raise ModelOutputError(
            f"User-story generation for {requirement.requirement_id} did not produce a result"
        )

    def _log_requirement_completed(
        self,
        *,
        requirement: RequirementInput,
        requirement_index: int,
        story_count: int,
        retry_count: int,
        context_source: str,
        response_path: str | None,
    ) -> None:
        if self.logger is None:
            return
        self.logger.info(
            "User stories generated for requirement",
            step="generate_user_stories.requirement",
            requirement_index=requirement_index,
            requirement_id=requirement.requirement_id,
            story_count=story_count,
            retry_count=retry_count,
            context_source=context_source,
            raw_response_path=response_path,
            status="completed",
        )

    def _log_validation_failure(
        self,
        *,
        requirement: RequirementInput,
        requirement_index: int,
        attempt: int,
        error: UserStoryValidationError,
        response_path: str | None,
    ) -> None:
        if self.logger is None:
            return
        self.logger.warning(
            "User-story generation failed validation",
            step="generate_user_stories.requirement",
            requirement_index=requirement_index,
            requirement_id=requirement.requirement_id,
            attempt=attempt,
            retry_count=attempt - 1,
            error=str(error),
            raw_response_path=response_path,
            status="failed" if attempt == 2 else "retrying",
        )


class UserStoryGeneratorAgent:
    """Public standalone stage-3 agent (mirrors ``IngestionDocumentAgent``)."""

    def run(
        self,
        request: UserStoryRequest,
        *,
        session: RunSession | None = None,
    ) -> UserStoryResult:
        from multi_agentic_graph_rag.workflows.user_story_graph import run_user_story_generation

        return run_user_story_generation(request, session=session)


def _verify_user_stories(
    requirement: RequirementInput,
    output: UserStoryGenerationOutput,
) -> None:
    seen_titles: set[str] = set()
    for story in output.user_stories:
        for label, value in (
            ("title", story.title),
            ("i_want", story.user_story.i_want),
            ("so_that", story.user_story.so_that),
            ("business_value", story.business_value),
        ):
            if len(_WORD.findall(value)) < 2:
                raise UserStoryValidationError(
                    f"user story {label} is not descriptive enough ({value!r}) for "
                    f"{requirement.requirement_id}"
                )
        title_key = _WHITESPACE.sub(" ", story.title.strip().lower())
        if title_key in seen_titles:
            raise UserStoryValidationError(
                f"duplicate user story title ({story.title!r}) for {requirement.requirement_id}"
            )
        seen_titles.add(title_key)


def _build_user_story_prompt(
    requirement: RequirementInput,
    context: RetrievedContext,
    validation_error: str | None = None,
    *,
    reviewer_directive: str | None = None,
) -> str:
    requirement_json = json.dumps(
        {
            "requirement_text": requirement.requirement_text,
            "requirement_type": requirement.requirement_type,
            "priority": requirement.priority,
        },
        ensure_ascii=False,
        indent=2,
    )
    if context.chunks:
        context_block = "\n".join(
            f"[{index}] {chunk.text}" for index, chunk in enumerate(context.chunks, start=1)
        )
    else:
        context_block = (
            "(no additional retrieved context; derive strictly from the requirement statement)"
        )

    feedback = ""
    if validation_error:
        feedback = (
            f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
            "Every title, i_want, so_that, and business_value must be a complete, "
            "descriptive phrase of at least two words. Story titles must be unique.\n"
            f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{validation_error}\n\n"
        )

    directive_block = ""
    if reviewer_directive:
        directive_block = (
            f"{PromptSharedFragments.REVIEWER_DIRECTIVE_HEADER.value}\n"
            f"{reviewer_directive.strip()}\n"
            f"{PromptSharedFragments.GENERATE_ONLY_ADDITIONAL_USER_STORIES.value}\n"
            f"{PromptSharedFragments.REVIEWER_DIRECTIVE_FOOTER.value}\n\n"
        )

    return (
        f"{PromptUserStoryGeneration.SYS_PROMPT_USER_STORY_GENERATION.value}"
        f"{feedback}"
        f"{directive_block}"
        f"Requirement:\n{requirement_json}\n\n"
        f"Retrieved context:\n{context_block}\n"
    )


def _set_response_context(
    reasoning_model: ReasoningModel,
    *,
    batch_index: int,
    attempt: int,
    chunk_ids: list[str],
) -> None:
    setter = getattr(reasoning_model, "set_response_context", None)
    if callable(setter):
        setter(batch_index=batch_index, attempt=attempt, chunk_ids=chunk_ids)


def _clear_response_context(reasoning_model: ReasoningModel) -> None:
    clearer = getattr(reasoning_model, "clear_response_context", None)
    if callable(clearer):
        clearer()


def _persist_last_response(
    reasoning_model: ReasoningModel,
    *,
    requirement_index: int,
    attempt: int,
) -> str | None:
    persister = getattr(reasoning_model, "persist_last_response", None)
    if not callable(persister):
        return _last_response_path(reasoning_model)
    path = persister(filename=f"llm_response_us_{requirement_index}_{attempt}.txt")
    return str(path) if path is not None else None


def _last_response_path(reasoning_model: ReasoningModel) -> str | None:
    response_path = getattr(reasoning_model, "last_response_path", None)
    return str(response_path) if response_path else None
