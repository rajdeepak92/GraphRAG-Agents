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
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary
from multi_agentic_graph_rag.observability.session import RunSession
from multi_agentic_graph_rag.services.knowledge_context import (
    authoritative_related,
    render_assertion_lines,
)
from multi_agentic_graph_rag.services.retrieval import RetrievedContext

_WORD = re.compile(r"[A-Za-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


class UserStoryGenerationAgent:
    """One strict prompt per requirement, validated with a single retry.

    Mirrors :class:`RequirementDiscoveryAgent`: the provider adapter already does
    JSON extraction + strict Pydantic + one internal retry inside
    ``generate_structured``; this agent adds a meaningfulness pass on top and a
    second validation retry, persisting the raw response on terminal failure.
    """

    def __init__(self, reasoning_model: ReasoningModel, *, logger: Any | None = None) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
        """
        self.reasoning_model = reasoning_model
        self.logger = logger

    def generate(
        self,
        requirement: RequirementInput,
        context: RetrievedContext,
        *,
        requirement_index: int = 1,
    ) -> UserStoryGenerationOutput:
        """Generate generate.

        Args:
            requirement (RequirementInput): Requirement required by the operation's typed contract.
            context (RetrievedContext): Context required by the operation's typed contract.
            requirement_index (int): Requirement index required by the operation's typed contract.

        Returns:
            UserStoryGenerationOutput: The typed result produced by the operation.

        Raises:
            ModelOutputError: If validated inputs or required dependencies cannot satisfy the
            contract.

        Side Effects:
            May invoke configured model or workflow providers.
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
                if self.logger is not None:
                    self.logger.debug(
                        "retry.attempt_started",
                        step="generate_user_stories.requirement",
                        operation="user_story.generate",
                        requirement_id=requirement.requirement_id,
                        attempt=attempt,
                        max_attempts=2,
                        status="attempting",
                    )
                prompt = _build_user_story_prompt(
                    requirement,
                    context,
                    validation_error=validation_error,
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
                    system_message=(
                        PromptUserStoryGeneration.SYS_PROMPT_USER_STORY_GENERATION.value
                    ),
                    operation="user_story_generation.requirement",
                    request_id=requirement.requirement_id,
                )
                try:
                    _verify_user_stories(requirement, output)
                except UserStoryValidationError as error:
                    response_path = _persist_last_response(self.reasoning_model)
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
                            f"{sanitized_exception_summary(error)}; "
                            f"raw_response_path={response_path}"
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
        """Log completion metrics for one requirement generation boundary.

        Args:
            requirement (RequirementInput): Requirement required by the operation's typed contract.
            requirement_index (int): Requirement index required by the operation's typed contract.
            story_count (int): Bounded story count used for deterministic processing.
            retry_count (int): Bounded retry count used for deterministic processing.
            context_source (str): Context source required by the operation's typed contract.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
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
        """Execute the log validation failure operation within its declared architectural boundary.

        Args:
            requirement (RequirementInput): Requirement required by the operation's typed contract.
            requirement_index (int): Requirement index required by the operation's typed contract.
            attempt (int): Bounded attempt used for deterministic processing.
            error (UserStoryValidationError): Validation failure summarized without payload text.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        if self.logger is None:
            return
        self.logger.warning(
            "User-story generation failed validation",
            step="generate_user_stories.requirement",
            requirement_index=requirement_index,
            requirement_id=requirement.requirement_id,
            attempt=attempt,
            max_attempts=2,
            retry_delay_seconds=0.0,
            exception_type=error.__class__.__name__,
            error_summary=sanitized_exception_summary(error),
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
        """Run run.

        Args:
            request (UserStoryRequest): Request required by the operation's typed contract.
            session (RunSession | None): Optional command session that owns run artifacts and
                                         diagnostics.

        Returns:
            UserStoryResult: The typed result produced by the operation.
        """
        from multi_agentic_graph_rag.workflows.user_story_graph import run_user_story_generation

        return run_user_story_generation(request, session=session)


def _verify_user_stories(
    requirement: RequirementInput,
    output: UserStoryGenerationOutput,
) -> None:
    """Verify user stories against the enforced runtime contract.

    Args:
        requirement (RequirementInput): Requirement required by the operation's typed contract.
        output (UserStoryGenerationOutput): Output required by the operation's typed contract.

    Raises:
        UserStoryValidationError: If validated inputs or required dependencies cannot satisfy
        the contract.
    """
    seen_titles: set[str] = set()
    for story in output.user_stories:
        for label, value in (
            ("title", story.title),
            ("i_want", story.user_story.i_want),
            ("so_that", story.user_story.so_that),
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
        for criterion in story.acceptance_criteria:
            if len(_WORD.findall(criterion)) < 2:
                raise UserStoryValidationError(
                    "user story acceptance criterion is not descriptive enough "
                    f"({criterion!r}) for {requirement.requirement_id}"
                )


def _build_user_story_prompt(
    requirement: RequirementInput,
    context: RetrievedContext,
    validation_error: str | None = None,
) -> str:
    """Build user story prompt.

    Args:
        requirement (RequirementInput): Requirement required by the operation's typed contract.
        context (RetrievedContext): Context required by the operation's typed contract.
        validation_error (str | None): Validation error required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    requirement_json = json.dumps(
        {
            "requirement_text": requirement.requirement_text,
            "requirement_type": requirement.requirement_type,
            "priority": requirement.priority,
            "source_req_id": requirement.source_req_id or "",
            "confidence": requirement.confidence,
        },
        ensure_ascii=False,
        indent=2,
    )
    feedback = ""
    if validation_error:
        feedback = (
            f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
            "Every title, i_want, so_that, and acceptance criterion must be a complete, "
            "descriptive phrase of at least two words. Story titles must be unique.\n"
            f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{validation_error}\n\n"
        )

    if context.assertions:
        context_section = _render_assertion_context(context)
    else:
        context_section = _render_chunk_context(context)

    return (
        f"{PromptUserStoryGeneration.SYS_PROMPT_USER_STORY_GENERATION.value}"
        f"{feedback}"
        f"Requirement:\n{requirement_json}\n\n"
        f"{context_section}\n"
    )


def _render_chunk_context(context: RetrievedContext) -> str:
    """Render chunk context.

    Args:
        context (RetrievedContext): Context required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    if context.chunks:
        context_block = "\n".join(
            f"[{index}] {chunk.text}" for index, chunk in enumerate(context.chunks, start=1)
        )
    else:
        context_block = (
            "(no additional retrieved context; derive strictly from the requirement statement)"
        )
    return f"Retrieved context:\n{context_block}"


def _render_assertion_context(context: RetrievedContext) -> str:
    """Render structured assertion context with authoritative vs. related split.

    Authoritative facts (mandatory / hop-0) are directly evidenced for this
    requirement and are the grounding for stories; related facts (hop >= 1) are
    labeled disambiguation-only so the model does not manufacture stories from
    context that is merely adjacent (plan §14).
    """
    authoritative, related = authoritative_related(context.assertions)
    authoritative_block = render_assertion_lines(authoritative)
    related_block = render_assertion_lines(related)
    excerpts = ""
    if context.chunks:
        chunk_block = "\n".join(
            f"[{index}] {chunk.text}" for index, chunk in enumerate(context.chunks, start=1)
        )
        excerpts = f"\n\nSupporting source excerpts (for wording only):\n{chunk_block}"
    return (
        "AUTHORITATIVE FACTS FOR THIS REQUIREMENT (grounded in this requirement's source "
        "evidence; base every user story on these together with the requirement statement):\n"
        f"{authoritative_block}\n\n"
        "RELATED CONTEXT (for disambiguation only; do not create user stories from these "
        "alone):\n"
        f"{related_block}"
        f"{excerpts}"
    )


def _set_response_context(
    reasoning_model: ReasoningModel,
    *,
    batch_index: int,
    attempt: int,
    chunk_ids: list[str],
) -> None:
    """Execute the set response context operation within its declared architectural boundary.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
        batch_index (int): Batch index required by the operation's typed contract.
        attempt (int): Bounded attempt used for deterministic processing.
        chunk_ids (list[str]): Chunk ids required by the operation's typed contract.
    """
    setter = getattr(reasoning_model, "set_response_context", None)
    if callable(setter):
        setter(batch_index=batch_index, attempt=attempt, chunk_ids=chunk_ids)


def _clear_response_context(reasoning_model: ReasoningModel) -> None:
    """Execute the clear response context operation within its declared architectural boundary.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.
    """
    clearer = getattr(reasoning_model, "clear_response_context", None)
    if callable(clearer):
        clearer()


def _persist_last_response(reasoning_model: ReasoningModel) -> str | None:
    """Persist the last raw response under the model's canonical diagnostic filename.

    Uses the adapter's own naming (operation + request id + call index + attempt) so every
    capture shares one scheme and one file per physical call, rather than a second,
    differently-named file written only on validation failures.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.

    Returns:
        str | None: The typed result produced by the operation.
    """
    persister = getattr(reasoning_model, "persist_last_response", None)
    if not callable(persister):
        return _last_response_path(reasoning_model)
    path = persister()
    return str(path) if path is not None else None


def _last_response_path(reasoning_model: ReasoningModel) -> str | None:
    """Execute the last response path operation within its declared architectural boundary.

    Args:
        reasoning_model (ReasoningModel): Provider-neutral model adapter used by the operation.

    Returns:
        str | None: The typed result produced by the operation.
    """
    response_path = getattr(reasoning_model, "last_response_path", None)
    return str(response_path) if response_path else None
