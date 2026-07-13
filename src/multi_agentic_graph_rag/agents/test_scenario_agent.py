"""Enterprise test-scenario generation agent (stage 4)."""

from __future__ import annotations

import json
import re
from typing import Any

from multi_agentic_graph_rag.common_prompt_defs import (
    PromptSharedFragments,
    PromptTestScenarioGeneration,
)
from multi_agentic_graph_rag.domain.errors import (
    ModelOutputError,
    TestScenarioValidationError,
)
from multi_agentic_graph_rag.domain.schemas import (
    TestScenarioGenerationOutput,
    TestScenarioRequest,
    TestScenarioResult,
    UserStoryRecord,
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


class TestScenarioGenerationAgent:
    """One strict prompt per user story, validated with a single retry."""

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
        story: UserStoryRecord,
        context: RetrievedContext,
        *,
        story_index: int = 1,
        requirement_text: str | None = None,
    ) -> TestScenarioGenerationOutput:
        """Generate generate.

        Args:
            story (UserStoryRecord): Story required by the operation's typed contract.
            context (RetrievedContext): Context required by the operation's typed contract.
            story_index (int): Story index required by the operation's typed contract.
            requirement_text (str | None): Optional requirement text excluded from logs.

        Returns:
            TestScenarioGenerationOutput: The typed result produced by the operation.

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
                        step="generate_test_scenarios.story",
                        operation="test_scenario.generate",
                        story_id=story.story_id,
                        attempt=attempt,
                        max_attempts=2,
                        status="attempting",
                    )
                prompt = _build_test_scenario_prompt(
                    story,
                    context,
                    requirement_text=requirement_text,
                    validation_error=validation_error,
                )
                _set_response_context(
                    self.reasoning_model,
                    batch_index=story_index,
                    attempt=attempt,
                    chunk_ids=[story.story_id],
                )
                output = self.reasoning_model.generate_structured(
                    prompt=prompt,
                    schema=TestScenarioGenerationOutput,
                    system_message=(
                        PromptTestScenarioGeneration.SYS_PROMPT_TEST_SCENARIO_GENERATION.value
                    ),
                    operation="test_scenario_generation.story",
                    request_id=story.story_id,
                )
                try:
                    _verify_test_scenarios(story, output)
                except TestScenarioValidationError as error:
                    response_path = _persist_last_response(self.reasoning_model)
                    self._log_validation_failure(
                        story=story,
                        story_index=story_index,
                        attempt=attempt,
                        error=error,
                        response_path=response_path,
                    )
                    if attempt == 2:
                        raise ModelOutputError(
                            "Test-scenario generation for "
                            f"{story.story_id} failed validation after retry: "
                            f"{sanitized_exception_summary(error)}; "
                            f"raw_response_path={response_path}"
                        ) from error
                    validation_error = str(error)
                    continue

                self._log_story_completed(
                    story=story,
                    story_index=story_index,
                    scenario_count=len(output.test_scenarios),
                    retry_count=attempt - 1,
                    context_source=context.source,
                    response_path=_last_response_path(self.reasoning_model),
                )
                return output
        finally:
            _clear_response_context(self.reasoning_model)

        raise ModelOutputError(
            f"Test-scenario generation for {story.story_id} did not produce a result"
        )

    def _log_story_completed(
        self,
        *,
        story: UserStoryRecord,
        story_index: int,
        scenario_count: int,
        retry_count: int,
        context_source: str,
        response_path: str | None,
    ) -> None:
        """Execute the log story completed operation within its declared architectural boundary.

        Args:
            story (UserStoryRecord): Story required by the operation's typed contract.
            story_index (int): Story index required by the operation's typed contract.
            scenario_count (int): Bounded scenario count used for deterministic processing.
            retry_count (int): Bounded retry count used for deterministic processing.
            context_source (str): Context source required by the operation's typed contract.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        if self.logger is None:
            return
        self.logger.info(
            "Test scenarios generated for user story",
            step="generate_test_scenarios.story",
            story_index=story_index,
            story_id=story.story_id,
            requirement_id=story.requirement_id,
            scenario_count=scenario_count,
            retry_count=retry_count,
            context_source=context_source,
            raw_response_path=response_path,
            status="completed",
        )

    def _log_validation_failure(
        self,
        *,
        story: UserStoryRecord,
        story_index: int,
        attempt: int,
        error: TestScenarioValidationError,
        response_path: str | None,
    ) -> None:
        """Execute the log validation failure operation within its declared architectural boundary.

        Args:
            story (UserStoryRecord): Story required by the operation's typed contract.
            story_index (int): Story index required by the operation's typed contract.
            attempt (int): Bounded attempt used for deterministic processing.
            error (TestScenarioValidationError): Validation failure summarized without payload text.
            response_path (str | None): Filesystem location authorized for this operation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        if self.logger is None:
            return
        self.logger.warning(
            "Test-scenario generation failed validation",
            step="generate_test_scenarios.story",
            story_index=story_index,
            story_id=story.story_id,
            requirement_id=story.requirement_id,
            attempt=attempt,
            max_attempts=2,
            retry_delay_seconds=0.0,
            exception_type=error.__class__.__name__,
            error_summary=sanitized_exception_summary(error),
            raw_response_path=response_path,
            status="failed" if attempt == 2 else "retrying",
        )


class TestScenarioGeneratorAgent:
    """Public standalone stage-4 agent (mirrors ``UserStoryGeneratorAgent``)."""

    def run(
        self,
        request: TestScenarioRequest,
        *,
        session: RunSession | None = None,
    ) -> TestScenarioResult:
        """Run run.

        Args:
            request (TestScenarioRequest): Request required by the operation's typed contract.
            session (RunSession | None): Optional command session that owns run artifacts and
                                         diagnostics.

        Returns:
            TestScenarioResult: The typed result produced by the operation.
        """
        from multi_agentic_graph_rag.workflows.test_scenario_graph import (
            run_test_scenario_generation,
        )

        return run_test_scenario_generation(request, session=session)


def _verify_test_scenarios(
    story: UserStoryRecord,
    output: TestScenarioGenerationOutput,
) -> None:
    """Verify test scenarios against the enforced runtime contract.

    Args:
        story (UserStoryRecord): Story required by the operation's typed contract.
        output (TestScenarioGenerationOutput): Output required by the operation's typed contract.

    Raises:
        TestScenarioValidationError: If validated inputs or required dependencies cannot satisfy
        the contract.
    """
    seen_titles: set[str] = set()
    for scenario in output.test_scenarios:
        for label, value in (
            ("title", scenario.title),
            ("description", scenario.description),
            ("expected_result", scenario.expected_result),
        ):
            if len(_WORD.findall(value)) < 2:
                raise TestScenarioValidationError(
                    f"test scenario {label} is not descriptive enough ({value!r}) "
                    f"for {story.story_id}"
                )
        for precondition in scenario.preconditions:
            if not precondition.strip():
                raise TestScenarioValidationError(
                    f"test scenario precondition is empty for {story.story_id}"
                )
        title_key = _WHITESPACE.sub(" ", scenario.title.strip().lower())
        if title_key in seen_titles:
            raise TestScenarioValidationError(
                f"duplicate test scenario title ({scenario.title!r}) for {story.story_id}"
            )
        seen_titles.add(title_key)


def _build_test_scenario_prompt(
    story: UserStoryRecord,
    context: RetrievedContext,
    *,
    requirement_text: str | None = None,
    validation_error: str | None = None,
) -> str:
    """Build test scenario prompt.

    Args:
        story (UserStoryRecord): Story required by the operation's typed contract.
        context (RetrievedContext): Context required by the operation's typed contract.
        requirement_text (str | None): Input text processed in memory and excluded from diagnostic
                                       logs.
        validation_error (str | None): Validation error required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    story_json = json.dumps(
        {
            "title": story.title,
            "priority": story.priority,
            "persona": story.persona,
            "user_story": story.user_story.model_dump(mode="json"),
            "acceptance_criteria": list(story.acceptance_criteria),
            "confidence": story.confidence,
        },
        ensure_ascii=False,
        indent=2,
    )
    linked_requirement = requirement_text or (
        "(requirement text unavailable; rely on the user story and retrieved context)"
    )
    if context.assertions:
        context_section = _render_assertion_context(context)
    else:
        context_section = _render_chunk_context(context)

    feedback = ""
    if validation_error:
        feedback = (
            f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
            "Every title, description, and expected_result must be a complete, descriptive "
            "phrase of at least two words. Preconditions must be non-empty and scenario "
            "titles must be unique per story.\n"
            f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{validation_error}\n\n"
        )

    return (
        f"{PromptTestScenarioGeneration.PROMPT_TEST_SCENARIO_GENERATION.value}"
        f"{feedback}"
        f"User story:\n{story_json}\n\n"
        f"Linked requirement:\n{linked_requirement}\n\n"
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
            "(no additional retrieved context; derive strictly from the user story and "
            "linked requirement)"
        )
    return f"Retrieved context:\n{context_block}"


def _render_assertion_context(context: RetrievedContext) -> str:
    """Structured assertion context for scenario generation (plan §15).

    Authoritative facts (mandatory / hop-0) ground the expected results; related
    facts (hop >= 1) supply constraints/conditions to probe as test dimensions
    but are not, on their own, sufficient to assert a mandatory expected result.
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
        "AUTHORITATIVE FACTS FOR THIS STORY (grounded in the story's linked requirement "
        "evidence; every expected result must be supported by these, the acceptance "
        "criteria, or the linked requirement):\n"
        f"{authoritative_block}\n\n"
        "RELATED CONSTRAINTS AND CONDITIONS (probe these as positive/negative/boundary/"
        "precondition/timeout test dimensions; do not assert a mandatory expected result "
        "from an unsupported related fact alone):\n"
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
