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
from multi_agentic_graph_rag.observability.session import RunSession
from multi_agentic_graph_rag.services.retrieval import RetrievedContext

_WORD = re.compile(r"[A-Za-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


class TestScenarioGenerationAgent:
    """One strict prompt per user story, validated with a single retry."""

    def __init__(self, reasoning_model: ReasoningModel, *, logger: Any | None = None) -> None:
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
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
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
                )
                try:
                    _verify_test_scenarios(story, output)
                except TestScenarioValidationError as error:
                    response_path = _persist_last_response(
                        self.reasoning_model,
                        story_index=story_index,
                        attempt=attempt,
                    )
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
                            f"{error}; raw_response_path={response_path}"
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
        if self.logger is None:
            return
        self.logger.warning(
            "Test-scenario generation failed validation",
            step="generate_test_scenarios.story",
            story_index=story_index,
            story_id=story.story_id,
            requirement_id=story.requirement_id,
            attempt=attempt,
            retry_count=attempt - 1,
            error=str(error),
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
        from multi_agentic_graph_rag.workflows.test_scenario_graph import (
            run_test_scenario_generation,
        )

        return run_test_scenario_generation(request, session=session)


def _verify_test_scenarios(
    story: UserStoryRecord,
    output: TestScenarioGenerationOutput,
) -> None:
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
    story_json = json.dumps(
        {
            "title": story.title,
            "priority": story.priority,
            "persona": story.persona,
            "user_story": story.user_story.model_dump(mode="json"),
            "business_value": story.business_value,
            "scope": story.scope.model_dump(mode="json"),
            "acceptance_criteria": [
                criterion.model_dump(mode="json") for criterion in story.acceptance_criteria
            ],
            "business_rules": [rule.model_dump(mode="json") for rule in story.business_rules],
            "test_scenario_hints": [
                scenario.model_dump(mode="json") for scenario in story.test_scenarios
            ],
            "definition_of_done": list(story.definition_of_done),
        },
        ensure_ascii=False,
        indent=2,
    )
    linked_requirement = requirement_text or (
        "(requirement text unavailable; rely on the user story and retrieved context)"
    )
    if context.chunks:
        context_block = "\n".join(
            f"[{index}] {chunk.text}" for index, chunk in enumerate(context.chunks, start=1)
        )
    else:
        context_block = (
            "(no additional retrieved context; derive strictly from the user story and "
            "linked requirement)"
        )

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
    story_index: int,
    attempt: int,
) -> str | None:
    persister = getattr(reasoning_model, "persist_last_response", None)
    if not callable(persister):
        return _last_response_path(reasoning_model)
    path = persister(filename=f"llm_response_ts_{story_index}_{attempt}.txt")
    return str(path) if path is not None else None


def _last_response_path(reasoning_model: ReasoningModel) -> str | None:
    response_path = getattr(reasoning_model, "last_response_path", None)
    return str(response_path) if response_path else None
