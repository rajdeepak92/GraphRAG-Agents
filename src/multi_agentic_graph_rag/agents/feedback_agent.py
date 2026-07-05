"""Human-feedback gate agent and public runner (standalone HFIL stage).

The gate is feedback LLM call 1: it decides whether a reviewer comment describes a
*new*, document-grounded user story / test scenario, and must cite supporting chunk
ids drawn strictly from the retrieved context (the feedback analogue of requirement
discovery's quote-trace, verified closed-world). Generation itself (call 2) reuses the
existing user-story / test-scenario generation agents via the feedback workflow.
"""

from __future__ import annotations

import json
from typing import Any

from multi_agentic_graph_rag.domain.errors import FeedbackValidationError, ModelOutputError
from multi_agentic_graph_rag.domain.schemas import (
    FeedbackGateOutput,
    FeedbackRequest,
    FeedbackResult,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.observability.session import RunSession
from multi_agentic_graph_rag.services.retrieval import RetrievedContext


class FeedbackGateAgent:
    """One strict gate prompt per feedback request, validated with a single fed-back retry."""

    def __init__(self, reasoning_model: ReasoningModel, *, logger: Any | None = None) -> None:
        self.reasoning_model = reasoning_model
        self.logger = logger

    def gate(
        self,
        *,
        comment: str,
        anchor_requirement_text: str,
        anchor_story_text: str | None,
        context: RetrievedContext,
        gate_index: int = 1,
    ) -> FeedbackGateOutput:
        allowed_ids = [chunk.chunk_id for chunk in context.chunks]
        allowed_set = set(allowed_ids)
        validation_error: str | None = None
        try:
            for attempt in (1, 2):
                prompt = _build_gate_prompt(
                    comment=comment,
                    anchor_requirement_text=anchor_requirement_text,
                    anchor_story_text=anchor_story_text,
                    context=context,
                    validation_error=validation_error,
                )
                _set_response_context(
                    self.reasoning_model,
                    batch_index=gate_index,
                    attempt=attempt,
                    chunk_ids=allowed_ids,
                )
                output = self.reasoning_model.generate_structured(
                    prompt=prompt,
                    schema=FeedbackGateOutput,
                )
                try:
                    _verify_gate_output(output, allowed_set)
                except FeedbackValidationError as error:
                    response_path = _persist_last_response(
                        self.reasoning_model, gate_index=gate_index, attempt=attempt
                    )
                    self._log_validation_failure(
                        attempt=attempt, error=error, response_path=response_path
                    )
                    if attempt == 2:
                        raise ModelOutputError(
                            "Feedback gate failed closed-world validation after retry: "
                            f"{error}; raw_response_path={response_path}"
                        ) from error
                    validation_error = str(error)
                    continue
                return output
        finally:
            _clear_response_context(self.reasoning_model)

        raise ModelOutputError("Feedback gate did not produce a result")

    def _log_validation_failure(
        self,
        *,
        attempt: int,
        error: FeedbackValidationError,
        response_path: str | None,
    ) -> None:
        if self.logger is None:
            return
        self.logger.warning(
            "Feedback gate failed validation",
            step="feedback.gate_validate",
            attempt=attempt,
            retry_count=attempt - 1,
            error=str(error),
            raw_response_path=response_path,
            status="failed" if attempt == 2 else "retrying",
        )


class FeedbackAgent:
    """Public standalone feedback stage agent (mirrors ``UserStoryGeneratorAgent``)."""

    def run(
        self,
        request: FeedbackRequest,
        *,
        session: RunSession | None = None,
    ) -> FeedbackResult:
        from multi_agentic_graph_rag.workflows.feedback_graph import run_feedback

        return run_feedback(request, session=session)


def _verify_gate_output(output: FeedbackGateOutput, allowed_ids: set[str]) -> None:
    # Closed-world: every cited chunk id must be one of the retrieved chunk ids.
    unknown = [cid for cid in output.supporting_chunk_ids if cid not in allowed_ids]
    if unknown:
        raise FeedbackValidationError(
            "gate cited chunk ids outside the retrieved context: " + ", ".join(sorted(unknown))
        )
    if output.verdict == "approve" and not output.supporting_chunk_ids:
        raise FeedbackValidationError(
            "gate approved without citing any supporting chunk id (approve requires evidence)"
        )


def _build_gate_prompt(
    *,
    comment: str,
    anchor_requirement_text: str,
    anchor_story_text: str | None,
    context: RetrievedContext,
    validation_error: str | None,
) -> str:
    if context.chunks:
        context_block = "\n".join(
            f"[{index}] (chunk_id={chunk.chunk_id}) {chunk.text}"
            for index, chunk in enumerate(context.chunks, start=1)
        )
    else:
        context_block = "(no retrieved context)"
    allowed_ids = json.dumps([chunk.chunk_id for chunk in context.chunks], ensure_ascii=False)
    story_block = (
        f"Anchor user story:\n{anchor_story_text}\n\n" if anchor_story_text is not None else ""
    )
    feedback = ""
    if validation_error:
        feedback = (
            "Previous output failed validation. Return one corrected JSON object only.\n"
            "supporting_chunk_ids must be a subset of the allowed chunk ids, and an "
            "'approve' verdict must cite at least one chunk id.\n"
            f"Validation error: {validation_error}\n\n"
        )
    return (
        "You are a requirements reviewer gate deciding whether a human comment describes a "
        "NEW, document-grounded work item to add.\n"
        "Return exactly one valid JSON object and no other text. Do not include markdown, "
        "code fences, commentary, XML tags, hidden reasoning, or explanations.\n\n"
        f"{feedback}"
        "Output schema:\n"
        "{\n"
        '  "verdict": "approve | decline",\n'
        '  "reason": "...",\n'
        '  "supporting_chunk_ids": ["CHUNK-...."]\n'
        "}\n\n"
        "Rules:\n"
        "Approve ONLY when the comment asks for a new item that is supported by the "
        "retrieved context below and is not already covered by the anchor.\n"
        "Decline when the request is destructive, ungrounded, or duplicates existing work.\n"
        "supporting_chunk_ids MUST be chosen only from these allowed chunk ids: "
        f"{allowed_ids}\n"
        "An 'approve' verdict must cite at least one supporting chunk id. reason must be a "
        "complete sentence.\n\n"
        f"Reviewer comment:\n{comment}\n\n"
        f"Anchor requirement:\n{anchor_requirement_text}\n\n"
        f"{story_block}"
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
    gate_index: int,
    attempt: int,
) -> str | None:
    persister = getattr(reasoning_model, "persist_last_response", None)
    if not callable(persister):
        response_path = getattr(reasoning_model, "last_response_path", None)
        return str(response_path) if response_path else None
    path = persister(filename=f"llm_response_feedback_gate_{gate_index}_{attempt}.txt")
    return str(path) if path is not None else None
