"""Structured Stage-4 patch producer using one explicitly selected model.

This adapter never constructs a provider and therefore cannot fall back to one.
It is the single retry authority for Stage-4 model calls: every underlying call
uses ``max_attempts=1`` and the adapter performs at most one same-instance retry.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from multi_agentic_graph_rag.common_prompt_defs import PromptStage4CodeGeneration
from multi_agentic_graph_rag.domain.codegen_schemas import (
    GeneratedPatchBundle,
    ProviderFingerprint,
    TestCaseImplementationPlan,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.errors import (
    AzureOpenAIModelFailure,
    ConfigurationError,
    HuggingFaceModelFailure,
    InputManifestChanged,
)
from multi_agentic_graph_rag.llm_models.ports import ReasoningModel
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary
from multi_agentic_graph_rag.services.patch_producer import FileOp, PatchRequest

T = TypeVar("T", bound=BaseModel)
_MODEL_ATTEMPTS = 2
_MAX_DIAGNOSTICS = 20
_MAX_DIAGNOSTIC_CHARS = 2_000


class LLMPatchProducer:
    """Plan, generate, and repair through one pinned reasoning-model instance."""

    def __init__(
        self,
        *,
        model: ReasoningModel,
        provider_fingerprint: ProviderFingerprint,
        resume_fingerprint: ProviderFingerprint | None = None,
        logger: Any | None = None,
    ) -> None:
        if model.provider_name != provider_fingerprint.provider:
            raise ConfigurationError(
                "selected reasoning model does not match the Stage-4 provider fingerprint"
            )
        if provider_fingerprint.provider not in {"azure_openai", "huggingface"}:
            raise ConfigurationError(
                f"unsupported Stage-4 reasoning provider: {provider_fingerprint.provider}"
            )
        if resume_fingerprint is not None and not provider_fingerprint.matches(resume_fingerprint):
            raise InputManifestChanged(
                "Stage-4 provider fingerprint changed; start a new run instead of resuming"
            )
        self._model = model
        self.provider_fingerprint = provider_fingerprint
        self._logger = logger
        self._retry_events: list[dict[str, Any]] = []

    @property
    def retry_events(self) -> tuple[dict[str, Any], ...]:
        """Sanitized retry events emitted by this producer."""

        return tuple(dict(event) for event in self._retry_events)

    def assert_fingerprint(self, expected: ProviderFingerprint) -> None:
        """Reject a resume using a different provider/model/prompt pin."""

        if not self.provider_fingerprint.matches(expected):
            raise InputManifestChanged(
                "Stage-4 provider fingerprint changed; start a new run instead of resuming"
            )

    def plan(self, request: PatchRequest) -> TestCaseImplementationPlan:
        payload = request.prompt_payload()
        prompt = _prompt("Plan this frozen test case.", payload)
        return self._structured_call(
            prompt=prompt,
            schema=TestCaseImplementationPlan,
            system_message=PromptStage4CodeGeneration.PLAN_SYSTEM.value,
            operation="stage4.plan",
            request_id=_request_id(request, "plan"),
            validate=lambda result: _validate_plan(result, request),
        )

    def generate_bundle(
        self,
        request: PatchRequest,
        plan: TestCaseImplementationPlan,
    ) -> GeneratedPatchBundle:
        tc_id = _require_tc_id(request)
        plan_checksum = canonical_checksum(plan)
        prompt = _prompt(
            "Generate the complete multi-file bundle for this fixed plan.",
            {
                "request": request.prompt_payload(),
                "implementation_plan": plan.model_dump(mode="json"),
                "required_tc_id": tc_id,
                "required_plan_checksum": plan_checksum,
            },
        )
        return self._structured_call(
            prompt=prompt,
            schema=GeneratedPatchBundle,
            system_message=PromptStage4CodeGeneration.BUNDLE_SYSTEM.value,
            operation="stage4.bundle",
            request_id=_request_id(request, "bundle"),
            validate=lambda result: _validate_bundle(
                result,
                request=request,
                plan=plan,
                expected_plan_checksum=plan_checksum,
            ),
        )

    def repair_bundle(
        self,
        request: PatchRequest,
        plan: TestCaseImplementationPlan,
        current_bundle: GeneratedPatchBundle,
        diagnostics: tuple[str, ...],
    ) -> GeneratedPatchBundle:
        plan_checksum = canonical_checksum(plan)
        bounded = _bounded_diagnostics((*request.repair_diagnostics, *diagnostics))
        prompt = _prompt(
            "Repair only the current bundle using these bounded diagnostics.",
            {
                "request": request.prompt_payload(),
                "implementation_plan": plan.model_dump(mode="json"),
                "current_bundle": current_bundle.model_dump(mode="json"),
                "validation_diagnostics": bounded,
                "required_plan_checksum": plan_checksum,
            },
        )
        return self._structured_call(
            prompt=prompt,
            schema=GeneratedPatchBundle,
            system_message=PromptStage4CodeGeneration.REPAIR_SYSTEM.value,
            operation="stage4.repair",
            request_id=_request_id(request, "repair"),
            validate=lambda result: _validate_bundle(
                result,
                request=request,
                plan=plan,
                expected_plan_checksum=plan_checksum,
            ),
        )

    def produce(self, request: PatchRequest) -> list[FileOp]:
        """Compatibility bridge for the superseded complete-content apply loop."""

        plan = self.plan(request)
        bundle = self.generate_bundle(request, plan)
        return [
            FileOp(kind="create", relative_path=item.relative_path, content=item.content)
            for item in bundle.files
        ]

    def _structured_call(
        self,
        *,
        prompt: str,
        schema: type[T],
        system_message: str,
        operation: str,
        request_id: str,
        validate: Callable[[T], None],
    ) -> T:
        last_error: Exception | None = None
        for attempt in range(1, _MODEL_ATTEMPTS + 1):
            try:
                result = self._model.generate_structured(
                    prompt=prompt,
                    schema=schema,
                    system_message=system_message,
                    operation=operation,
                    request_id=request_id,
                    max_attempts=1,
                )
                validate(result)
                return result
            except Exception as exc:
                last_error = exc
                if attempt == 1:
                    self._emit_retry(operation=operation, request_id=request_id, error=exc)
                    continue
                raise self._provider_failure(exc) from exc
        raise AssertionError(f"unreachable model retry state: {last_error!r}")

    def _emit_retry(self, *, operation: str, request_id: str, error: Exception) -> None:
        event: dict[str, Any] = {
            "event": "MODEL_REQUEST_RETRYING",
            "operation": operation,
            "request_id": request_id,
            "attempt": 1,
            "remaining_retries": 1,
            "detail": "1 retry remaining",
            "provider": self.provider_fingerprint.provider,
            "error_summary": sanitized_exception_summary(error),
        }
        self._retry_events.append(event)
        warning = getattr(self._logger, "warning", None)
        if not callable(warning):
            return
        try:
            warning("MODEL_REQUEST_RETRYING", **{k: v for k, v in event.items() if k != "event"})
        except TypeError:
            warning(
                "MODEL_REQUEST_RETRYING provider=%s operation=%s 1 retry remaining",
                self.provider_fingerprint.provider,
                operation,
            )

    def _provider_failure(self, error: Exception) -> Exception:
        diagnostic = sanitized_exception_summary(error)
        if self.provider_fingerprint.provider == "azure_openai":
            return AzureOpenAIModelFailure(
                "Azure OpenAI model request failed after two attempts. Correct the Azure "
                f"deployment, endpoint, authentication, or service issue and retry the run. "
                f"Sanitized diagnostic: {diagnostic}"
            )
        return HuggingFaceModelFailure(
            "Hugging Face model request failed after two attempts. Correct the pinned private "
            "model installation/configuration and retry the run. Sanitized diagnostic: "
            f"{diagnostic}"
        )


def _prompt(instruction: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"{instruction}\n\nINPUT_JSON:\n{encoded}"


def _request_id(request: PatchRequest, operation: str) -> str:
    return f"{request.scenario_id}:{request.variant_id}:{request.tc_id or 'unreserved'}:{operation}"


def _require_tc_id(request: PatchRequest) -> int:
    if request.tc_id is None:
        raise ConfigurationError("a permanent tc_id must be reserved before bundle generation")
    return request.tc_id


def _validate_plan(result: TestCaseImplementationPlan, request: PatchRequest) -> None:
    if result.scenario_id != request.scenario_id:
        raise ValueError("implementation plan scenario_id does not match the frozen request")


def _validate_bundle(
    result: GeneratedPatchBundle,
    *,
    request: PatchRequest,
    plan: TestCaseImplementationPlan,
    expected_plan_checksum: str,
) -> None:
    if result.scenario_id != request.scenario_id:
        raise ValueError("generated bundle scenario_id does not match the frozen request")
    if result.tc_id != _require_tc_id(request):
        raise ValueError("generated bundle tc_id does not match the permanent reservation")
    if result.plan_checksum != expected_plan_checksum:
        raise ValueError("generated bundle plan_checksum does not match the fixed plan")
    planned_paths = {item.relative_path for item in plan.files if not item.reuse_existing}
    generated_paths = {item.relative_path for item in result.files}
    if planned_paths != generated_paths:
        raise ValueError("generated bundle paths do not match the fixed implementation plan")


def _bounded_diagnostics(values: tuple[str, ...]) -> list[str]:
    return [value[:_MAX_DIAGNOSTIC_CHARS] for value in values[:_MAX_DIAGNOSTICS]]


__all__ = ["LLMPatchProducer"]
