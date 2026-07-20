"""Stage-4 structured producer isolation, retry, and prompt-contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from multi_agentic_graph_rag.common_prompt_defs import PromptStage4CodeGeneration
from multi_agentic_graph_rag.domain.codegen_schemas import (
    GeneratedFileContent,
    GeneratedPatchBundle,
    PlannedFile,
    PlannedStep,
    ProviderFingerprint,
    TestCaseImplementationPlan,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.errors import (
    AzureOpenAIModelFailure,
    GeminiModelFailure,
    InputManifestChanged,
)
from multi_agentic_graph_rag.domain.identifiers import make_provider_fingerprint_hash
from multi_agentic_graph_rag.services.llm_patch_producer import LLMPatchProducer
from multi_agentic_graph_rag.services.patch_producer import PatchRequest, StaticPatchProducer


class _Model:
    def __init__(self, provider: str, outcomes: list[BaseModel | Exception]) -> None:
        self.provider_name = provider
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[BaseModel],
        system_message: str,
        operation: str,
        request_id: str,
        max_attempts: int = 2,
    ) -> BaseModel:
        self.calls.append(
            {
                "prompt": prompt,
                "schema": schema,
                "system_message": system_message,
                "operation": operation,
                "request_id": request_id,
                "max_attempts": max_attempts,
            }
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Logger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))


def _fingerprint(provider: str = "azure_openai", model: str = "deployment") -> ProviderFingerprint:
    params = {"temperature": 0.0}
    return ProviderFingerprint(
        provider=provider,
        model=model,
        model_revision=None,
        generation_params=params,
        prompt_revision="stage4-v1",
        fingerprint_hash=make_provider_fingerprint_hash(
            provider=provider,
            model=model,
            model_revision=None,
            generation_params_checksum=canonical_checksum({"params": params}),
            prompt_revision="stage4-v1",
        ),
    )


def _request() -> PatchRequest:
    return PatchRequest(
        scenario_id="TS-1",
        execution_profile_id="EP-1",
        resolved_bundle_id="TDB-1",
        context_manifest_id="CTX-1",
        tc_id=100001,
        redacted_bundle={"credential": "secret://vault/device"},
        context={"scenario": {"title": "Validate sensor threshold"}},
    )


def _plan() -> TestCaseImplementationPlan:
    return TestCaseImplementationPlan(
        scenario_id="TS-1",
        module="sensor",
        test_title="Validate Sensor Threshold",
        steps=[PlannedStep(step=1, description="Read threshold", critical=True)],
        files=[
            PlannedFile(
                role="test",
                relative_path="tests/sensor/Tc100001ValidateSensorThreshold.py",
            )
        ],
    )


def _bundle(plan: TestCaseImplementationPlan) -> GeneratedPatchBundle:
    content = "class Tc100001ValidateSensorThreshold:\n    pass\n"
    digest = hashlib.sha256(content.encode()).hexdigest()
    return GeneratedPatchBundle(
        scenario_id="TS-1",
        tc_id=100001,
        plan_checksum=canonical_checksum(plan),
        files=[
            GeneratedFileContent(
                role="test",
                relative_path="tests/sensor/Tc100001ValidateSensorThreshold.py",
                content=content,
                content_hash=f"sha256:{digest}",
            )
        ],
    )


def test_plan_uses_one_selected_model_and_preserves_secret_reference() -> None:
    plan = _plan()
    model = _Model("azure_openai", [plan])
    producer = LLMPatchProducer(model=model, provider_fingerprint=_fingerprint())  # type: ignore[arg-type]

    assert producer.plan(_request()) == plan
    assert len(model.calls) == 1
    assert model.calls[0]["max_attempts"] == 1
    assert "secret://vault/device" in model.calls[0]["prompt"]
    assert model.calls[0]["operation"] == "stage4.plan"


def test_bundle_and_static_producer_use_structured_contract() -> None:
    plan = _plan()
    bundle = _bundle(plan)
    model = _Model("azure_openai", [bundle])
    producer = LLMPatchProducer(model=model, provider_fingerprint=_fingerprint())  # type: ignore[arg-type]

    assert producer.generate_bundle(_request(), plan) == bundle
    static = StaticPatchProducer(implementation_plan=plan, bundle=bundle, repairs=[bundle])
    assert static.plan(_request()) == plan
    assert static.generate_bundle(_request(), plan) == bundle
    assert static.repair_bundle(_request(), plan, bundle, ("syntax",)) == bundle


def test_one_adapter_retry_emits_required_event() -> None:
    logger = _Logger()
    plan = _plan()
    model = _Model("azure_openai", [TimeoutError("temporary token=secret"), plan])
    producer = LLMPatchProducer(
        model=model,  # type: ignore[arg-type]
        provider_fingerprint=_fingerprint(),
        logger=logger,
    )

    assert producer.plan(_request()) == plan
    assert len(model.calls) == 2
    assert {call["max_attempts"] for call in model.calls} == {1}
    assert logger.events[0][0] == "MODEL_REQUEST_RETRYING"
    assert logger.events[0][1]["remaining_retries"] == 1
    assert logger.events[0][1]["detail"] == "1 retry remaining"
    assert producer.retry_events[0]["event"] == "MODEL_REQUEST_RETRYING"


@pytest.mark.parametrize(
    ("provider", "error_type"),
    [
        ("azure_openai", AzureOpenAIModelFailure),
        ("gemini", GeminiModelFailure),
    ],
)
def test_second_failure_is_sanitized_and_provider_specific(
    provider: str,
    error_type: type[Exception],
) -> None:
    model = _Model(
        provider,
        [RuntimeError("credential=super-secret"), RuntimeError("credential=super-secret")],
    )
    producer = LLMPatchProducer(
        model=model,  # type: ignore[arg-type]
        provider_fingerprint=_fingerprint(provider, "private-model"),
    )

    with pytest.raises(error_type) as captured:
        producer.plan(_request())
    assert len(model.calls) == 2
    assert "super-secret" not in str(captured.value)


def test_resume_fingerprint_must_match_before_any_model_call() -> None:
    model = _Model("azure_openai", [_plan()])
    with pytest.raises(InputManifestChanged):
        LLMPatchProducer(
            model=model,  # type: ignore[arg-type]
            provider_fingerprint=_fingerprint(model="deployment-a"),
            resume_fingerprint=_fingerprint(model="deployment-b"),
        )
    assert model.calls == []


def test_producer_source_has_no_provider_factory_or_cross_provider_construction() -> None:
    source = Path("src/multi_agentic_graph_rag/services/llm_patch_producer.py").read_text(
        encoding="utf-8"
    )
    assert "create_reasoning_model" not in source
    assert "AzureOpenAIReasoningModel" not in source
    assert "GeminiReasoningModel" not in source


def test_stage4_prompts_lock_python_robot_data_and_no_production_contracts() -> None:
    combined = "\n".join(item.value for item in PromptStage4CodeGeneration)
    for required in (
        "Never propose a production-code edit",
        "test_lib/<module>/<module>_wrappers.py",
        "test_setup()",
        "bool(step_results) and all(step_results)",
        "calls only Run Test",
        "secret://",
        "Never invoke Git",
    ):
        assert required in combined
