from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from multi_agentic_graph_rag.config.settings import AzureOpenAISettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError, ModelOutputError
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalScenario,
    DuplicateJudgeResult,
    KnowledgeExtractionChunkOutput,
    RequirementDeltaDecision,
    RequirementDiscoveryChunkOutput,
    TestScenarioGenerationOutput,
    UserStoryGenerationOutput,
)
from multi_agentic_graph_rag.llm_models.azure_openai import (
    _AZURE_UNSUPPORTED_SCHEMA_KEYWORDS,
    AzureOpenAIReasoningModel,
    _strict_response_format,
)
from multi_agentic_graph_rag.services.requirement_memory import (
    _BidirectionalEntailmentOutput,
)


class AzureOpenAIModelTests(unittest.TestCase):
    def test_unsupported_structured_outputs_api_fails_readiness_clearly(self) -> None:
        model = AzureOpenAIReasoningModel(
            AzureOpenAISettings(
                endpoint="https://example.openai.azure.com",
                api_key="key",
                api_version="2024-02-15-preview",
                reasoning_deployment="legacy-deployment",
            )
        )

        with self.assertRaisesRegex(
            ConfigurationError,
            "legacy-deployment.*2024-02-15-preview.*strict=true",
        ):
            model.warmup()

    def test_all_production_response_schemas_fit_azure_supported_subset(self) -> None:
        schemas = (
            RequirementDiscoveryChunkOutput,
            UserStoryGenerationOutput,
            TestScenarioGenerationOutput,
            KnowledgeExtractionChunkOutput,
            RequirementDeltaDecision,
            CanonicalScenario,
            DuplicateJudgeResult,
            _BidirectionalEntailmentOutput,
        )

        for schema in schemas:
            with self.subTest(schema=schema.__name__):
                response_format = _strict_response_format(schema)
                strict_schema = response_format["json_schema"]["schema"]
                for keyword in _AZURE_UNSUPPORTED_SCHEMA_KEYWORDS:
                    self.assertFalse(_contains_key(strict_schema, keyword))

    def test_native_structured_outputs_supply_schema_and_omit_temperature(self) -> None:
        result = _response(content='{"facts":[]}')
        model, completions, client_kwargs, openai = _model_with_completions([result])

        with patch(
            "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
            return_value=openai,
        ):
            parsed = model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="discovery system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(parsed.facts, [])
        self.assertEqual(len(completions.calls), 1)
        request = completions.calls[0]
        response_format = request["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        strict_schema = response_format["json_schema"]["schema"]
        self.assertFalse(_contains_key(strict_schema, "minimum"))
        self.assertFalse(_contains_key(strict_schema, "maximum"))
        self.assertFalse(_contains_key(strict_schema, "default"))
        self.assertEqual(request["messages"][0]["content"], "discovery system")
        self.assertNotIn("temperature", request)
        self.assertEqual(client_kwargs[0]["max_retries"], 1)

    def test_attempt_limit_bounds_transport_retries(self) -> None:
        result = _response(content='{"facts":[]}')
        model, _, client_kwargs, openai = _model_with_completions([result])

        with patch(
            "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
            return_value=openai,
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
                max_attempts=1,
            )

        self.assertEqual(client_kwargs[0]["max_retries"], 0)

    def test_refusal_is_not_retried(self) -> None:
        result = _response(content="", refusal="refused")
        model, completions, _, openai = _model_with_completions([result])

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaisesRegex(ModelOutputError, "refused"),
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(len(completions.calls), 1)

    def test_content_filtered_response_is_not_retried(self) -> None:
        result = _response(content=None, finish_reason="content_filter")
        model, completions, _, openai = _model_with_completions([result])

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaisesRegex(ModelOutputError, "content filtering"),
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(len(completions.calls), 1)

    def test_missing_structured_content_is_not_retried(self) -> None:
        result = _response(content=None)
        model, completions, _, openai = _model_with_completions([result])

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaisesRegex(ModelOutputError, "no parsed structured response content"),
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(len(completions.calls), 1)

    def test_provider_rejection_of_strict_outputs_is_configuration_error(self) -> None:
        model, completions, _, openai = _model_with_completions(
            [_FakeBadRequestError("response_format is not supported")]
        )

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaisesRegex(ConfigurationError, "deployment.*strict=true"),
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(len(completions.calls), 1)

    def test_invalid_structured_output_is_not_schema_retried(self) -> None:
        result = _response(content='{"facts":"invalid"}')
        model, completions, _, openai = _model_with_completions([result])

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaisesRegex(ModelOutputError, "Python schema"),
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(len(completions.calls), 1)

    def test_python_validation_rejects_invalid_parsed_payload(self) -> None:
        result = _response(content='{"facts":[],"unexpected":true}')
        model, completions, _, openai = _model_with_completions([result])

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaisesRegex(ModelOutputError, "Python schema"),
        ):
            model.generate_structured(
                prompt="prompt",
                schema=RequirementDiscoveryChunkOutput,
                system_message="system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(len(completions.calls), 1)

    def test_diagnostic_files_are_unique_for_repeated_request_identifiers(self) -> None:
        results = [_response(content='{"facts":[]}'), _response(content='{"facts":[]}')]
        with tempfile.TemporaryDirectory() as temp_dir:
            model, _, _, openai = _model_with_completions(
                results,
                run_dir=Path(temp_dir),
                log_llm_responses=True,
            )
            model.set_response_context(batch_index=13, attempt=2, chunk_ids=["CHUNK-13"])

            with patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ):
                model.generate_structured(
                    prompt="prompt",
                    schema=RequirementDiscoveryChunkOutput,
                    system_message="identity system",
                    operation="requirement_identity.entailment",
                    request_id="pair-1",
                )
                model.generate_structured(
                    prompt="prompt",
                    schema=RequirementDiscoveryChunkOutput,
                    system_message="identity system",
                    operation="requirement_identity.entailment",
                    request_id="pair-1",
                )

            paths = sorted(
                Path(temp_dir).glob(
                    "llm_response_requirement_identity.entailment_pair-1_call-*_attempt-1.txt"
                )
            )
            self.assertEqual(len(paths), 2)
            self.assertNotEqual(paths[0], paths[1])
            self.assertTrue(
                all(path.read_text(encoding="utf-8") == '{"facts":[]}' for path in paths)
            )


class _FakeStructuredCompletions:
    def __init__(self, results: list[Any | Exception]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _FakeBadRequestError(Exception):
    """Stand in for the SDK's non-retryable strict-output capability error."""


class _FakeNotFoundError(Exception):
    """Stand in for a missing Azure deployment configuration error."""


def _response(
    *,
    content: str | None,
    refusal: str | None = None,
    finish_reason: str = "stop",
) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, refusal=refusal),
                finish_reason=finish_reason,
            )
        ]
    )


def _model_with_completions(
    results: list[Any | Exception],
    *,
    run_dir: Path | None = None,
    log_llm_responses: bool = False,
) -> tuple[AzureOpenAIReasoningModel, _FakeStructuredCompletions, list[dict[str, Any]], Any]:
    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com",
        api_key="key",
        reasoning_deployment="deployment",
        reasoning_temperature=0.0,
    )
    completions = _FakeStructuredCompletions(results)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    client_kwargs: list[dict[str, Any]] = []

    def _client_factory(**kwargs: Any) -> Any:
        client_kwargs.append(dict(kwargs))
        return client

    openai = SimpleNamespace(
        AzureOpenAI=_client_factory,
        BadRequestError=_FakeBadRequestError,
        NotFoundError=_FakeNotFoundError,
    )
    model = AzureOpenAIReasoningModel(
        settings,
        run_dir=run_dir,
        log_llm_responses=log_llm_responses,
    )
    return model, completions, client_kwargs, openai


def _contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(child, target) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, target) for child in value)
    return False


if __name__ == "__main__":
    unittest.main()
