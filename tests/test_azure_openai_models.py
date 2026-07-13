from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from multi_agentic_graph_rag.config.settings import AzureOpenAISettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import RequirementDiscoveryChunkOutput
from multi_agentic_graph_rag.llm_models.azure_openai import AzureOpenAIReasoningModel


class AzureOpenAIModelTests(unittest.TestCase):
    def test_reasoning_temperature_is_sent_when_configured_and_supported(self) -> None:
        model, completions, openai = _model_with_completions(["ok"], temperature=0.0)

        with patch(
            "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
            return_value=openai,
        ):
            self.assertEqual(model._generate_completion("prompt"), "ok")

        self.assertEqual(len(completions.calls), 1)
        self.assertEqual(completions.calls[0]["temperature"], 0.0)

    def test_reasoning_temperature_is_omitted_when_configured_none(self) -> None:
        model, completions, openai = _model_with_completions(["ok"], temperature=None)

        with patch(
            "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
            return_value=openai,
        ):
            self.assertEqual(model._generate_completion("prompt"), "ok")

        self.assertEqual(len(completions.calls), 1)
        self.assertNotIn("temperature", completions.calls[0])

    def test_unsupported_temperature_is_retried_once_without_parameter(self) -> None:
        unsupported = _FakeBadRequestError(
            param="temperature",
            code="unsupported_value",
        )
        model, completions, openai = _model_with_completions([unsupported, "ok"], temperature=0.0)

        with patch(
            "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
            return_value=openai,
        ):
            self.assertEqual(model._generate_completion("prompt"), "ok")

        self.assertEqual(len(completions.calls), 2)
        self.assertEqual(completions.calls[0]["temperature"], 0.0)
        self.assertNotIn("temperature", completions.calls[1])

    def test_unrelated_bad_request_is_propagated_without_retry(self) -> None:
        unrelated = _FakeBadRequestError(param="messages", code="invalid_value")
        model, completions, openai = _model_with_completions([unrelated], temperature=0.0)

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaises(_FakeBadRequestError) as raised,
        ):
            model._generate_completion("prompt")

        self.assertIs(raised.exception, unrelated)
        self.assertEqual(len(completions.calls), 1)

    def test_temperature_fallback_has_no_unbounded_retry(self) -> None:
        first = _FakeBadRequestError(param="temperature", code="unsupported_value")
        second = _FakeBadRequestError(param="temperature", code="unsupported_value")
        model, completions, openai = _model_with_completions([first, second], temperature=0.0)

        with (
            patch(
                "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
                return_value=openai,
            ),
            self.assertRaises(_FakeBadRequestError) as raised,
        ):
            model._generate_completion("prompt")

        self.assertIs(raised.exception, second)
        self.assertEqual(len(completions.calls), 2)

    def test_failed_reasoning_output_is_persisted_per_parse_attempt(self) -> None:
        settings = AzureOpenAISettings(
            endpoint="https://example.openai.azure.com",
            api_key="key",
            reasoning_deployment="deployment",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model = _FailingAzureReasoningModel(
                settings,
                discovery_batch_size=1,
                run_dir=Path(temp_dir),
            )
            model.set_response_context(batch_index=3, attempt=1, chunk_ids=["CHUNK-3"])

            with self.assertRaises(ModelOutputError):
                model.generate_structured(
                    prompt="extract requirements",
                    schema=RequirementDiscoveryChunkOutput,
                )

            first = Path(temp_dir) / "llm_response_3_1.txt"
            second = Path(temp_dir) / "llm_response_3_1_parse2.txt"
            self.assertEqual(first.read_text(encoding="utf-8"), '{"facts": [')
            self.assertEqual(second.read_text(encoding="utf-8"), '{"facts": [')


class _FailingAzureReasoningModel(AzureOpenAIReasoningModel):
    def _generate_completion(self, prompt: str) -> str:
        return '{"facts": ['


class _FakeBadRequestError(Exception):
    def __init__(self, *, param: str, code: str) -> None:
        super().__init__(f"{param}: {code}")
        self.param = param
        self.code = code
        self.body = {"error": {"param": param, "code": code}}


class _FakeCompletions:
    def __init__(self, results: list[str | Exception]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=result))])


def _model_with_completions(
    results: list[str | Exception],
    *,
    temperature: float | None,
) -> tuple[AzureOpenAIReasoningModel, _FakeCompletions, Any]:
    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com",
        api_key="key",
        reasoning_deployment="deployment",
        reasoning_temperature=temperature,
    )
    completions = _FakeCompletions(results)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    openai = SimpleNamespace(
        AzureOpenAI=lambda **_kwargs: client,
        BadRequestError=_FakeBadRequestError,
    )
    return AzureOpenAIReasoningModel(settings), completions, openai


if __name__ == "__main__":
    unittest.main()
