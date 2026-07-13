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
    AzureOpenAIEmbeddingModel,
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


class AzureOpenAIEmbeddingModelTests(unittest.TestCase):
    def test_client_is_constructed_once_and_fingerprint_is_unchanged(self) -> None:
        model, embeddings, client_kwargs, encoding_names = _embedding_model(
            lambda batch, call_index: _embedding_response(
                [(index, [float(call_index), float(index)]) for index in range(len(batch))]
            )
        )

        first = model.embed_documents(["first"])
        second = model.embed_documents(["second"])

        self.assertEqual(len(client_kwargs), 1)
        self.assertEqual(len(embeddings.calls), 2)
        self.assertEqual(encoding_names, ["cl100k_base"])
        self.assertEqual(model.provider_name, "azure_openai")
        self.assertEqual(model.embedding_fingerprint, "azure:embedding-deployment")
        self.assertEqual(first, [[0.0, 0.0]])
        self.assertEqual(second, [[1.0, 0.0]])

    def test_item_limit_splitting_preserves_global_order(self) -> None:
        texts = [f"item-{index}" for index in range(2_050)]
        model, embeddings, _, _ = _embedding_model(
            lambda batch, call_index: _embedding_response(
                [(index, [float(text.rsplit("-", 1)[1])]) for index, text in enumerate(batch)]
            )
        )

        vectors = model.embed_documents(texts)

        self.assertEqual([len(call["input"]) for call in embeddings.calls], [2_048, 2])
        self.assertEqual(vectors, [[float(index)] for index in range(2_050)])

    def test_aggregate_token_limit_splitting_preserves_order(self) -> None:
        texts = [f"item-{index}" for index in range(40)]
        token_counts = {text: 8_000 for text in texts}
        model, embeddings, _, _ = _embedding_model(
            lambda batch, call_index: _embedding_response(
                [(index, [float(text.rsplit("-", 1)[1])]) for index, text in enumerate(batch)]
            ),
            token_counts=token_counts,
        )

        vectors = model.embed_documents(texts)

        self.assertEqual([len(call["input"]) for call in embeddings.calls], [37, 3])
        self.assertEqual(vectors, [[float(index)] for index in range(40)])

    def test_out_of_order_provider_indices_are_restored(self) -> None:
        model, _, _, _ = _embedding_model(
            lambda batch, call_index: _embedding_response([(1, [2.0, 0.0]), (0, [1.0, 0.0])])
        )

        vectors = model.embed_documents(["first", "second"])

        self.assertEqual(vectors, [[1.0, 0.0], [2.0, 0.0]])

    def test_malformed_provider_responses_fail_closed(self) -> None:
        malformed = {
            "missing_data": SimpleNamespace(),
            "wrong_length": _embedding_response([(0, [1.0])]),
            "duplicate_index": _embedding_response([(0, [1.0]), (0, [2.0])]),
            "out_of_range": _embedding_response([(0, [1.0]), (2, [2.0])]),
            "missing_index": _embedding_response([(0, [1.0]), (None, [2.0])]),
            "empty_vector": _embedding_response([(0, [1.0]), (1, [])]),
            "wrong_dimension": _embedding_response([(0, [1.0]), (1, [2.0, 3.0])]),
        }
        for name, response in malformed.items():
            with self.subTest(name=name):
                model, _, _, _ = _embedding_model(
                    lambda batch, call_index, response=response: response
                )

                with self.assertRaises(ModelOutputError):
                    model.embed_documents(["first", "second"])

    def test_empty_input_makes_no_provider_call(self) -> None:
        model, embeddings, _, _ = _embedding_model(
            lambda batch, call_index: _embedding_response([])
        )

        self.assertEqual(model.embed_documents([]), [])
        self.assertEqual(embeddings.calls, [])

    def test_oversized_input_is_rejected_before_any_provider_call(self) -> None:
        model, embeddings, _, _ = _embedding_model(
            lambda batch, call_index: _embedding_response([]),
            token_counts={"valid": 1, "oversized": 8_193},
        )

        with self.assertRaisesRegex(ValueError, "index 1.*per-input token limit"):
            model.embed_documents(["valid", "oversized"])

        self.assertEqual(embeddings.calls, [])

    def test_logging_contains_only_sanitized_request_counts(self) -> None:
        logger = _FakeEmbeddingLogger()
        model, _, _, _ = _embedding_model(
            lambda batch, call_index: _embedding_response([(0, [1.0])]),
            logger=logger,
        )

        model.embed_documents(["secret source text"])

        self.assertEqual(len(logger.debugs), 1)
        record = logger.debugs[0]
        self.assertEqual(record["input_count"], 1)
        self.assertEqual(record["batch_count"], 1)
        self.assertEqual(record["provider_call_count"], 1)
        self.assertNotIn("secret source text", str(record))


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


class _FakeEmbeddingLogger:
    def __init__(self) -> None:
        self.debugs: list[dict[str, Any]] = []

    def debug(self, message: str, **kwargs: Any) -> None:
        self.debugs.append({"message": message, **kwargs})


class _FakeEncoding:
    def __init__(self, token_counts: dict[str, int]) -> None:
        self._token_counts = token_counts

    def encode(self, text: str, *, disallowed_special: tuple[()] = ()) -> list[int]:
        return [0] * self._token_counts.get(text, 1)


class _FakeEmbeddings:
    def __init__(self, responder: Any) -> None:
        self._responder = responder
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self._responder(kwargs["input"], len(self.calls) - 1)


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


def _embedding_response(indexed_vectors: list[tuple[int | None, list[float]]]) -> Any:
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in indexed_vectors]
    )


def _embedding_model(
    responder: Any,
    *,
    token_counts: dict[str, int] | None = None,
    logger: Any | None = None,
) -> tuple[AzureOpenAIEmbeddingModel, _FakeEmbeddings, list[dict[str, Any]], list[str]]:
    settings = AzureOpenAISettings(
        endpoint="https://example.openai.azure.com",
        api_key="key",
        embedding_deployment="embedding-deployment",
    )
    embeddings = _FakeEmbeddings(responder)
    client = SimpleNamespace(embeddings=embeddings)
    client_kwargs: list[dict[str, Any]] = []
    encoding_names: list[str] = []

    def _client_factory(**kwargs: Any) -> Any:
        client_kwargs.append(dict(kwargs))
        return client

    def _get_encoding(name: str) -> _FakeEncoding:
        encoding_names.append(name)
        return _FakeEncoding(token_counts or {})

    modules = {
        "openai": SimpleNamespace(AzureOpenAI=_client_factory),
        "tiktoken": SimpleNamespace(get_encoding=_get_encoding),
    }
    with patch(
        "multi_agentic_graph_rag.llm_models.azure_openai.import_module",
        side_effect=lambda name: modules[name],
    ):
        model = AzureOpenAIEmbeddingModel(settings, logger=logger)
    return model, embeddings, client_kwargs, encoding_names


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
