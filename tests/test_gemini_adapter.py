"""Google Gemini adapter unit tests (SDK mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from multi_agentic_graph_rag.common_prompt_defs import PromptSharedFragments
from multi_agentic_graph_rag.config.settings import GeminiSettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError, MalformedModelOutputError
from multi_agentic_graph_rag.llm_models import gemini


class _Schema(BaseModel):
    value: int


class _FakeConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeTypes:
    GenerateContentConfig = _FakeConfig
    EmbedContentConfig = _FakeConfig


class _FakeModels:
    def __init__(
        self,
        *,
        gen_responses: list[Any] | None = None,
        embed_response: Any | None = None,
    ) -> None:
        self._gen = list(gen_responses or [])
        self._embed = embed_response
        self.generate_calls: list[SimpleNamespace] = []
        self.embed_calls: list[SimpleNamespace] = []

    def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.generate_calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        response = self._gen.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def embed_content(self, *, model: str, contents: list[str], config: Any) -> Any:
        self.embed_calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        return self._embed


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


class _FakeGenAI:
    def __init__(self, client: _FakeClient) -> None:
        self._client = client
        self.api_keys: list[str] = []

    def Client(self, *, api_key: str) -> _FakeClient:
        self.api_keys.append(api_key)
        return self._client


def _text_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, candidates=[], prompt_feedback=None)


def _install(monkeypatch: pytest.MonkeyPatch, models: _FakeModels) -> _FakeGenAI:
    fake_genai = _FakeGenAI(_FakeClient(models))
    monkeypatch.setattr(gemini, "_load_genai", lambda: (fake_genai, _FakeTypes))
    return fake_genai


def test_generate_structured_returns_validated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _FakeModels(gen_responses=[_text_response('{"value": 7}')])
    _install(monkeypatch, models)
    adapter = gemini.GeminiReasoningModel(GeminiSettings(api_key="k", reasoning_model="m"))

    result = adapter.generate_structured(
        prompt="p",
        schema=_Schema,
        system_message="sys",
        operation="op",
        request_id="rid",
    )

    assert adapter.provider_name == "gemini"
    assert result == _Schema(value=7)
    assert len(models.generate_calls) == 1
    assert models.generate_calls[0].model == "m"
    assert models.generate_calls[0].config.kwargs["response_json_schema"] == (
        _Schema.model_json_schema()
    )
    assert "response_schema" not in models.generate_calls[0].config.kwargs
    assert models.generate_calls[0].config.kwargs["system_instruction"] == "sys"
    assert models.generate_calls[0].config.kwargs["temperature"] == 0.0


def test_generate_structured_repairs_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _FakeModels(gen_responses=[_text_response("not json"), _text_response('{"value": 3}')])
    _install(monkeypatch, models)
    adapter = gemini.GeminiReasoningModel(GeminiSettings(api_key="k", reasoning_model="m"))

    result = adapter.generate_structured(
        prompt="p",
        schema=_Schema,
        system_message="sys",
        operation="op",
        request_id="rid",
    )

    assert result == _Schema(value=3)
    assert len(models.generate_calls) == 2
    assert PromptSharedFragments.CORRECTED_JSON_ONLY.value in models.generate_calls[1].contents


def test_generate_structured_raises_after_persistent_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = _FakeModels(
        gen_responses=[_text_response("not json"), _text_response("still not json")]
    )
    _install(monkeypatch, models)
    adapter = gemini.GeminiReasoningModel(GeminiSettings(api_key="k", reasoning_model="m"))

    with pytest.raises(MalformedModelOutputError):
        adapter.generate_structured(
            prompt="p",
            schema=_Schema,
            system_message="sys",
            operation="op",
            request_id="rid",
        )
    assert len(models.generate_calls) == 2


def test_warmup_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _FakeModels())
    adapter = gemini.GeminiReasoningModel(GeminiSettings(api_key="", reasoning_model="m"))

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        adapter.warmup()


def test_generate_structured_reports_quota_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("quota")
    error.code = 429  # type: ignore[attr-defined]
    error.status = "RESOURCE_EXHAUSTED"  # type: ignore[attr-defined]
    models = _FakeModels(gen_responses=[error])
    _install(monkeypatch, models)
    adapter = gemini.GeminiReasoningModel(
        GeminiSettings(api_key="k", reasoning_model="gemini-2.5-pro")
    )

    with pytest.raises(
        ConfigurationError,
        match=r"quota is exhausted.*GEMINI_REASONING_MODEL",
    ):
        adapter.generate_structured(
            prompt="p",
            schema=_Schema,
            system_message="sys",
            operation="op",
            request_id="rid",
        )


def test_embed_documents_returns_ordered_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    embed_response = SimpleNamespace(
        embeddings=[
            SimpleNamespace(values=[1.0, 2.0]),
            SimpleNamespace(values=[3.0, 4.0]),
        ]
    )
    models = _FakeModels(embed_response=embed_response)
    _install(monkeypatch, models)
    adapter = gemini.GeminiEmbeddingModel(GeminiSettings(api_key="k", embedding_model="e"))

    vectors = adapter.embed_documents(["alpha", "beta"])

    assert adapter.provider_name == "gemini"
    assert adapter.embedding_fingerprint == "gemini:e"
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert models.embed_calls[0].model == "e"


def test_embed_documents_empty_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    models = _FakeModels()
    _install(monkeypatch, models)
    adapter = gemini.GeminiEmbeddingModel(GeminiSettings(api_key="k", embedding_model="e"))

    assert adapter.embed_documents([]) == []
    assert models.embed_calls == []
