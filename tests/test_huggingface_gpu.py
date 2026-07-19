"""Hugging Face device-selection regression tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import HuggingFaceSettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.llm_models import huggingface


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self.available = available

    def is_available(self) -> bool:
        return self.available

    def is_bf16_supported(self) -> bool:
        return self.available


class _FakeTorch:
    float16 = object()
    bfloat16 = object()

    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(available=cuda_available)


class _FakeFactory:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, model_name: str, **kwargs: Any) -> Any:
        self.calls.append((model_name, kwargs))
        return self.value


class _FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.eval_called = False

    def to(self, device: str) -> _FakeModel:
        self.device = device
        return self

    def eval(self) -> None:
        self.eval_called = True


def test_config_loads_explicit_huggingface_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_DEVICE", "CUDA")
    monkeypatch.setenv("HUGGINGFACE_QUANTIZATION", "BITSANDBYTES_4BIT")
    monkeypatch.setenv("HUGGINGFACE_DISABLE_THINKING", "true")

    settings = load_config().huggingface
    assert settings.device == "cuda"
    assert settings.quantization == "bitsandbytes_4bit"
    assert settings.disable_thinking is True


def test_cuda_device_fails_fast_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(cuda_available=False)
    monkeypatch.setattr(huggingface, "import_module", lambda _name: fake_torch)

    with pytest.raises(ConfigurationError, match="CUDA is unavailable"):
        huggingface._resolve_device("cuda")


def test_reasoning_model_is_loaded_in_fp16_and_moved_to_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(cuda_available=True)
    fake_tokenizer = object()
    fake_model = _FakeModel()
    tokenizer_factory = _FakeFactory(fake_tokenizer)
    model_factory = _FakeFactory(fake_model)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_factory,
        AutoModelForCausalLM=model_factory,
    )

    def fake_import(name: str) -> Any:
        return fake_torch if name == "torch" else fake_transformers

    monkeypatch.setattr(huggingface, "import_module", fake_import)
    adapter = huggingface.HuggingFaceReasoningModel(
        HuggingFaceSettings(
            device="cuda",
            reasoning_model="test/reasoning-model",
        )
    )

    tokenizer, model = adapter._load_components()

    assert tokenizer is fake_tokenizer
    assert model is fake_model
    assert fake_model.device == "cuda"
    assert fake_model.eval_called is True
    assert model_factory.calls == [
        (
            "test/reasoning-model",
            {"local_files_only": False, "dtype": fake_torch.float16},
        )
    ]


def test_reasoning_model_pins_revision_and_offline_loading_for_both_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(cuda_available=False)
    fake_tokenizer = object()
    fake_model = _FakeModel()
    tokenizer_factory = _FakeFactory(fake_tokenizer)
    model_factory = _FakeFactory(fake_model)
    fake_transformers = SimpleNamespace(
        AutoTokenizer=tokenizer_factory,
        AutoModelForCausalLM=model_factory,
    )

    def fake_import(name: str) -> Any:
        return fake_torch if name == "torch" else fake_transformers

    monkeypatch.setattr(huggingface, "import_module", fake_import)
    adapter = huggingface.HuggingFaceReasoningModel(
        HuggingFaceSettings(
            device="cpu",
            reasoning_model="private/reasoning-model",
            model_revision="pinned-revision",
            offline=True,
        )
    )

    tokenizer, model = adapter._load_components()

    assert tokenizer is fake_tokenizer
    assert model is fake_model
    expected_kwargs = {
        "local_files_only": True,
        "revision": "pinned-revision",
    }
    assert tokenizer_factory.calls == [("private/reasoning-model", expected_kwargs)]
    assert model_factory.calls == [("private/reasoning-model", expected_kwargs)]


def test_quantized_reasoning_model_uses_nf4_without_moving_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(cuda_available=True)
    fake_model = _FakeModel()
    model_factory = _FakeFactory(fake_model)

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_transformers = SimpleNamespace(
        AutoTokenizer=_FakeFactory(object()),
        AutoModelForCausalLM=model_factory,
        BitsAndBytesConfig=FakeBitsAndBytesConfig,
    )

    def fake_import(name: str) -> Any:
        if name == "torch":
            return fake_torch
        if name == "transformers":
            return fake_transformers
        if name == "bitsandbytes":
            return object()
        raise ImportError(name)

    monkeypatch.setattr(huggingface, "import_module", fake_import)
    adapter = huggingface.HuggingFaceReasoningModel(
        HuggingFaceSettings(
            device="cuda",
            quantization="bitsandbytes_4bit",
            reasoning_model="test/quantized-model",
        )
    )

    adapter._load_components()

    model_kwargs = model_factory.calls[0][1]
    quantization_config = model_kwargs["quantization_config"]
    assert model_kwargs["device_map"] == {"": 0}
    assert quantization_config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": fake_torch.bfloat16,
    }
    assert fake_model.device == "cpu"


def test_chat_prompt_disables_qwen_thinking() -> None:
    captured: dict[str, Any] = {}

    class FakeTokenizer:
        chat_template = "template"

        def apply_chat_template(self, _messages: Any, **kwargs: Any) -> str:
            captured.update(kwargs)
            return "rendered"

    rendered = huggingface._build_chat_prompt(
        FakeTokenizer(),
        "prompt",
        "system",
        enable_thinking=False,
    )

    assert rendered == "rendered"
    assert captured["enable_thinking"] is False


def test_embedding_and_reranker_receive_selected_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str, str | None]] = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, *, device: str | None = None, **_kwargs: Any) -> None:
            created.append(("embedding", model_name, device))

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str | None = None, **_kwargs: Any) -> None:
            created.append(("reranker", model_name, device))

    fake_module = SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer,
        CrossEncoder=FakeCrossEncoder,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(
        huggingface,
        "import_module",
        lambda _name: _FakeTorch(cuda_available=True),
    )
    settings = HuggingFaceSettings(
        device="cuda",
        embedding_model="test/embedding-model",
        reranker_model="test/reranker-model",
    )

    huggingface.HuggingFaceEmbeddingModel(settings)
    huggingface.HuggingFaceRerankerModel(settings)

    assert created == [
        ("embedding", "test/embedding-model", "cuda"),
        ("reranker", "test/reranker-model", "cuda"),
    ]
