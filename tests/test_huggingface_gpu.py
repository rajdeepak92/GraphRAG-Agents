"""Hugging Face reranker device-selection regression tests."""

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


class _FakeTorch:
    def __init__(self, *, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(available=cuda_available)


def test_config_loads_explicit_huggingface_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("HUGGINGFACE_DEVICE", "CUDA")

    settings = load_config().huggingface
    assert settings.device == "cuda"


def test_cuda_device_fails_fast_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = _FakeTorch(cuda_available=False)
    monkeypatch.setattr(huggingface, "import_module", lambda _name: fake_torch)

    with pytest.raises(ConfigurationError, match="CUDA is unavailable"):
        huggingface._resolve_device("cuda")


def test_reranker_receives_selected_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, str, str | None]] = []

    class FakeCrossEncoder:
        def __init__(self, model_name: str, *, device: str | None = None, **_kwargs: Any) -> None:
            created.append(("reranker", model_name, device))

    fake_module = SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(
        huggingface,
        "import_module",
        lambda _name: _FakeTorch(cuda_available=True),
    )
    settings = HuggingFaceSettings(
        device="cuda",
        reranker_model="test/reranker-model",
    )

    huggingface.HuggingFaceRerankerModel(settings)

    assert created == [("reranker", "test/reranker-model", "cuda")]
