"""Hugging Face reranker adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from multi_agentic_graph_rag.config.settings import HuggingFaceSettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError


class HuggingFaceRerankerModel:
    """Coordinate hugging face reranker model behavior within the llm_models boundary."""

    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (HuggingFaceSettings): Validated settings that control this operation.
        """
        self.settings = settings
        self.reranker_fingerprint = f"hf:{settings.reranker_model}"
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {"local_files_only": settings.offline}
        if settings.token:
            kwargs["token"] = settings.token
        self.model = CrossEncoder(
            settings.reranker_model,
            device=_resolve_device(settings.device),
            **kwargs,
        )

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """Execute the rerank operation within its declared architectural boundary.

        Args:
            query (str): Input text processed in memory and excluded from diagnostic logs.
            documents (list[str]): Documents required by the operation's typed contract.

        Returns:
            list[int]: The typed result produced by the operation.
        """
        if not documents:
            return []
        scores = self.model.predict([(query, document) for document in documents])
        score_values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return sorted(
            range(len(documents)),
            key=lambda index: float(score_values[index]),
            reverse=True,
        )


def _resolve_device(configured_device: str) -> str:
    """Resolve and validate the device used by the local Hugging Face reranker."""
    torch = import_module("torch")
    if configured_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if configured_device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError(
            "HUGGINGFACE_DEVICE=cuda was requested, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
        )
    return configured_device
