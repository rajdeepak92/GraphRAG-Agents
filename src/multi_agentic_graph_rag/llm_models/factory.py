"""Model factory."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.llm_models.azure_openai import (
    AzureOpenAIEmbeddingModel,
    AzureOpenAIReasoningModel,
)
from multi_agentic_graph_rag.llm_models.huggingface import (
    HuggingFaceEmbeddingModel,
    HuggingFaceReasoningModel,
    HuggingFaceRerankerModel,
)
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, ReasoningModel, RerankerModel


def create_reasoning_model(
    settings: AppSettings,
    *,
    logger: Any | None = None,
    run_dir: Path | None = None,
) -> ReasoningModel:
    """Create reasoning model.

    Args:
        settings (AppSettings): Validated settings that control this operation.
        logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
        run_dir (Path | None): Filesystem location authorized for this operation.

    Returns:
        ReasoningModel: The typed result produced by the operation.

    Raises:
        ConfigurationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    provider = settings.reasoning_model.provider
    if provider == "azure_openai":
        if not settings.azure_openai.endpoint:
            raise ConfigurationError("REASONING_MODEL_PROVIDER=azure_openai requires endpoint")
        if not settings.azure_openai.api_key:
            raise ConfigurationError("REASONING_MODEL_PROVIDER=azure_openai requires api key")
        if not settings.azure_openai.reasoning_deployment:
            raise ConfigurationError(
                "REASONING_MODEL_PROVIDER=azure_openai requires reasoning deployment"
            )
        if find_spec("openai") is None:
            raise ConfigurationError(
                "REASONING_MODEL_PROVIDER=azure_openai requires openai; "
                "install with: uv sync --dev --extra azure"
            )
        return AzureOpenAIReasoningModel(
            settings.azure_openai,
            discovery_batch_size=settings.discovery.batch_size,
            log_llm_responses=settings.discovery.log_llm_responses,
            logger=logger,
            run_dir=run_dir,
        )
    if provider == "huggingface":
        if not settings.huggingface.reasoning_model:
            raise ConfigurationError(
                "REASONING_MODEL_PROVIDER=huggingface requires HUGGINGFACE_REASONING_MODEL"
            )
        if find_spec("transformers") is None:
            raise ConfigurationError(
                "REASONING_MODEL_PROVIDER=huggingface requires transformers; "
                "install with: uv sync --dev --extra local-llm"
            )
        return HuggingFaceReasoningModel(
            settings.huggingface,
            logger=logger,
            run_dir=run_dir,
        )
    raise ConfigurationError(f"Unsupported reasoning model provider for ingest: {provider}")


def create_embedding_model(
    settings: AppSettings,
    *,
    logger: Any | None = None,
) -> EmbeddingModel:
    """Create embedding model.

    Args:
        settings (AppSettings): Validated settings that control this operation.
        logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        EmbeddingModel: The typed result produced by the operation.

    Raises:
        ConfigurationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    provider = settings.embedding_model.provider
    if provider == "azure_openai":
        if not settings.azure_openai.endpoint:
            raise ConfigurationError("EMBEDDING_MODEL_PROVIDER=azure_openai requires endpoint")
        if not settings.azure_openai.api_key:
            raise ConfigurationError("EMBEDDING_MODEL_PROVIDER=azure_openai requires api key")
        if not settings.azure_openai.embedding_deployment:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=azure_openai requires embedding deployment"
            )
        if find_spec("openai") is None:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=azure_openai requires openai; "
                "install with: uv sync --dev --extra azure"
            )
        if find_spec("tiktoken") is None:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=azure_openai requires tiktoken; "
                "install with: uv sync --dev --extra azure"
            )
        return AzureOpenAIEmbeddingModel(settings.azure_openai, logger=logger)
    if provider == "huggingface":
        if not settings.huggingface.embedding_model:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=huggingface requires HUGGINGFACE_EMBEDDING_MODEL"
            )
        if find_spec("sentence_transformers") is None:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=huggingface requires sentence-transformers; "
                "install with: uv sync --dev --extra local-llm"
            )
        return HuggingFaceEmbeddingModel(settings.huggingface)
    raise ConfigurationError(f"Unsupported embedding model provider for ingest: {provider}")


def create_reranker_model(settings: AppSettings) -> RerankerModel:
    """Create reranker model.

    Args:
        settings (AppSettings): Validated settings that control this operation.

    Returns:
        RerankerModel: The typed result produced by the operation.

    Raises:
        ConfigurationError: If validated inputs or required dependencies cannot satisfy the
        contract.
    """
    provider = settings.reranker_model.provider
    if provider == "huggingface":
        if not settings.huggingface.reranker_model:
            raise ConfigurationError(
                "RERANKER_MODEL_PROVIDER=huggingface requires HUGGINGFACE_RERANKER_MODEL"
            )
        if find_spec("sentence_transformers") is None:
            raise ConfigurationError(
                "RERANKER_MODEL_PROVIDER=huggingface requires sentence-transformers; "
                "install with: uv sync --dev --extra local-llm"
            )
        return HuggingFaceRerankerModel(settings.huggingface)
    raise ConfigurationError(f"Unsupported reranker model provider for ingest: {provider}")
