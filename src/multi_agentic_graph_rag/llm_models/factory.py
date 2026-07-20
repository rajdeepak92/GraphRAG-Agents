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
from multi_agentic_graph_rag.llm_models.gemini import (
    GeminiEmbeddingModel,
    GeminiReasoningModel,
)
from multi_agentic_graph_rag.llm_models.huggingface import HuggingFaceRerankerModel
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, ReasoningModel, RerankerModel


def create_stage4_reasoning_model(
    settings: AppSettings,
    *,
    provider: str,
    logger: Any | None = None,
    run_dir: Path | None = None,
) -> ReasoningModel:
    """Create only the explicitly selected Stage-4 reasoning provider.

    Configuration belonging to the other provider is deliberately neither
    inspected nor initialized. SDK retries remain disabled in the provider
    adapters, leaving the Stage-4 producer as the single two-attempt authority.
    """
    if provider == "azure_openai":
        azure = settings.azure_openai
        if not azure.endpoint:
            raise ConfigurationError("azure_openai Stage 4 mode requires an Azure endpoint")
        if not azure.api_key:
            raise ConfigurationError("azure_openai Stage 4 mode requires Azure authentication")
        if not azure.reasoning_deployment:
            raise ConfigurationError("azure_openai Stage 4 mode requires a reasoning deployment")
        if find_spec("openai") is None:
            raise ConfigurationError("azure_openai Stage 4 mode requires the 'azure' project extra")
        return AzureOpenAIReasoningModel(
            azure,
            discovery_batch_size=1,
            log_llm_responses=bool(getattr(azure, "log_llm_responses", False)),
            logger=logger,
            run_dir=run_dir,
        )
    if provider == "gemini":
        gemini = settings.gemini
        if not gemini.api_key:
            raise ConfigurationError("gemini Stage 4 mode requires GEMINI_API_KEY")
        if not gemini.reasoning_model:
            raise ConfigurationError("gemini Stage 4 mode requires a reasoning model")
        if find_spec("google.genai") is None:
            raise ConfigurationError("gemini Stage 4 mode requires the 'gemini' project extra")
        return GeminiReasoningModel(
            gemini,
            discovery_batch_size=1,
            log_llm_responses=gemini.log_llm_responses,
            logger=logger,
            run_dir=run_dir,
        )
    raise ConfigurationError(f"Unsupported Stage 4 reasoning provider: {provider}")


def create_reasoning_model(
    settings: AppSettings,
    *,
    logger: Any | None = None,
    run_dir: Path | None = None,
    stage12: bool = False,
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
            discovery_batch_size=1,
            log_llm_responses=settings.azure_openai.log_llm_responses,
            logger=logger,
            run_dir=run_dir,
        )
    if provider == "gemini":
        if not settings.gemini.api_key:
            raise ConfigurationError("REASONING_MODEL_PROVIDER=gemini requires GEMINI_API_KEY")
        if not settings.gemini.reasoning_model:
            raise ConfigurationError(
                "REASONING_MODEL_PROVIDER=gemini requires GEMINI_REASONING_MODEL"
            )
        if find_spec("google.genai") is None:
            raise ConfigurationError(
                "REASONING_MODEL_PROVIDER=gemini requires google-genai; "
                "install with: uv sync --dev --extra gemini"
            )
        return GeminiReasoningModel(
            settings.gemini,
            discovery_batch_size=1,
            log_llm_responses=settings.gemini.log_llm_responses,
            logger=logger,
            run_dir=run_dir,
        )
    raise ConfigurationError(f"Unsupported reasoning model provider: {provider}")


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
    if provider == "gemini":
        if not settings.gemini.api_key:
            raise ConfigurationError("EMBEDDING_MODEL_PROVIDER=gemini requires GEMINI_API_KEY")
        if not settings.gemini.embedding_model:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=gemini requires GEMINI_EMBEDDING_MODEL"
            )
        if find_spec("google.genai") is None:
            raise ConfigurationError(
                "EMBEDDING_MODEL_PROVIDER=gemini requires google-genai; "
                "install with: uv sync --dev --extra gemini"
            )
        return GeminiEmbeddingModel(settings.gemini, logger=logger)
    raise ConfigurationError(f"Unsupported embedding model provider: {provider}")


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
    raise ConfigurationError(f"Unsupported reranker model provider: {provider}")
