"""Model factory."""

from __future__ import annotations

from importlib.util import find_spec

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.llm_models.azure_openai import (
    AzureOpenAIEmbeddingModel,
    AzureOpenAIReasoningModel,
)
from multi_agentic_graph_rag.llm_models.huggingface import (
    HashEmbeddingModel,
    HuggingFaceEmbeddingModel,
    HuggingFaceReasoningModel,
    LocalHeuristicReasoningModel,
    NoopRerankerModel,
)
from multi_agentic_graph_rag.llm_models.ports import EmbeddingModel, ReasoningModel, RerankerModel


def create_reasoning_model(settings: AppSettings) -> ReasoningModel:
    provider = settings.reasoning_model.provider
    if provider == "azure_openai":
        if (
            settings.azure_openai.endpoint
            and settings.azure_openai.api_key
            and find_spec("openai") is not None
        ):
            return AzureOpenAIReasoningModel(settings.azure_openai)
        return LocalHeuristicReasoningModel()
    if provider == "huggingface":
        if settings.huggingface.reasoning_model and find_spec("transformers") is not None:
            return HuggingFaceReasoningModel(settings.huggingface)
        return LocalHeuristicReasoningModel()
    return LocalHeuristicReasoningModel()


def create_embedding_model(settings: AppSettings) -> EmbeddingModel:
    provider = settings.embedding_model.provider
    if provider == "azure_openai":
        if (
            settings.azure_openai.endpoint
            and settings.azure_openai.api_key
            and find_spec("openai") is not None
        ):
            return AzureOpenAIEmbeddingModel(settings.azure_openai)
        return HashEmbeddingModel()
    if provider == "huggingface":
        try:
            return HuggingFaceEmbeddingModel(settings.huggingface)
        except Exception:
            return HashEmbeddingModel()
    return HashEmbeddingModel()


def create_reranker_model(settings: AppSettings) -> RerankerModel:
    return NoopRerankerModel()
