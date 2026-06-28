"""Provider identifiers used by configuration and adapters."""

from __future__ import annotations

from enum import StrEnum


class ReasoningLLMProvider(StrEnum):
    """Supported reasoning LLM providers."""

    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"


class EmbeddingProvider(StrEnum):
    """Supported embedding providers."""

    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"


class VectorStoreProvider(StrEnum):
    """Supported vector-store providers."""

    CHROMA_LOCAL = "chroma_local"
    CHROMA_HTTP = "chroma_http"
    CHROMA_CLOUD = "chroma_cloud"


class GraphStoreProvider(StrEnum):
    """Supported graph-store providers."""

    NEO4J_LOCAL = "neo4j_local"
    NEO4J_REMOTE = "neo4j_remote"
