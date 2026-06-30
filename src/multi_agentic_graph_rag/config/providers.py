"""Provider name constants."""

from enum import StrEnum


class ReasoningProvider(StrEnum):
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"


class EmbeddingProvider(StrEnum):
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"


class RerankerProvider(StrEnum):
    HUGGINGFACE = "huggingface"
