"""Provider name constants."""

from enum import StrEnum


class ReasoningProvider(StrEnum):
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL_HEURISTIC = "local_heuristic"


class EmbeddingProvider(StrEnum):
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    LOCAL_HASH = "local_hash"


class RerankerProvider(StrEnum):
    HUGGINGFACE = "huggingface"
    NONE = "none"
