"""Small shared runtime literals."""

from enum import StrEnum


class EnvVar(StrEnum):
    """Hugging Face environment aliases used by provider setup."""

    HF_TOKEN = "HF_TOKEN"
    HUGGINGFACE_TOKEN = "HUGGINGFACE_TOKEN"
    HUGGING_FACE_HUB_TOKEN = "HUGGING_FACE_HUB_TOKEN"
    HF_HUB_OFFLINE = "HF_HUB_OFFLINE"
    TRANSFORMERS_OFFLINE = "TRANSFORMERS_OFFLINE"


class ProviderName(StrEnum):
    """Supported model providers."""

    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"


__all__ = ["EnvVar", "ProviderName"]
