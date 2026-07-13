"""Provider-neutral model ports."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ReasoningModel(Protocol):
    """Specify the provider-neutral reasoning model interface required by this boundary."""

    provider_name: str

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[T],
        system_message: str,
        operation: str,
        request_id: str,
        max_attempts: int = 2,
    ) -> T:
        """Return structured data validated against the supplied Pydantic schema."""


class EmbeddingModel(Protocol):
    """Specify the provider-neutral embedding model interface required by this boundary."""

    provider_name: str
    embedding_fingerprint: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in order."""


class RerankerModel(Protocol):
    """Specify the provider-neutral reranker model interface required by this boundary."""

    provider_name: str

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """Return document indexes ordered by relevance."""
