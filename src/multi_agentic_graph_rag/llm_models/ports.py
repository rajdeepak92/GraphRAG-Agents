"""Provider-neutral model ports."""

from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class ReasoningModel(Protocol):
    provider_name: str

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[T],
    ) -> T:
        """Return structured data validated against the supplied Pydantic schema."""


class EmbeddingModel(Protocol):
    provider_name: str
    embedding_fingerprint: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents in order."""


class RerankerModel(Protocol):
    provider_name: str

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """Return document indexes ordered by relevance."""
