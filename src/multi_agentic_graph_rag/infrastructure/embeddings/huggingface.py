"""Hugging Face embedding adapter."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class HuggingFaceEmbeddingAdapter:
    def __init__(
        self,
        *,
        model_name: str,
        normalize_embeddings: bool,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._normalize_embeddings = normalize_embeddings
        self._model: SentenceTransformer = SentenceTransformer(model_name, device=device)

    def fingerprint(self) -> str:
        payload = f"huggingface:{self._model_name}:normalize={self._normalize_embeddings}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"emb-{digest}"

    def embed_documents(self, texts: tuple[str, ...]) -> list[list[float]]:
        if not texts:
            return []

        vectors: Any = self._model.encode(
            list(texts),
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        rows = cast(list[list[float]], vectors.astype(float).tolist())
        return [[float(value) for value in row] for row in rows]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents((text,))
        return vectors[0]
