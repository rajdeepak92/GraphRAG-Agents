from __future__ import annotations

import unittest
from typing import Any

from multi_agentic_graph_rag.config.settings import UserStorySettings
from multi_agentic_graph_rag.services.retrieval import RetrievalService


class RetrievalServiceTests(unittest.TestCase):
    def test_fusion_dedups_and_always_includes_evidence(self) -> None:
        chroma = _FakeChroma([("C2", "dense", 0.1), ("C1", "duplicate evidence", 0.2)])
        neo4j = _FakeNeo4j(
            evidence=[("C1", "evidence text")],
            fulltext=[("C3", "sparse", 1.0)],
            neighbors=[("C4", "neighbour")],
        )
        # Rank evidence (index 0) last so top_k would drop it without the guarantee.
        reranker = _FakeReranker(order=[3, 2, 1, 0])
        service = _service(chroma, neo4j, reranker, top_k=2)

        context = service.retrieve_context(
            requirement_text="configure thresholds",
            document_version_id="DV-1",
            evidence_chunk_ids=["C1"],
        )

        ids = [chunk.chunk_id for chunk in context.chunks]
        self.assertEqual(context.source, "hybrid")
        self.assertIn("C1", ids)  # evidence retained despite being ranked last / beyond top_k
        self.assertEqual(len(ids), len(set(ids)))  # deduped

    def test_reranker_ordering_is_respected(self) -> None:
        chroma = _FakeChroma([("C1", "a", 0.1), ("C2", "b", 0.2), ("C3", "c", 0.3)])
        neo4j = _FakeNeo4j()
        reranker = _FakeReranker(order=[2, 0, 1])
        service = _service(chroma, neo4j, reranker, top_k=3)

        context = service.retrieve_context(
            requirement_text="configure thresholds",
            document_version_id="DV-1",
            evidence_chunk_ids=[],
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["C3", "C1", "C2"])
        self.assertEqual(reranker.calls[0][0], "configure thresholds")

    def test_empty_retrieval_falls_back_to_requirement_text(self) -> None:
        service = _service(_FakeChroma([]), _FakeNeo4j(), _FakeReranker(), top_k=4)

        context = service.retrieve_context(
            requirement_text="configure thresholds",
            document_version_id="DV-1",
            evidence_chunk_ids=[],
        )

        self.assertEqual(context.chunks, [])
        self.assertEqual(context.source, "requirement_text_fallback")

    def test_store_failures_degrade_without_raising(self) -> None:
        service = _service(_ExplodingChroma(), _ExplodingNeo4j(), _FakeReranker(), top_k=4)

        context = service.retrieve_context(
            requirement_text="configure thresholds",
            document_version_id="DV-1",
            evidence_chunk_ids=["C9"],
        )

        self.assertEqual(context.chunks, [])
        self.assertEqual(context.source, "requirement_text_fallback")

    def test_evidence_only_context_is_labelled_evidence(self) -> None:
        neo4j = _FakeNeo4j(evidence=[("C1", "evidence text")])
        service = _service(_FakeChroma([]), neo4j, _FakeReranker(), top_k=4)

        context = service.retrieve_context(
            requirement_text="configure thresholds",
            document_version_id="DV-1",
            evidence_chunk_ids=["C1"],
        )

        self.assertEqual([chunk.chunk_id for chunk in context.chunks], ["C1"])
        self.assertEqual(context.source, "evidence")


class _FakeChroma:
    def __init__(self, results: list[tuple[str, str, float]]) -> None:
        self.results = results

    def query_chunks(
        self, query_embedding: list[float], document_version_id: str, n_results: int
    ) -> list[tuple[str, str, float]]:
        return self.results


class _ExplodingChroma:
    def query_chunks(self, *args: Any, **kwargs: Any) -> list[tuple[str, str, float]]:
        raise RuntimeError("chroma unavailable")


class _FakeNeo4j:
    def __init__(
        self,
        *,
        evidence: list[tuple[str, str]] | None = None,
        fulltext: list[tuple[str, str, float]] | None = None,
        neighbors: list[tuple[str, str]] | None = None,
    ) -> None:
        self._evidence = evidence or []
        self._fulltext = fulltext or []
        self._neighbors = neighbors or []

    def fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        return self._evidence

    def fulltext_search_chunks(
        self, query: str, document_version_id: str, limit: int
    ) -> list[tuple[str, str, float]]:
        return self._fulltext

    def neighbor_chunks(
        self, chunk_ids: list[str], document_version_id: str, window: int
    ) -> list[tuple[str, str]]:
        return self._neighbors


class _ExplodingNeo4j:
    def fetch_chunks(self, *args: Any, **kwargs: Any) -> list[tuple[str, str]]:
        raise RuntimeError("neo4j unavailable")

    def fulltext_search_chunks(self, *args: Any, **kwargs: Any) -> list[tuple[str, str, float]]:
        raise RuntimeError("neo4j unavailable")

    def neighbor_chunks(self, *args: Any, **kwargs: Any) -> list[tuple[str, str]]:
        raise RuntimeError("neo4j unavailable")


class _FakeReranker:
    provider_name = "huggingface"

    def __init__(self, order: list[int] | None = None) -> None:
        self.order = order
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        self.calls.append((query, documents))
        if self.order is not None and sorted(self.order) == list(range(len(documents))):
            return self.order
        return list(range(len(documents)))


class _FakeEmbedding:
    provider_name = "huggingface"
    embedding_fingerprint = "fake"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _service(
    chroma: Any,
    neo4j: Any,
    reranker: Any,
    *,
    top_k: int,
) -> RetrievalService:
    return RetrievalService(
        chroma=chroma,
        neo4j=neo4j,
        embedding_model=_FakeEmbedding(),
        reranker_model=reranker,
        settings=UserStorySettings(top_k=top_k, dense_k=5, sparse_k=5, neighbor_window=1),
    )


if __name__ == "__main__":
    unittest.main()
