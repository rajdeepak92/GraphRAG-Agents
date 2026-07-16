"""Current-run manifest isolation tests for hybrid retrieval."""

from __future__ import annotations

import hashlib
from typing import Any

from multi_agentic_graph_rag.config.settings import RetrievalSettings
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalRequirement,
    ChunkLayout,
    Evidence,
    ManifestChunk,
)
from multi_agentic_graph_rag.services.manifest import build_chunk_manifest
from multi_agentic_graph_rag.services.retrieval import RetrievalService


class _Neo4j:
    def fetch_chunks(self, project: str, chunk_ids: set[str]) -> list[tuple[str, str]]:
        return [(chunk_id, f"text {chunk_id}") for chunk_id in chunk_ids]

    def retrieve_semantic_chunks(self, **_: Any) -> list[tuple]:
        return [
            ("CHK-1", "allowed graph", 0.9, ["ENT-1"], ["REL-1"]),
            ("CHK-OLD", "stale graph", 1.0, ["ENT-1"], ["REL-1"]),
        ]


class _Chroma:
    def query_chunks(self, **_: Any) -> list[tuple[str, str, float]]:
        return [("CHK-1", "allowed vector", 0.1), ("CHK-OLD", "stale vector", 0.0)]


class _Embedding:
    provider_name = "test"
    embedding_fingerprint = "test:2"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


class _Reranker:
    provider_name = "test"

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        return list(range(len(documents)))


def test_retrieval_rejects_candidates_outside_manifest() -> None:
    text = "The system shall log events."
    chunk = ManifestChunk(
        chunk_id="CHK-1",
        sequence_index=0,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section=None,
            block_types=["paragraph"],
            source_location=None,
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="persisted",
    )
    manifest = build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[chunk])
    requirement = CanonicalRequirement(
        requirement_id="REQ-1",
        source_req_id=None,
        source_req_id_type="generated",
        requirement_text=text,
        requirement_type="Functional Requirement",
        priority="Medium",
        confidence=0.9,
        constraints=[],
        entity_ids=["ENT-1"],
        relationship_ids=["REL-1"],
        evidence=[
            Evidence(
                evidence_id="EVD-1",
                chunk_id="CHK-1",
                quote=text,
                start_char=0,
                end_char=len(text),
            )
        ],
    )
    service = RetrievalService(
        neo4j=_Neo4j(),  # type: ignore[arg-type]
        chroma=_Chroma(),  # type: ignore[arg-type]
        embedding=_Embedding(),
        reranker=_Reranker(),
        settings=RetrievalSettings(),
    )
    context = service.story_context(project="alpha", manifest=manifest, requirement=requirement)
    assert {item.chunk_id for item in context.ranked_evidence} == {"CHK-1"}
