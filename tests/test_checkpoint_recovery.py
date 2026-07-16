"""Checkpoint boundary tests for completed external writes."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

from multi_agentic_graph_rag.domain.schemas import ChunkLayout, ManifestChunk
from multi_agentic_graph_rag.services.manifest import build_chunk_manifest
from multi_agentic_graph_rag.workflows.ingestion_graph import _persist_chunk_chroma
from multi_agentic_graph_rag.workflows.requirement_discovery_graph import (
    _project_chunk_semantics,
)


def _chunk() -> ManifestChunk:
    text = "The service shall retain audit events."
    return ManifestChunk(
        chunk_id="CHK-1",
        sequence_index=0,
        chunk_text=text,
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
        start_char=0,
        end_char=len(text),
        layout=ChunkLayout(
            page_start=1,
            page_end=1,
            section="Audit",
            block_types=["paragraph"],
            source_location="page=1",
        ),
        source_provenance=None,
        neo4j_status="persisted",
        chroma_status="pending",
    )


class _Chroma:
    def __init__(self) -> None:
        self.writes = 0

    def upsert_chunk(self, **_: Any) -> None:
        self.writes += 1


class _Neo4j:
    def __init__(self) -> None:
        self.writes = 0

    def upsert_semantic_projection(self, **_: Any) -> None:
        self.writes += 1

    def validate_relationships(self, project: str, expected: set[str]) -> set[str]:
        return expected


def test_completed_chroma_write_is_not_repeated_before_validation() -> None:
    chroma = _Chroma()
    runtime = SimpleNamespace(
        chroma=chroma,
        embedding=SimpleNamespace(embedding_fingerprint="test:3"),
    )
    state: dict[str, Any] = {
        "project": "alpha",
        "run_id": "RUN-1",
        "chunks": [_chunk().model_dump(mode="json")],
        "current_index": 0,
        "current_embedding": [0.1, 0.2, 0.3],
    }
    update = _persist_chunk_chroma(state, runtime)  # type: ignore[arg-type]
    state.update(update)
    _persist_chunk_chroma(state, runtime)  # type: ignore[arg-type]
    assert chroma.writes == 1


def test_completed_semantic_projection_is_not_repeated_before_map_insert() -> None:
    neo4j = _Neo4j()
    runtime = SimpleNamespace(neo4j=neo4j)
    chunk = _chunk().model_copy(update={"chroma_status": "persisted"})
    manifest = build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[chunk])
    state: dict[str, Any] = {
        "project": "alpha",
        "run_id": "RUN-1",
        "manifest": manifest.model_dump(mode="json"),
        "current_index": 0,
        "current_response": {
            "chunk_id": "CHK-1",
            "requirements": [],
            "entities": [],
            "relationships": [],
        },
    }
    update = _project_chunk_semantics(state, runtime)  # type: ignore[arg-type]
    state.update(update)
    _project_chunk_semantics(state, runtime)  # type: ignore[arg-type]
    assert neo4j.writes == 1
