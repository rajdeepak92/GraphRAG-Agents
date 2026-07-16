"""Schema and checksum regression tests for the simplified contracts."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from multi_agentic_graph_rag.domain.schemas import (
    ChunkLayout,
    ManifestChunk,
    SourceProvenance,
)
from multi_agentic_graph_rag.services.manifest import build_chunk_manifest


def _chunk(index: int = 0) -> ManifestChunk:
    text = "BR-1 The service shall retain audit events."
    return ManifestChunk(
        chunk_id=f"CHK-{index}",
        sequence_index=index,
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
        chroma_status="persisted",
    )


def test_source_provenance_pair_is_strict() -> None:
    assert (
        SourceProvenance(source_req_id="BR-1", source_req_id_type="source").source_req_id == "BR-1"
    )
    with pytest.raises(ValidationError):
        SourceProvenance(source_req_id=None, source_req_id_type="source")
    with pytest.raises(ValidationError):
        SourceProvenance(source_req_id="BR-1", source_req_id_type="generated")


def test_manifest_requires_contiguous_persisted_chunks_and_checksum() -> None:
    manifest = build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[_chunk()])
    assert manifest.checksum.startswith("sha256:")
    pending = _chunk().model_copy(update={"chroma_status": "pending"})
    with pytest.raises(ValidationError):
        build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[pending])
    with pytest.raises(ValidationError):
        build_chunk_manifest(project="alpha", run_id="RUN-1", chunks=[_chunk(1)])
