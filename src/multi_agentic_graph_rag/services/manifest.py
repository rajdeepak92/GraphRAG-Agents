"""Canonical artifact IO and checksum validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

from multi_agentic_graph_rag.domain.schemas import (
    ChunkManifest,
    ManifestChunk,
    canonical_checksum,
)


def build_chunk_manifest(
    *,
    project: str,
    run_id: str,
    chunks: list[ManifestChunk],
) -> ChunkManifest:
    """Build and validate the final Stage 1.1 manifest."""
    payload = ChunkManifest.model_construct(
        project=project,
        run_id=run_id,
        checksum="",
        chunks=chunks,
    )
    return ChunkManifest.model_validate(
        {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
    )


def atomic_write_model(model: BaseModel, path: Path) -> Path:
    """Atomically publish a JSON model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(model.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def load_model[T: BaseModel](path: Path, schema: type[T]) -> T:
    """Load and revalidate a strict JSON artifact."""
    return schema.model_validate_json(path.read_text(encoding="utf-8"))


__all__ = ["atomic_write_model", "build_chunk_manifest", "load_model"]
