"""Manifest construction and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from multi_agentic_graph_rag.domain.identifiers import document_id, document_version_id
from multi_agentic_graph_rag.domain.schemas import DocumentChunk, DocumentManifest


def build_manifest(
    *,
    project: str,
    logical_name: str,
    version: str,
    source_path: Path,
    source_checksum: str,
    parser_fingerprint: str,
    chunker_fingerprint: str,
    chunks: list[DocumentChunk],
) -> DocumentManifest:
    doc_id = document_id(project, logical_name)
    doc_version_id = document_version_id(doc_id, version, source_checksum)
    return DocumentManifest(
        project=project,
        document_id=doc_id,
        document_version_id=doc_version_id,
        logical_name=logical_name,
        version=version,
        source_path=str(source_path),
        source_checksum=source_checksum,
        parser_fingerprint=parser_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        chunks=chunks,
    )


def write_manifest(manifest: DocumentManifest, staging_dir: Path, run_id: str) -> Path:
    path = staging_dir / run_id / "chunk_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path
