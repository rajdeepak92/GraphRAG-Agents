"""Manifest construction and persistence."""

from __future__ import annotations

import json
from pathlib import Path

from multi_agentic_graph_rag.domain.identifiers import document_id, document_version_id
from multi_agentic_graph_rag.domain.schemas import DocumentChunk, DocumentManifest
from multi_agentic_graph_rag.observability.logging import RunLogger


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
    logger: RunLogger | None = None,
) -> DocumentManifest:
    """Build manifest.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        logical_name (str): Logical name required by the operation's typed contract.
        version (str): Document version label within the project scope.
        source_path (Path): Filesystem location authorized for this operation.
        source_checksum (str): Source checksum required by the operation's typed contract.
        parser_fingerprint (str): Parser fingerprint required by the operation's typed contract.
        chunker_fingerprint (str): Chunker fingerprint required by the operation's typed contract.
        chunks (list[DocumentChunk]): Ordered chunks processed without changing their identities.
        logger (RunLogger | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        DocumentManifest: The typed result produced by the operation.

    Side Effects:
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
    if logger is not None:
        logger.debug(
            "Building manifest for {project}:{logical_name}:{version}",
            step="build_manifest",
            project=project,
            logical_name=logical_name,
            version=version,
            chunk_count=len(chunks),
        )
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


def write_manifest(
    manifest: DocumentManifest,
    run_dir: Path,
    logger: RunLogger | None = None,
) -> Path:
    """Write manifest through the owning storage boundary.

    Args:
        manifest (DocumentManifest): Manifest required by the operation's typed contract.
        run_dir (Path): Filesystem location authorized for this operation.
        logger (RunLogger | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        Path: The typed result produced by the operation.

    Side Effects:
        May create or atomically replace files in the configured artifact boundary.
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
    path = run_dir / "chunk_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.debug(
            "Writing manifest to {path}",
            step="write_manifest",
            path=str(path),
            document_version_id=manifest.document_version_id,
        )
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return path
