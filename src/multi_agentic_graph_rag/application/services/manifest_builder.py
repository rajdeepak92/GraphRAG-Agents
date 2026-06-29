"""Build deterministic chunk manifests from parsed documents."""

from __future__ import annotations

import hashlib
import json
from uuid import UUID

from multi_agentic_graph_rag.domain.chunks import ChunkManifest
from multi_agentic_graph_rag.domain.documents import ParsedDocument
from multi_agentic_graph_rag.infrastructure.documents.chunker import (
    ChunkingConfig,
    StructureAwareChunker,
)


def build_chunk_manifest(
    *,
    document_version_id: UUID,
    parsed_document: ParsedDocument,
    chunking_config: ChunkingConfig,
) -> ChunkManifest:
    """Build a validated chunk manifest from parser output.

    The parser produces ParsedDocument.
    The chunker produces deterministic Chunk records.
    The manifest binds those chunks to a canonical document_version_id.
    """
    chunker = StructureAwareChunker(chunking_config)

    chunks = chunker.chunk(
        document_version_id=document_version_id,
        source_checksum=parsed_document.source_checksum,
        blocks=list(parsed_document.blocks),
    )

    parser_fingerprint = parsed_document.parser_fingerprint or _fingerprint(
        {
            "parser_name": parsed_document.parser_name,
            "parser_version": parsed_document.parser_version,
        }
    )

    return ChunkManifest(
        manifest_id=None,
        document_version_id=document_version_id,
        source_checksum=parsed_document.source_checksum,
        parser_fingerprint=parser_fingerprint,
        chunker_fingerprint=chunking_config.fingerprint(),
        chunks=tuple(chunks),
    )


def _fingerprint(payload: dict[str, str]) -> str:
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
