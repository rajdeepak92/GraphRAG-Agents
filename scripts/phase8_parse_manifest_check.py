"""Phase 8 verification script.

Usage:
    uv run python scripts/phase8_parse_manifest_check.py documents/inbox/PROJECT_1/requirements.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import UUID

from multi_agentic_graph_rag.application.services.manifest_builder import (
    build_chunk_manifest,
)
from multi_agentic_graph_rag.infrastructure.documents.chunker import ChunkingConfig
from multi_agentic_graph_rag.infrastructure.documents.parser_registry import ParserRegistry

DOCUMENT_VERSION_ID = UUID("00000000-0000-0000-0000-000000000001")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/phase8_parse_manifest_check.py <document-path>")
        return 2

    document_path = Path(sys.argv[1])
    parsed_document = ParserRegistry().parse(document_path)

    parsed_document = parsed_document.model_copy(
        update={"document_version_id": DOCUMENT_VERSION_ID}
    )

    manifest = build_chunk_manifest(
        document_version_id=DOCUMENT_VERSION_ID,
        parsed_document=parsed_document,
        chunking_config=ChunkingConfig(),
    )

    payload = {
        "document_path": str(document_path),
        "source_checksum": parsed_document.source_checksum,
        "parser_name": parsed_document.parser_name,
        "parser_version": parsed_document.parser_version,
        "total_blocks": len(parsed_document.blocks),
        "total_chunks": len(manifest.chunks),
        "chunk_ids": [chunk.chunk_id for chunk in manifest.chunks],
        "manifest_schema_version": manifest.manifest_schema_version,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
