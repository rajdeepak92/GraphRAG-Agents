"""Layout-aware deterministic chunk construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from multi_agentic_graph_rag.config.settings import ChunkingSettings
from multi_agentic_graph_rag.domain.identifiers import make_chunk_id
from multi_agentic_graph_rag.domain.schemas import ChunkLayout, ManifestChunk, ParsedBlock
from multi_agentic_graph_rag.services.parsing import normalize_text

CHUNKER_FINGERPRINT = "layout-recursive-v2"


@dataclass(frozen=True)
class _Span:
    block: ParsedBlock
    start: int
    end: int


def chunk_blocks(
    blocks: list[ParsedBlock],
    settings: ChunkingSettings,
) -> tuple[list[ManifestChunk], str]:
    """Split parsed blocks while retaining source offsets and layout."""
    combined, spans = _combine(blocks)
    if not combined.strip():
        return [], fingerprint(settings)
    splitter_module = cast(Any, import_module("langchain_text_splitters"))
    splitter_cls = splitter_module.RecursiveCharacterTextSplitter
    splitter = splitter_cls(
        chunk_size=min(settings.chunk_size, settings.maximum_chunk_size),
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    result: list[ManifestChunk] = []
    for document in splitter.create_documents([combined]):
        text = normalize_text(str(document.page_content))
        if not text:
            continue
        start = int(document.metadata.get("start_index", combined.find(document.page_content)))
        end = start + len(str(document.page_content))
        overlapping = [span for span in spans if span.end > start and span.start < end]
        if not overlapping:
            continue
        first, last = overlapping[0], overlapping[-1]
        source_start = first.block.start_char + max(0, start - first.start)
        source_end = last.block.start_char + min(
            len(last.block.original_text), max(0, end - last.start)
        )
        source_location = first.block.source_location
        chunk_identifier = make_chunk_id(
            chunk_text=text,
            start_char=source_start,
            end_char=source_end,
            source_location=source_location,
        )
        result.append(
            ManifestChunk(
                chunk_id=chunk_identifier,
                sequence_index=len(result),
                chunk_text=text,
                content_hash=f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}",
                start_char=source_start,
                end_char=source_end,
                layout=ChunkLayout(
                    page_start=first.block.page,
                    page_end=last.block.page,
                    section=first.block.section,
                    block_types=list(dict.fromkeys(span.block.block_type for span in overlapping)),
                    source_location=source_location,
                ),
                source_provenance=None,
                neo4j_status="pending",
                chroma_status="pending",
            )
        )
    return result, fingerprint(settings)


def fingerprint(settings: ChunkingSettings) -> str:
    """Return the frozen chunker fingerprint."""
    return (
        f"{CHUNKER_FINGERPRINT}:size={settings.chunk_size}:"
        f"overlap={settings.chunk_overlap}:max={settings.maximum_chunk_size}"
    )


def _combine(blocks: list[ParsedBlock]) -> tuple[str, list[_Span]]:
    parts: list[str] = []
    spans: list[_Span] = []
    cursor = 0
    for block in blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(block.original_text)
        cursor += len(block.original_text)
        spans.append(_Span(block=block, start=start, end=cursor))
    return "".join(parts), spans


__all__ = ["CHUNKER_FINGERPRINT", "chunk_blocks", "fingerprint"]
