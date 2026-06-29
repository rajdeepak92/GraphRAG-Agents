"""Structure-first chunking with size limits."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

from multi_agentic_graph_rag.config.settings import ChunkingSettings
from multi_agentic_graph_rag.domain.identifiers import chunk_id
from multi_agentic_graph_rag.domain.schemas import DocumentChunk, ParsedBlock
from multi_agentic_graph_rag.observability.logging import RunLogger
from multi_agentic_graph_rag.services.parsing import normalize_text

CHUNKER_FINGERPRINT = "langchain-recursive-v1"


@dataclass(frozen=True)
class _BlockSpan:
    block: ParsedBlock
    start: int
    end: int


def chunk_blocks(
    *,
    document_version_id: str,
    blocks: list[ParsedBlock],
    settings: ChunkingSettings,
    logger: RunLogger | None = None,
) -> tuple[list[DocumentChunk], str]:
    if logger is not None:
        logger.debug(
            "Chunking {block_count} blocks for {document_version_id}",
            step="chunk_document",
            block_count=len(blocks),
            document_version_id=document_version_id,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
    combined_text, spans = _combine_blocks(blocks)
    if not combined_text.strip():
        if logger is not None:
            logger.warning(
                "No content available for chunking",
                step="chunk_document",
                document_version_id=document_version_id,
            )
        return [], _fingerprint(settings)

    splitter_module = cast(Any, import_module("langchain_text_splitters"))
    splitter_cls = splitter_module.RecursiveCharacterTextSplitter
    splitter = splitter_cls(
        chunk_size=min(settings.chunk_size, settings.maximum_chunk_size),
        chunk_overlap=settings.chunk_overlap,
        add_start_index=True,
    )
    split_documents = splitter.create_documents([combined_text])

    chunks: list[DocumentChunk] = []
    search_start = 0
    for document in split_documents:
        text = str(document.page_content)
        if not text.strip():
            continue
        metadata = cast(dict[str, Any], document.metadata)
        start_index = metadata.get("start_index")
        if isinstance(start_index, int):
            combined_start = start_index
        else:
            combined_start = max(0, combined_text.find(text, search_start))
        combined_end = combined_start + len(text)
        search_start = max(combined_start, 0)
        _append_chunk(
            chunks,
            document_version_id,
            text,
            _overlapping_spans(spans, combined_start, combined_end),
            combined_start,
            combined_end,
        )

    if logger is not None:
        logger.debug(
            "Chunked {chunk_count} chunks for {document_version_id}",
            step="chunk_document",
            chunk_count=len(chunks),
            document_version_id=document_version_id,
        )
    return chunks, _fingerprint(settings)


def _fingerprint(settings: ChunkingSettings) -> str:
    return f"{CHUNKER_FINGERPRINT}:size={settings.chunk_size}:overlap={settings.chunk_overlap}"


def _combine_blocks(blocks: list[ParsedBlock]) -> tuple[str, list[_BlockSpan]]:
    parts: list[str] = []
    spans: list[_BlockSpan] = []
    cursor = 0
    for block in blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        parts.append(block.original_text)
        start = cursor
        cursor += len(block.original_text)
        spans.append(_BlockSpan(block=block, start=start, end=cursor))
    return "".join(parts), spans


def _overlapping_spans(
    spans: list[_BlockSpan],
    start: int,
    end: int,
) -> list[_BlockSpan]:
    overlapping = [span for span in spans if span.end > start and span.start < end]
    if overlapping:
        return overlapping
    return [min(spans, key=lambda span: abs(span.start - start))] if spans else []


def _append_chunk(
    chunks: list[DocumentChunk],
    document_version_id: str,
    text: str,
    spans: list[_BlockSpan],
    combined_start: int,
    combined_end: int,
) -> None:
    if not spans:
        return
    first = spans[0]
    last = spans[-1]
    start_char = _source_start(first, combined_start)
    end_char = _source_end(last, combined_end)
    ordinal = len(chunks) + 1
    chunks.append(
        DocumentChunk(
            chunk_id=chunk_id(document_version_id, ordinal, text),
            ordinal=ordinal,
            text=text,
            normalized_text=normalize_text(text),
            page=first.block.page,
            section=first.block.section,
            start_char=start_char,
            end_char=end_char,
            source_block_ids=[span.block.block_id for span in spans],
        )
    )


def _source_start(span: _BlockSpan, combined_start: int) -> int:
    if span.start <= combined_start <= span.end:
        return span.block.start_char + max(0, combined_start - span.start)
    return span.block.start_char


def _source_end(span: _BlockSpan, combined_end: int) -> int:
    if span.start <= combined_end <= span.end:
        return span.block.start_char + max(0, combined_end - span.start)
    return span.block.end_char
