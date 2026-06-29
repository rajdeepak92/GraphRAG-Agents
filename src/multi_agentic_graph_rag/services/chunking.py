"""Structure-first chunking with size limits."""

from __future__ import annotations

from multi_agentic_graph_rag.config.settings import ChunkingSettings
from multi_agentic_graph_rag.domain.identifiers import chunk_id
from multi_agentic_graph_rag.domain.schemas import DocumentChunk, ParsedBlock
from multi_agentic_graph_rag.services.parsing import normalize_text

CHUNKER_FINGERPRINT = "structure-size-v1"


def chunk_blocks(
    *,
    document_version_id: str,
    blocks: list[ParsedBlock],
    settings: ChunkingSettings,
) -> tuple[list[DocumentChunk], str]:
    chunks: list[DocumentChunk] = []
    current: list[ParsedBlock] = []
    current_size = 0

    for block in blocks:
        block_size = len(block.normalized_text)
        if current and current_size + block_size > settings.chunk_size:
            _append_chunk(chunks, document_version_id, current)
            current = _overlap_blocks(current, settings.chunk_overlap)
            current_size = sum(len(item.normalized_text) for item in current)
        current.append(block)
        current_size += block_size

    if current:
        _append_chunk(chunks, document_version_id, current)

    return chunks, (
        f"{CHUNKER_FINGERPRINT}:size={settings.chunk_size}:overlap={settings.chunk_overlap}"
    )


def _overlap_blocks(blocks: list[ParsedBlock], overlap: int) -> list[ParsedBlock]:
    if overlap <= 0:
        return []
    retained: list[ParsedBlock] = []
    size = 0
    for block in reversed(blocks):
        retained.insert(0, block)
        size += len(block.normalized_text)
        if size >= overlap:
            break
    return retained


def _append_chunk(
    chunks: list[DocumentChunk],
    document_version_id: str,
    blocks: list[ParsedBlock],
) -> None:
    text = "\n\n".join(block.original_text for block in blocks)
    ordinal = len(chunks) + 1
    chunks.append(
        DocumentChunk(
            chunk_id=chunk_id(document_version_id, ordinal, text),
            ordinal=ordinal,
            text=text,
            normalized_text=normalize_text(text),
            page=blocks[0].page,
            section=blocks[0].section,
            start_char=blocks[0].start_char,
            end_char=blocks[-1].end_char,
            source_block_ids=[block.block_id for block in blocks],
        )
    )
