"""Layout-aware deterministic chunk construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from multi_agentic_graph_rag.config.settings import ChunkingSettings
from multi_agentic_graph_rag.domain.identifiers import make_chunk_id
from multi_agentic_graph_rag.domain.schemas import ChunkLayout, ManifestChunk, ParsedBlock
from multi_agentic_graph_rag.services.parsing import normalize_text

CHUNKER_FINGERPRINT = "block-pack-v4-structural-headings"


@dataclass(frozen=True)
class _Piece:
    block: ParsedBlock
    text: str
    start_char: int
    end_char: int


def chunk_blocks(
    blocks: list[ParsedBlock],
    settings: ChunkingSettings,
) -> tuple[list[ManifestChunk], str]:
    """Pack complete blocks first and split only blocks above the hard maximum."""
    pieces = _pieces(blocks, settings)
    if not pieces:
        return [], fingerprint(settings)
    target = min(settings.chunk_size, settings.maximum_chunk_size)
    groups: list[list[_Piece]] = []
    current: list[_Piece] = []
    current_size = 0
    for piece in pieces:
        if _starts_structural_group(piece, current):
            groups.append(current)
            current = []
            current_size = 0
        separator_size = 2 if current else 0
        if current and current_size + separator_size + len(piece.text) > target:
            groups.append(current)
            current = []
            current_size = 0
        current.append(piece)
        current_size += (2 if current_size else 0) + len(piece.text)
        if len(piece.text) >= target:
            groups.append(current)
            current = []
            current_size = 0
    if current:
        groups.append(current)

    result: list[ManifestChunk] = []
    for group in groups:
        text = "\n\n".join(piece.text for piece in group)
        if not text:
            continue
        first, last = group[0], group[-1]
        source_start = first.start_char
        source_end = last.end_char
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
                    block_types=list(dict.fromkeys(piece.block.block_type for piece in group)),
                    source_location=source_location,
                ),
                source_provenance={
                    "block_ids": [piece.block.block_id for piece in group],
                    "source_locations": [
                        piece.block.source_location
                        for piece in group
                        if piece.block.source_location is not None
                    ],
                },
                neo4j_status="pending",
                chroma_status="pending",
            )
        )
    return result, fingerprint(settings)


def _starts_structural_group(piece: _Piece, current: list[_Piece]) -> bool:
    """Keep a new heading out of the preceding explicit requirement table."""
    if not current or piece.block.block_type != "heading":
        return False
    return any(current_piece.block.block_type == "table_row" for current_piece in current)


def fingerprint(settings: ChunkingSettings) -> str:
    """Return the frozen chunker fingerprint."""
    return (
        f"{CHUNKER_FINGERPRINT}:size={settings.chunk_size}:"
        f"overlap={settings.chunk_overlap}:max={settings.maximum_chunk_size}"
    )


def _pieces(blocks: list[ParsedBlock], settings: ChunkingSettings) -> list[_Piece]:
    pieces: list[_Piece] = []
    maximum = settings.maximum_chunk_size
    overlap = min(settings.chunk_overlap, max(0, maximum - 1))
    for block in blocks:
        text = normalize_text(block.normalized_text)
        if not text:
            continue
        if len(text) <= maximum:
            pieces.append(
                _Piece(
                    block=block,
                    text=text,
                    start_char=block.start_char,
                    end_char=block.end_char,
                )
            )
            continue
        start = 0
        while start < len(text):
            hard_end = min(len(text), start + maximum)
            end = hard_end
            if hard_end < len(text):
                boundary = max(
                    text.rfind("\n", start + maximum // 2, hard_end),
                    text.rfind(" ", start + maximum // 2, hard_end),
                )
                if boundary > start:
                    end = boundary
            fragment = text[start:end].strip()
            if fragment:
                source_start = min(block.end_char, block.start_char + start)
                source_end = min(block.end_char, block.start_char + end)
                pieces.append(
                    _Piece(
                        block=block,
                        text=fragment,
                        start_char=source_start,
                        end_char=source_end,
                    )
                )
            if end >= len(text):
                break
            start = max(start + 1, end - overlap)
    return pieces


__all__ = ["CHUNKER_FINGERPRINT", "chunk_blocks", "fingerprint"]
