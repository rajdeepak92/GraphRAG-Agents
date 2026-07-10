"""Deterministic, domain-agnostic atomic source-unit segmentation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from multi_agentic_graph_rag.domain.knowledge_models import (
    ChunkTextUnitLink,
    LexicalKnowledgeProjection,
    TextUnit,
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_WHITESPACE = re.compile(r"\s+")


class SourceBlockLike(Protocol):
    block_id: str
    original_text: str
    page: int | None
    section: str | None
    start_char: int
    end_char: int


class ChunkLike(Protocol):
    chunk_id: str
    start_char: int
    end_char: int


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def build_lexical_projection(
    *,
    project: str,
    document_id: str,
    document_version_id: str,
    blocks: Sequence[SourceBlockLike],
    chunks: Sequence[ChunkLike],
) -> LexicalKnowledgeProjection:
    units = segment_text_units(document_version_id=document_version_id, blocks=blocks)
    links = link_chunks_to_text_units(chunks=chunks, text_units=units)
    return LexicalKnowledgeProjection(
        project=project,
        document_id=document_id,
        document_version_id=document_version_id,
        text_units=units,
        chunk_links=links,
    )


def segment_text_units(
    *, document_version_id: str, blocks: Sequence[SourceBlockLike]
) -> list[TextUnit]:
    units: list[TextUnit] = []
    for block in blocks:
        for unit_type, text, relative_start, relative_end in _split_block(block.original_text):
            normalized = normalize_text(text)
            if not normalized:
                continue
            ordinal = len(units) + 1
            units.append(
                TextUnit(
                    text_unit_id=_text_unit_id(
                        document_version_id=document_version_id,
                        block_id=block.block_id,
                        ordinal=ordinal,
                        text=normalized,
                    ),
                    document_version_id=document_version_id,
                    block_id=block.block_id,
                    ordinal=ordinal,
                    unit_type=unit_type,
                    text=normalized,
                    normalized_text=normalized.lower(),
                    page=block.page,
                    section=block.section,
                    start_char=block.start_char + relative_start,
                    end_char=block.start_char + relative_end,
                )
            )
    return units


def link_chunks_to_text_units(
    *, chunks: Sequence[ChunkLike], text_units: Sequence[TextUnit]
) -> list[ChunkTextUnitLink]:
    links: list[ChunkTextUnitLink] = []
    for chunk in chunks:
        overlapping = [
            unit
            for unit in text_units
            if unit.end_char > chunk.start_char and unit.start_char < chunk.end_char
        ]
        for ordinal, unit in enumerate(overlapping, start=1):
            links.append(
                ChunkTextUnitLink(
                    chunk_id=chunk.chunk_id,
                    text_unit_id=unit.text_unit_id,
                    ordinal_in_chunk=ordinal,
                )
            )
    return links


def _split_block(text: str) -> Iterable[tuple[str, str, int, int]]:
    lines = text.splitlines(keepends=True)
    if len(lines) > 1 and any(_BULLET.match(line) for line in lines if line.strip()):
        cursor = 0
        for line in lines:
            raw = line.rstrip("\r\n")
            if raw.strip():
                match = _BULLET.match(raw)
                content_start = match.end() if match else 0
                content = raw[content_start:].strip()
                if content:
                    leading = len(raw[content_start:]) - len(raw[content_start:].lstrip())
                    start = cursor + content_start + leading
                    yield "bullet", content, start, start + len(content)
            cursor += len(line)
        return

    position = 0
    for part in _SENTENCE_BOUNDARY.split(text):
        stripped = part.strip()
        if not stripped:
            position += len(part)
            continue
        start = text.find(stripped, position)
        end = start + len(stripped)
        yield "sentence", stripped, start, end
        position = end


def _text_unit_id(*, document_version_id: str, block_id: str, ordinal: int, text: str) -> str:
    payload = f"{document_version_id}|{block_id}|{ordinal}|{text}".encode("utf-8")
    return f"TU-{hashlib.sha256(payload).hexdigest()[:20].upper()}"
