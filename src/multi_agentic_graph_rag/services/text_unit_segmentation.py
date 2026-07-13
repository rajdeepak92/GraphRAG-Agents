"""Deterministic segmentation of ingested chunks into atomic TextUnits.

Units are derived from chunk text but identified by *source-level* character
spans (``chunk.start_char`` + local offset), so the same sentence appearing in
two overlapping chunk windows dedupes to a single unit that lists both chunks.
Unit text is always an exact substring of the owning chunk text, preserving
evidence integrity; re-runs produce identical IDs.
"""

from __future__ import annotations

import re
from typing import Literal

from multi_agentic_graph_rag.domain.identifiers import text_unit_id
from multi_agentic_graph_rag.domain.schemas import (
    AssertionEvidenceRecord,
    DocumentChunk,
    TextUnit,
)

_LINE_SPLIT = re.compile(r"[^\n]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9\"'(\[])")
_BULLET_MARKER = re.compile(r"^\s*(?:[-*•▪●◦]|\d{1,3}[.)])\s+")


def segment_version_chunks(
    *,
    document_version_id: str,
    chunks: list[DocumentChunk],
) -> list[TextUnit]:
    """Segment all chunks of one document version into deduplicated TextUnits."""
    units_by_span: dict[tuple[int, int], TextUnit] = {}
    for chunk in chunks:
        for local_start, local_end, unit_type in _segment_text(chunk.text):
            span = (chunk.start_char + local_start, chunk.start_char + local_end)
            existing = units_by_span.get(span)
            if existing is not None:
                if chunk.chunk_id not in existing.chunk_ids:
                    existing.chunk_ids.append(chunk.chunk_id)
                continue
            text = chunk.text[local_start:local_end]
            units_by_span[span] = TextUnit(
                text_unit_id=text_unit_id(document_version_id, span[0], span[1], text),
                document_version_id=document_version_id,
                ordinal=0,
                unit_type=unit_type,
                text=text,
                start_char=span[0],
                end_char=span[1],
                page=chunk.page,
                section=chunk.section,
                chunk_ids=[chunk.chunk_id],
            )

    ordered = sorted(units_by_span.values(), key=lambda unit: (unit.start_char, unit.end_char))
    return [
        unit.model_copy(update={"ordinal": ordinal})
        for ordinal, unit in enumerate(ordered, start=1)
    ]


def attach_evidence_text_units(
    evidence: list[AssertionEvidenceRecord],
    *,
    chunks_by_id: dict[str, DocumentChunk],
    text_units: list[TextUnit],
) -> list[AssertionEvidenceRecord]:
    """Fill ``text_unit_ids`` on each evidence row from overlapping source spans.

    Evidence spans are chunk-relative; they are shifted into source coordinates
    via the owning chunk before overlap matching, and only units that list the
    evidence chunk are considered.
    """
    updated: list[AssertionEvidenceRecord] = []
    for record in evidence:
        trace = record.source_trace
        chunk = chunks_by_id.get(trace.chunk_id)
        if chunk is None:
            updated.append(record)
            continue
        start = chunk.start_char + trace.start_char
        end = chunk.start_char + trace.end_char
        unit_ids = [
            unit.text_unit_id
            for unit in text_units
            if trace.chunk_id in unit.chunk_ids and unit.start_char < end and unit.end_char > start
        ]
        updated.append(record.model_copy(update={"text_unit_ids": unit_ids}))
    return updated


_UnitType = Literal["sentence", "bullet"]


def _segment_text(text: str) -> list[tuple[int, int, _UnitType]]:
    """Yield (start, end, unit_type) segments over one chunk's raw text."""
    segments: list[tuple[int, int, _UnitType]] = []
    for line in _LINE_SPLIT.finditer(text):
        line_text = line.group(0)
        if not line_text.strip():
            continue
        if _BULLET_MARKER.match(line_text):
            start, end = _trimmed_span(line_text, line.start(), 0, len(line_text))
            if start < end:
                segments.append((start, end, "bullet"))
            continue
        previous = 0
        for boundary in _SENTENCE_BOUNDARY.finditer(line_text):
            start, end = _trimmed_span(line_text, line.start(), previous, boundary.start())
            if start < end:
                segments.append((start, end, "sentence"))
            previous = boundary.end()
        start, end = _trimmed_span(line_text, line.start(), previous, len(line_text))
        if start < end:
            segments.append((start, end, "sentence"))
    return segments


def _trimmed_span(
    line_text: str,
    line_offset: int,
    local_start: int,
    local_end: int,
) -> tuple[int, int]:
    """Execute the trimmed span operation within its declared architectural boundary.

    Args:
        line_text (str): Input text processed in memory and excluded from diagnostic logs.
        line_offset (int): Line offset required by the operation's typed contract.
        local_start (int): Local start required by the operation's typed contract.
        local_end (int): Local end required by the operation's typed contract.

    Returns:
        tuple[int, int]: The typed result produced by the operation.
    """
    segment = line_text[local_start:local_end]
    stripped = segment.strip()
    if not stripped:
        return (line_offset + local_start, line_offset + local_start)
    leading = len(segment) - len(segment.lstrip())
    start = local_start + leading
    return (line_offset + start, line_offset + start + len(stripped))
