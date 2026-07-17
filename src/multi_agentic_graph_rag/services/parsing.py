"""Document parsing for text, Markdown, DOCX, and PDF files."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.domain.schemas import ParsedBlock

PARSER_FINGERPRINT = "positioned-layout-v2"


def checksum_bytes(data: bytes) -> str:
    """Execute the checksum bytes operation within its declared architectural boundary.

    Args:
        data (bytes): Validated structured data for the operation.

    Returns:
        str: The typed result produced by the operation.
    """
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text deterministically within the active scope.

    Args:
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.splitlines()]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        if line:
            normalized.append(line)
            previous_blank = False
        elif normalized and not previous_blank:
            normalized.append("")
            previous_blank = True
    return "\n".join(normalized).strip()


def parse_document(
    path: Path,
    logger: Any | None = None,
) -> tuple[list[ParsedBlock], str]:
    """Parse document.

    Args:
        path (Path): Filesystem location authorized for this operation.
        logger (RunLogger | None): Optional run-scoped logger used only for sanitized diagnostics.

    Returns:
        tuple[list[ParsedBlock], str]: The typed result produced by the operation.

    Raises:
        ValueError: If validated inputs or required dependencies cannot satisfy the contract.

    Side Effects:
        Emits sanitized run-scoped diagnostics when a logger is available.
    """
    if logger is not None:
        logger.debug(
            "Parsing document at {path}",
            step="parse_document",
            path=str(path),
            suffix=path.suffix.lower(),
        )
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        blocks = _parse_text(path)
        if logger is not None:
            logger.debug(
                "Parsed {count} blocks from {path}",
                step="parse_document",
                path=str(path),
                count=len(blocks),
            )
        return blocks, PARSER_FINGERPRINT
    if suffix == ".docx":
        blocks = _parse_docx(path)
        if logger is not None:
            logger.debug(
                "Parsed {count} blocks from {path}",
                step="parse_document",
                path=str(path),
                count=len(blocks),
            )
        return blocks, PARSER_FINGERPRINT
    if suffix == ".pdf":
        blocks = _parse_pdf(path)
        if logger is not None:
            logger.debug(
                "Parsed {count} blocks from {path}",
                step="parse_document",
                path=str(path),
                count=len(blocks),
            )
        return blocks, PARSER_FINGERPRINT
    raise ValueError(f"Unsupported document type: {suffix}")


def _parse_text(path: Path) -> list[ParsedBlock]:
    """Parse text.

    Args:
        path (Path): Filesystem location authorized for this operation.

    Returns:
        list[ParsedBlock]: The typed result produced by the operation.
    """
    text = path.read_text(encoding="utf-8")
    blocks: list[ParsedBlock] = []
    cursor = 0
    section: str | None = None
    for index, part in enumerate(re.split(r"\n\s*\n", text)):
        raw = part.strip()
        start = text.find(part, cursor)
        end = start + len(part)
        cursor = end
        if not raw:
            continue
        if raw.startswith("#"):
            section = raw.lstrip("#").strip() or section
        blocks.append(
            ParsedBlock(
                block_id=f"B{len(blocks) + 1}",
                original_text=raw,
                normalized_text=normalize_text(raw),
                section=section,
                paragraph=index + 1,
                start_char=max(start, 0),
                end_char=max(end, 0),
                block_type=(
                    "heading"
                    if raw.startswith("#")
                    else "list_item"
                    if raw.lstrip().startswith(("-", "*", "+"))
                    else "paragraph"
                ),
                source_location=f"char={max(start, 0)}:{max(end, 0)}",
            )
        )
    if not blocks and text.strip():
        blocks.append(
            ParsedBlock(
                block_id="B1",
                original_text=text,
                normalized_text=normalize_text(text),
                paragraph=1,
                start_char=0,
                end_char=len(text),
                block_type="paragraph",
                source_location=f"char=0:{len(text)}",
            )
        )
    return blocks


def _parse_docx(path: Path) -> list[ParsedBlock]:
    """Parse docx.

    Args:
        path (Path): Filesystem location authorized for this operation.

    Returns:
        list[ParsedBlock]: The typed result produced by the operation.
    """
    from docx import Document

    doc = Document(str(path))
    blocks: list[ParsedBlock] = []
    cursor = 0
    section: str | None = None
    for index, paragraph in enumerate(doc.paragraphs, start=1):
        raw = paragraph.text.strip()
        if not raw:
            continue
        if paragraph.style and paragraph.style.name.lower().startswith("heading"):
            section = raw
        blocks.append(
            ParsedBlock(
                block_id=f"B{len(blocks) + 1}",
                original_text=raw,
                normalized_text=normalize_text(raw),
                section=section,
                paragraph=index,
                start_char=cursor,
                end_char=cursor + len(raw),
                block_type=(
                    "heading"
                    if paragraph.style and paragraph.style.name.lower().startswith("heading")
                    else "paragraph"
                ),
                source_location=f"paragraph={index}",
                metadata={"style": paragraph.style.name if paragraph.style else None},
            )
        )
        cursor += len(raw) + 1
    return blocks


@dataclass(frozen=True)
class _PdfLine:
    page: int
    page_height: float
    y0: float
    y1: float
    words: tuple[tuple[float, str], ...]

    @property
    def text(self) -> str:
        return " ".join(word for _, word in self.words).strip()


_HEADING = re.compile(r"^[1-9]\d*(?:\.\d+)*\.?\s+\S")
_SOURCE_ID_PREFIX = re.compile(r"^[A-Z]{2,}(?:-[A-Z0-9]+)*-$")
_SOURCE_ID = re.compile(r"^[A-Z]{2,}(?:-[A-Z0-9]+)+$")


def _parse_pdf(path: Path) -> list[ParsedBlock]:
    """Parse positioned PDF words into headings, complete table rows, and paragraphs."""
    import fitz  # type: ignore[import-untyped]

    page_lines: list[list[_PdfLine]] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            raw_words: list[tuple[float, str, float, float]] = []
            for word in page.get_text("words"):
                x0, y0, _x1, y1, text, *_ = word
                raw_words.append((float(x0), str(text), float(y0), float(y1)))
            visual_groups: list[list[tuple[float, str, float, float]]] = []
            for item in sorted(raw_words, key=lambda value: (value[2], value[0])):
                if visual_groups and (
                    abs(item[2] - min(word[2] for word in visual_groups[-1])) <= 2.5
                ):
                    visual_groups[-1].append(item)
                else:
                    visual_groups.append([item])
            lines = [
                _PdfLine(
                    page=page_index,
                    page_height=float(page.rect.height),
                    y0=min(item[2] for item in values),
                    y1=max(item[3] for item in values),
                    words=tuple((item[0], item[1]) for item in sorted(values)),
                )
                for values in visual_groups
            ]
            page_lines.append(sorted(lines, key=lambda item: (item.y0, item.words[0][0])))

    noise = _repeated_page_noise(page_lines)
    blocks: list[ParsedBlock] = []
    cursor = 0
    section: str | None = None
    row_number = 0
    for lines in page_lines:
        filtered = [
            line
            for line in lines
            if (line.page, round(line.y0, 1), normalize_text(line.text)) not in noise
            and not (
                line.y1 > line.page_height - 55
                and re.fullmatch(r"\d+\s*/\s*\d+", normalize_text(line.text))
            )
            and normalize_text(line.text)
        ]
        explicit_rows, consumed = _explicit_id_rows(filtered)
        row_by_start = {
            start: (end, source_id, text) for start, end, source_id, text in explicit_rows
        }
        index = 0
        paragraph_lines: list[_PdfLine] = []

        while index < len(filtered):
            line = filtered[index]
            if index in row_by_start:
                cursor = _append_pdf_paragraph(blocks, paragraph_lines, section, cursor)
                paragraph_lines = []
                end, source_id, requirement_text = row_by_start[index]
                row_number += 1
                raw = f"{source_id} {requirement_text}"
                location = f"page={line.page},section={section or ''},row={row_number}"
                blocks.append(
                    ParsedBlock(
                        block_id=f"B{len(blocks) + 1}",
                        original_text=raw,
                        normalized_text=raw,
                        page=line.page,
                        section=section,
                        paragraph=len(blocks) + 1,
                        start_char=cursor,
                        end_char=cursor + len(raw),
                        block_type="table_row",
                        source_location=location,
                        metadata={
                            "row": row_number,
                            "source_req_id": source_id,
                            "requirement_text": requirement_text,
                            "bbox_y": [round(line.y0, 2), round(filtered[end].y1, 2)],
                        },
                    )
                )
                cursor += len(raw) + 2
                index = end + 1
                continue
            if index in consumed:
                cursor = _append_pdf_paragraph(blocks, paragraph_lines, section, cursor)
                paragraph_lines = []
                index += 1
                continue
            text = normalize_text(line.text)
            if _is_heading(text):
                cursor = _append_pdf_paragraph(blocks, paragraph_lines, section, cursor)
                paragraph_lines = []
                section = text
                blocks.append(
                    ParsedBlock(
                        block_id=f"B{len(blocks) + 1}",
                        original_text=text,
                        normalized_text=text,
                        page=line.page,
                        section=section,
                        paragraph=len(blocks) + 1,
                        start_char=cursor,
                        end_char=cursor + len(text),
                        block_type="heading",
                        source_location=f"page={line.page},section={section}",
                        metadata={"bbox_y": [round(line.y0, 2), round(line.y1, 2)]},
                    )
                )
                cursor += len(text) + 2
            else:
                if paragraph_lines and line.y0 - paragraph_lines[-1].y1 > 12:
                    cursor = _append_pdf_paragraph(blocks, paragraph_lines, section, cursor)
                    paragraph_lines = []
                paragraph_lines.append(line)
            index += 1
        cursor = _append_pdf_paragraph(blocks, paragraph_lines, section, cursor)
    return blocks


def _append_pdf_paragraph(
    blocks: list[ParsedBlock],
    lines: list[_PdfLine],
    section: str | None,
    cursor: int,
) -> int:
    if not lines:
        return cursor
    raw = "\n".join(line.text for line in lines)
    normalized = normalize_text(raw)
    if not normalized:
        return cursor
    location = f"page={lines[0].page}"
    blocks.append(
        ParsedBlock(
            block_id=f"B{len(blocks) + 1}",
            original_text=raw,
            normalized_text=normalized,
            page=lines[0].page,
            section=section,
            paragraph=len(blocks) + 1,
            start_char=cursor,
            end_char=cursor + len(normalized),
            block_type="paragraph",
            source_location=location,
            metadata={"bbox_y": [round(lines[0].y0, 2), round(lines[-1].y1, 2)]},
        )
    )
    return cursor + len(normalized) + 2


def _repeated_page_noise(page_lines: list[list[_PdfLine]]) -> set[tuple[int, float, str]]:
    candidates: Counter[str] = Counter()
    for lines in page_lines:
        seen: set[str] = set()
        for line in lines:
            text = normalize_text(line.text)
            if text and (line.y0 < 35 or line.y1 > line.page_height - 55):
                seen.add(text)
        candidates.update(seen)
    repeated = {text for text, count in candidates.items() if count >= 2}
    return {
        (line.page, round(line.y0, 1), normalize_text(line.text))
        for lines in page_lines
        for line in lines
        if normalize_text(line.text) in repeated
        and (line.y0 < 35 or line.y1 > line.page_height - 55)
    }


def _explicit_id_rows(
    lines: list[_PdfLine],
) -> tuple[list[tuple[int, int, str, str]], set[int]]:
    """Reconstruct two-column ID/Requirement rows, including vertically split IDs."""
    rows: list[tuple[int, int, str, str]] = []
    consumed: set[int] = set()
    index = 0
    while index < len(lines):
        header = lines[index]
        header_words = list(header.words)
        if (
            len(header_words) < 2
            or header_words[0][1].casefold() != "id"
            or header_words[1][1].casefold() not in {"requirement", "acceptance"}
        ):
            index += 1
            continue
        split_x = header_words[1][0] - 3
        consumed.add(index)
        index += 1
        while index < len(lines):
            line = lines[index]
            text = normalize_text(line.text)
            if _is_heading(text) or (
                len(line.words) >= 2
                and line.words[0][1].casefold() == "id"
                and line.words[1][1].casefold() in {"requirement", "acceptance"}
            ):
                break
            left = " ".join(word for x, word in line.words if x < split_x).strip()
            if not left:
                index += 1
                continue
            source_id = left
            row_start = index
            if _SOURCE_ID_PREFIX.fullmatch(left):
                for suffix_index in range(index + 1, min(index + 4, len(lines))):
                    next_left = " ".join(
                        word for x, word in lines[suffix_index].words if x < split_x
                    ).strip()
                    if next_left.isdigit():
                        source_id = f"{left}{next_left}"
                        break
                    if next_left:
                        break
            if not _SOURCE_ID.fullmatch(source_id):
                break
            right_parts: list[str] = []
            row_end = index
            scan = index
            while scan < len(lines):
                candidate = lines[scan]
                candidate_left = " ".join(
                    word for x, word in candidate.words if x < split_x
                ).strip()
                if scan > index and (
                    _SOURCE_ID.fullmatch(candidate_left)
                    or _SOURCE_ID_PREFIX.fullmatch(candidate_left)
                    or _is_heading(normalize_text(candidate.text))
                ):
                    break
                right = " ".join(word for x, word in candidate.words if x >= split_x).strip()
                if right:
                    right_parts.append(right)
                row_end = scan
                scan += 1
            requirement_text = normalize_text(" ".join(right_parts))
            if not requirement_text:
                break
            rows.append((row_start, row_end, source_id, requirement_text))
            consumed.update(range(row_start, row_end + 1))
            index = row_end + 1
    return rows, consumed


def _is_heading(text: str) -> bool:
    return bool(_HEADING.match(text)) and len(text) <= 120
