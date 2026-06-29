"""Document parsing for text, Markdown, DOCX, and PDF files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from multi_agentic_graph_rag.domain.schemas import ParsedBlock

PARSER_FINGERPRINT = "parsing-v1"


def checksum_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_document(path: Path) -> tuple[list[ParsedBlock], str]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _parse_text(path), PARSER_FINGERPRINT
    if suffix == ".docx":
        return _parse_docx(path), PARSER_FINGERPRINT
    if suffix == ".pdf":
        return _parse_pdf(path), PARSER_FINGERPRINT
    raise ValueError(f"Unsupported document type: {suffix}")


def _parse_text(path: Path) -> list[ParsedBlock]:
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
            )
        )
    return blocks


def _parse_docx(path: Path) -> list[ParsedBlock]:
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
                metadata={"style": paragraph.style.name if paragraph.style else None},
            )
        )
        cursor += len(raw) + 1
    return blocks


def _parse_pdf(path: Path) -> list[ParsedBlock]:
    import fitz  # type: ignore[import-untyped]

    blocks: list[ParsedBlock] = []
    cursor = 0
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            for part in re.split(r"\n\s*\n", text):
                raw = part.strip()
                if not raw:
                    continue
                blocks.append(
                    ParsedBlock(
                        block_id=f"B{len(blocks) + 1}",
                        original_text=raw,
                        normalized_text=normalize_text(raw),
                        page=page_index,
                        paragraph=len(blocks) + 1,
                        start_char=cursor,
                        end_char=cursor + len(raw),
                    )
                )
                cursor += len(raw) + 1
    return blocks
