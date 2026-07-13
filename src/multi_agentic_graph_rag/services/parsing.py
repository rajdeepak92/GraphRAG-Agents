"""Document parsing for text, Markdown, DOCX, and PDF files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from multi_agentic_graph_rag.domain.schemas import ParsedBlock
from multi_agentic_graph_rag.observability.logging import RunLogger

PARSER_FINGERPRINT = "parsing-v1"


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
    return re.sub(r"\s+", " ", text).strip()


def parse_document(
    path: Path,
    logger: RunLogger | None = None,
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
                metadata={"style": paragraph.style.name if paragraph.style else None},
            )
        )
        cursor += len(raw) + 1
    return blocks


def _parse_pdf(path: Path) -> list[ParsedBlock]:
    """Parse pdf.

    Args:
        path (Path): Filesystem location authorized for this operation.

    Returns:
        list[ParsedBlock]: The typed result produced by the operation.

    Side Effects:
        May create or atomically replace files in the configured artifact boundary.
    """
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
