# src/multi_agentic_graph_rag/infrastructure/documents/markdown_parser.py

from __future__ import annotations

import re
from pathlib import Path

from charset_normalizer import from_bytes

from multi_agentic_graph_rag.domain.documents import ParsedBlock, ParsedDocument
from multi_agentic_graph_rag.infrastructure.documents.normalization import (
    normalize_text,
    sha256_file,
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownParser:
    parser_name = "markdown-aware-text-reader"
    parser_version = "1.0"
    supported_extensions = frozenset({".md", ".markdown"})

    def parse(self, path: Path) -> ParsedDocument:
        path = path.resolve()
        checksum = sha256_file(path)

        raw_bytes = path.read_bytes()
        match = from_bytes(raw_bytes).best()
        if match is None:
            raise ValueError(f"Could not detect Markdown encoding: {path}")

        text = str(match)
        blocks: list[ParsedBlock] = []
        section_path: list[str] = []
        cursor = 0

        for index, line in enumerate(text.splitlines(), start=1):
            raw_text = line
            normalized = normalize_text(raw_text)

            heading = _HEADING_RE.match(raw_text)
            if heading:
                level = len(heading.group(1))
                title = normalize_text(heading.group(2))
                section_path = [*section_path[: level - 1], title]

            if not normalized:
                cursor += len(raw_text) + 1
                continue

            start = cursor
            end = start + len(raw_text)
            cursor = end + 1

            blocks.append(
                ParsedBlock(
                    source_path=str(path),
                    source_checksum=checksum,
                    page_number=None,
                    section_path=tuple(section_path),
                    paragraph_number=index,
                    character_start=start,
                    character_end=end,
                    raw_text=raw_text,
                    normalized_text=normalized,
                    parser_name=self.parser_name,
                    parser_version=self.parser_version,
                )
            )

        return ParsedDocument(
            source_path=str(path),
            source_checksum=checksum,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            blocks=tuple(blocks),
        )
