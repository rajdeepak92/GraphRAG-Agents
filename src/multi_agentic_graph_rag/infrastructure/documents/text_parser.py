# src/multi_agentic_graph_rag/infrastructure/documents/text_parser.py

from __future__ import annotations

from pathlib import Path

from charset_normalizer import from_bytes

from multi_agentic_graph_rag.domain.documents import ParsedBlock, ParsedDocument
from multi_agentic_graph_rag.infrastructure.documents.normalization import (
    normalize_text,
    sha256_file,
)


class TextParser:
    parser_name = "validated-text-reader"
    parser_version = "1.0"
    supported_extensions = frozenset({".txt"})

    def parse(self, path: Path) -> ParsedDocument:
        path = path.resolve()
        checksum = sha256_file(path)

        raw_bytes = path.read_bytes()
        match = from_bytes(raw_bytes).best()
        if match is None:
            raise ValueError(f"Could not detect text encoding: {path}")

        full_text = str(match)
        blocks: list[ParsedBlock] = []
        cursor = 0

        for index, raw_para in enumerate(full_text.split("\n\n"), start=1):
            normalized = normalize_text(raw_para)
            if not normalized:
                cursor += len(raw_para) + 2
                continue

            start = cursor
            end = start + len(raw_para)
            cursor = end + 2

            blocks.append(
                ParsedBlock(
                    source_path=str(path),
                    source_checksum=checksum,
                    page_number=None,
                    section_path=(),
                    paragraph_number=index,
                    character_start=start,
                    character_end=end,
                    raw_text=raw_para,
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
