# src/multi_agentic_graph_rag/infrastructure/documents/pdf_parser.py

from __future__ import annotations

from pathlib import Path

import fitz  # type: ignore[import-untyped]

from multi_agentic_graph_rag.domain.documents import ParsedBlock, ParsedDocument
from multi_agentic_graph_rag.infrastructure.documents.normalization import (
    normalize_text,
    sha256_file,
)


class PdfParser:
    parser_name = "pymupdf"
    parser_version = fitz.VersionBind
    supported_extensions = frozenset({".pdf"})

    def parse(self, path: Path) -> ParsedDocument:
        path = path.resolve()
        checksum = sha256_file(path)

        doc = fitz.open(path)
        if doc.is_encrypted:
            raise ValueError(f"Encrypted PDF is not supported: {path}")

        blocks: list[ParsedBlock] = []
        cursor = 0

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            page_blocks = page.get_text("blocks", sort=True)

            for block in page_blocks:
                # PyMuPDF block tuple commonly includes:
                # x0, y0, x1, y1, text, block_no, block_type
                if len(block) < 7:
                    continue

                raw_text = str(block[4] or "")
                block_type = int(block[6])

                # block_type 0 is text.
                if block_type != 0 or not raw_text.strip():
                    continue

                normalized = normalize_text(raw_text)
                if not normalized:
                    continue

                start = cursor
                end = start + len(raw_text)
                cursor = end + 1

                blocks.append(
                    ParsedBlock(
                        source_path=str(path),
                        source_checksum=checksum,
                        page_number=page_index + 1,
                        section_path=(),
                        paragraph_number=None,
                        character_start=start,
                        character_end=end,
                        raw_text=raw_text,
                        normalized_text=normalized,
                        parser_name=self.parser_name,
                        parser_version=self.parser_version,
                        metadata={
                            "bbox": [
                                float(block[0]),
                                float(block[1]),
                                float(block[2]),
                                float(block[3]),
                            ],
                            "block_number": int(block[5]),
                        },
                    )
                )

        return ParsedDocument(
            source_path=str(path),
            source_checksum=checksum,
            parser_name=self.parser_name,
            parser_version=self.parser_version,
            blocks=tuple(blocks),
        )
