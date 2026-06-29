from __future__ import annotations

from pathlib import Path

from multi_agentic_graph_rag.domain.documents import ParsedDocument
from multi_agentic_graph_rag.infrastructure.documents.base import DocumentParser
from multi_agentic_graph_rag.infrastructure.documents.docx_parser import DocxParser
from multi_agentic_graph_rag.infrastructure.documents.markdown_parser import MarkdownParser
from multi_agentic_graph_rag.infrastructure.documents.pdf_parser import PdfParser
from multi_agentic_graph_rag.infrastructure.documents.text_parser import TextParser


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser] | None = None) -> None:
        self._by_extension: dict[str, DocumentParser] = {}

        for parser in parsers or [PdfParser(), DocxParser(), TextParser(), MarkdownParser()]:
            for extension in parser.supported_extensions:
                self._by_extension[extension.lower()] = parser

    def get_parser(self, path: Path) -> DocumentParser:
        extension = path.suffix.lower()
        parser = self._by_extension.get(extension)

        if parser is None:
            supported = ", ".join(sorted(self._by_extension))
            raise ValueError(
                f"Unsupported document type: {extension or '<no extension>'}. "
                f"Supported extensions: {supported}"
            )

        return parser

    def parse(self, path: Path) -> ParsedDocument:
        return self.get_parser(path).parse(path)
