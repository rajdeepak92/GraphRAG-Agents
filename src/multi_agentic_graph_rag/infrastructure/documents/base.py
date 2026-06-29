"""Document parser interface contracts.

This module defines the parser protocol used by the parser registry.
It intentionally contains no PDF, DOCX, TXT or Markdown parsing logic.
Concrete parser implementations live in sibling modules:

- pdf_parser.py
- docx_parser.py
- text_parser.py
- markdown_parser.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from multi_agentic_graph_rag.domain.documents import ParsedDocument


class DocumentParser(Protocol):
    """Structural contract implemented by every document parser.

    A parser satisfies this protocol when it exposes:

    - parser_name
    - parser_version
    - supported_extensions
    - parse(path) -> ParsedDocument

    Concrete parser classes do not need to inherit from this protocol.
    Static type checkers accept them if their structure matches.
    """

    parser_name: str
    parser_version: str
    supported_extensions: frozenset[str]

    def parse(self, path: Path) -> ParsedDocument:
        """Parse a supported source file into a normalized ParsedDocument."""
        ...
