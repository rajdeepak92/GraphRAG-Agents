# src/multi_agentic_graph_rag/infrastructure/documents/chunker.py

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_agentic_graph_rag.domain.chunks import Chunk
from multi_agentic_graph_rag.domain.documents import ParsedBlock
from multi_agentic_graph_rag.infrastructure.documents.normalization import sha256_text


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 150
    minimum_chunk_size: int = 100
    maximum_chunk_size: int = 1800
    preserve_page_boundaries: bool = True
    preserve_heading_context: bool = True
    length_strategy: str = "characters"

    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StructureAwareChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(
        self,
        *,
        document_version_id: UUID,
        source_checksum: str,
        blocks: list[ParsedBlock],
    ) -> list[Chunk]:
        grouped_blocks = self._group_blocks(blocks)

        chunks: list[Chunk] = []
        ordinal = 1

        for group in grouped_blocks:
            group_text = "\n\n".join(block.normalized_text for block in group).strip()
            if not group_text:
                continue

            if len(group_text) > self.config.maximum_chunk_size:
                split_texts = self._splitter.split_text(group_text)
            else:
                split_texts = [group_text]

            for split_text in split_texts:
                normalized = split_text.strip()
                if len(normalized) < self.config.minimum_chunk_size and chunks:
                    continue

                raw_text = self._best_effort_raw_text(group, normalized)
                content_hash = sha256_text(normalized)

                chunks.append(
                    Chunk(
                        chunk_id=self._chunk_id(document_version_id, ordinal, content_hash),
                        ordinal=ordinal,
                        document_version_id=document_version_id,
                        source_checksum=source_checksum,
                        page_start=_min_page(group),
                        page_end=_max_page(group),
                        section_path=tuple(group[0].section_path),
                        character_start=min(block.character_start for block in group),
                        character_end=max(block.character_end for block in group),
                        raw_text=raw_text,
                        normalized_text=normalized,
                        content_hash=content_hash,
                    )
                )
                ordinal += 1

        return chunks

    def _group_blocks(self, blocks: list[ParsedBlock]) -> list[list[ParsedBlock]]:
        groups: list[list[ParsedBlock]] = []
        current: list[ParsedBlock] = []

        for block in blocks:
            if not current:
                current = [block]
                continue

            previous = current[-1]

            boundary_changed = False

            if self.config.preserve_page_boundaries:
                boundary_changed = boundary_changed or block.page_number != previous.page_number

            if self.config.preserve_heading_context:
                boundary_changed = boundary_changed or block.section_path != previous.section_path

            current_text = "\n\n".join(existing_block.normalized_text for existing_block in current)
            projected_size = len(current_text) + len(block.normalized_text)

            if boundary_changed or projected_size > self.config.chunk_size:
                groups.append(current)
                current = [block]
            else:
                current.append(block)

        if current:
            groups.append(current)

        return groups

    @staticmethod
    def _best_effort_raw_text(blocks: list[ParsedBlock], normalized: str) -> str:
        raw = "\n\n".join(block.raw_text for block in blocks).strip()
        return raw if raw else normalized

    @staticmethod
    def _chunk_id(document_version_id: UUID, ordinal: int, content_hash: str) -> str:
        seed = f"{document_version_id}:{ordinal}:{content_hash}"
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
        return f"CHUNK-{ordinal:06d}-{digest}"


def _min_page(blocks: Iterable[ParsedBlock]) -> int | None:
    pages = [block.page_number for block in blocks if block.page_number is not None]
    return min(pages) if pages else None


def _max_page(blocks: Iterable[ParsedBlock]) -> int | None:
    pages = [block.page_number for block in blocks if block.page_number is not None]
    return max(pages) if pages else None
