"""Strict source-knowledge contracts independent of generated artifacts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextUnit(KnowledgeModel):
    text_unit_id: str
    document_version_id: str
    block_id: str
    ordinal: int = Field(ge=1)
    unit_type: Literal["sentence", "bullet", "table_row", "clause"]
    text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    page: int | None = None
    section: str | None = None
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> "TextUnit":
        if self.end_char < self.start_char:
            raise ValueError("end_char must be greater than or equal to start_char")
        return self


class ChunkTextUnitLink(KnowledgeModel):
    chunk_id: str
    text_unit_id: str
    ordinal_in_chunk: int = Field(ge=1)


class LexicalKnowledgeProjection(KnowledgeModel):
    project: str
    document_id: str
    document_version_id: str
    text_units: list[TextUnit]
    chunk_links: list[ChunkTextUnitLink]
