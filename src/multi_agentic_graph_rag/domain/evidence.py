"""Evidence and source-trace contracts."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceReference(BaseModel):
    """Verified or candidate evidence location."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_version_id: UUID | None = None
    exact_quote: str = Field(min_length=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceReference:
        if self.character_start is None or self.character_end is None:
            return self

        if self.character_end <= self.character_start:
            msg = "character_end must be greater than character_start."
            raise ValueError(msg)

        return self
