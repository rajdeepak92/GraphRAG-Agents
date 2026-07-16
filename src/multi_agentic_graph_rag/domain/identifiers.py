"""Stable project/item identities for the simplified workflow."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from uuid import uuid4

from uuid_utils import uuid7

_UNSAFE = re.compile(r"[^a-z0-9]+")


def stable_token(*parts: object, length: int = 20) -> str:
    """Return an uppercase SHA-256 prefix for deterministic identities."""
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length].upper()


def normalize_project(value: str) -> str:
    """Normalize a project only for database/collection/path compatibility."""
    normalized = _UNSAFE.sub("-", value.strip().lower()).strip("-")
    return normalized or "project"


def new_run_id(project: str) -> str:
    """Allocate an operational run ID that is never used as artifact identity."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"RUN-{timestamp}-{stable_token(project, uuid4().hex, length=6)}"


def make_chunk_id(
    *,
    chunk_text: str,
    start_char: int,
    end_char: int,
    source_location: str | None,
) -> str:
    """Build a stable chunk ID independent of project, run, and document identity."""
    return "CHK-" + stable_token(
        chunk_text,
        start_char,
        end_char,
        source_location or "",
        length=20,
    )


def new_requirement_id() -> str:
    """Allocate a permanent requirement UUIDv7."""
    return f"REQ-{str(uuid7()).upper()}"


def new_story_id() -> str:
    """Allocate a permanent user-story UUIDv7."""
    return f"US-{str(uuid7()).upper()}"


def new_scenario_id() -> str:
    """Allocate a permanent scenario UUIDv7."""
    return f"TS-{str(uuid7()).upper()}"


def make_criterion_id(story_id: str, ordinal: int, content: str) -> str:
    """Build a permanent criterion ID under its canonical story."""
    return f"AC-{stable_token(story_id, ordinal, content, length=18)}"


def make_evidence_id(parent_id: str, chunk_id: str, start_char: int, end_char: int) -> str:
    """Build a deterministic evidence ID."""
    return f"EVD-{stable_token(parent_id, chunk_id, start_char, end_char, length=20)}"


def make_entity_id(project: str, normalized_name: str, entity_type: str) -> str:
    """Build a project-scoped canonical entity ID."""
    token = stable_token(normalize_project(project), normalized_name, entity_type, length=20)
    return f"ENT-{token}"


def make_relationship_id(
    *,
    project: str,
    chunk_id: str,
    requirement_text_hash: str,
    source_entity_id: str,
    relationship_type: str,
    target_entity_id: str,
    evidence_hash: str,
) -> str:
    """Build the deterministic semantic projection ID."""
    return "REL-" + stable_token(
        normalize_project(project),
        chunk_id,
        requirement_text_hash,
        source_entity_id,
        relationship_type,
        target_entity_id,
        evidence_hash,
        length=24,
    )


def make_checkpoint_thread_id(project: str, run_id: str, stage: str) -> str:
    """Derive the internal LangGraph thread identity."""
    return f"{normalize_project(project)}:{run_id}:{stage}"


__all__ = [
    "make_checkpoint_thread_id",
    "make_chunk_id",
    "make_criterion_id",
    "make_entity_id",
    "make_evidence_id",
    "make_relationship_id",
    "new_requirement_id",
    "new_run_id",
    "new_scenario_id",
    "new_story_id",
    "normalize_project",
    "stable_token",
]
