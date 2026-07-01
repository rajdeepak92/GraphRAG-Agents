"""Identifier generation for externally visible ingestion records."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

_SAFE = re.compile(r"[^A-Za-z0-9]+")


def stable_token(*parts: object, length: int = 16) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length].upper()


def safe_slug(value: str) -> str:
    slug = _SAFE.sub("-", value.strip()).strip("-").upper()
    return slug or "ITEM"


def run_id(project: str, document: Path, version: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    token = stable_token(project, document.name, version, timestamp, uuid4().hex, length=8)
    return f"RUN-{timestamp}-{token}"


def document_id(project: str, logical_name: str) -> str:
    return f"DOC-{safe_slug(project)}-{stable_token(logical_name, length=10)}"


def document_version_id(document_identifier: str, version: str, checksum: str) -> str:
    return f"DV-{stable_token(document_identifier, version, checksum, length=18)}"


def chunk_id(document_version_identifier: str, ordinal: int, text: str) -> str:
    return (
        f"CHUNK-{ordinal:04d}-{stable_token(document_version_identifier, ordinal, text, length=12)}"
    )


def fact_id(project: str, version: str, text: str, ordinal: int) -> str:
    return f"FACT-{stable_token(project, version, text, ordinal, length=14)}"


def requirement_id(project: str, version: str, text: str, ordinal: int) -> str:
    return f"REQ-{stable_token(project, version, text, ordinal, length=14)}"


def canonical_fact_id(project: str, document_identifier: str, normalized_text: str) -> str:
    return f"FACTCAN-{stable_token(project, document_identifier, normalized_text, length=14)}"


def fact_occurrence_id(
    *,
    project: str,
    document_version_identifier: str,
    chunk_identifier: str,
    text: str,
    quote: str,
    start_char: int,
    end_char: int,
) -> str:
    token = stable_token(
        project,
        document_version_identifier,
        chunk_identifier,
        text,
        quote,
        start_char,
        end_char,
        length=14,
    )
    return f"FACT-{token}"


def requirement_lineage_id(
    project: str,
    document_identifier: str,
    requirement_key: str,
) -> str:
    return f"REQ-{stable_token(project, document_identifier, requirement_key, length=14)}"


def requirement_revision_id(requirement_identifier: str, normalized_statement: str) -> str:
    return f"REQREV-{stable_token(requirement_identifier, normalized_statement, length=14)}"


def requirement_evidence_id(
    *,
    requirement_identifier: str,
    revision_identifier: str,
    document_version_identifier: str,
    chunk_identifier: str,
    quote: str,
    start_char: int,
    end_char: int,
) -> str:
    token = stable_token(
        requirement_identifier,
        revision_identifier,
        document_version_identifier,
        chunk_identifier,
        quote,
        start_char,
        end_char,
        length=14,
    )
    return f"REQEVID-{token}"


def user_story_id(
    project: str,
    requirement_identifier: str,
    normalized_title: str,
    ordinal: int,
) -> str:
    token = stable_token(
        project,
        requirement_identifier,
        normalized_title,
        ordinal,
        length=14,
    )
    return f"US-{token}"


def requirement_delta_event_id(
    *,
    event_type: str,
    requirement_identifier: str,
    revision_identifier: str | None,
    previous_revision_identifier: str | None,
    document_version_identifier: str,
) -> str:
    token = stable_token(
        event_type,
        requirement_identifier,
        revision_identifier or "",
        previous_revision_identifier or "",
        document_version_identifier,
        length=14,
    )
    return f"REQDELTA-{token}"
