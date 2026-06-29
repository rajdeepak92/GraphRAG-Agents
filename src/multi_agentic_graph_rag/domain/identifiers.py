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
