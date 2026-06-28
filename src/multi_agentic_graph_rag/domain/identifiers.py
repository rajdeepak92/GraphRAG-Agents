"""Public identifier and version normalization rules."""

from __future__ import annotations

import hashlib
import re
from uuid import UUID

from multi_agentic_graph_rag.domain.errors import IdentifierError

_PUBLIC_ID_PATTERN = re.compile(r"^[A-Z]+-[A-Z0-9]+(?:-[A-Z0-9]+)*$")


def validate_public_id(value: str, *, prefix: str) -> str:
    """Validate a public traceability id such as REQ-000001."""

    normalized = value.strip().upper()

    if not normalized.startswith(f"{prefix}-"):
        msg = f"Expected {prefix}- public id. Got: {value!r}"
        raise IdentifierError(msg)

    if not _PUBLIC_ID_PATTERN.fullmatch(normalized):
        msg = f"Invalid public id format: {value!r}"
        raise IdentifierError(msg)

    return normalized


def validate_temporary_key(value: str, *, prefix: str) -> str:
    """Validate LLM temporary keys such as F1 or R1."""

    normalized = value.strip().upper()

    if not re.fullmatch(rf"{re.escape(prefix)}[1-9][0-9]*", normalized):
        msg = f"Invalid temporary {prefix} key: {value!r}"
        raise IdentifierError(msg)

    return normalized


def normalize_version(value: str) -> str:
    """Normalize a supplied document version without forcing numeric parsing."""

    normalized = " ".join(value.strip().split())

    if not normalized:
        msg = "document_version cannot be empty."
        raise IdentifierError(msg)

    return normalized.casefold()


def sha256_text(value: str) -> str:
    """Return a deterministic SHA-256 hash for normalized text."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_chunk_id(
    *,
    document_version_id: UUID,
    ordinal: int,
    content_hash: str,
) -> str:
    """Create a deterministic chunk id from document version, ordinal, and hash."""

    if ordinal < 1:
        msg = "Chunk ordinal must be >= 1."
        raise IdentifierError(msg)

    seed = f"{document_version_id}:{ordinal}:{content_hash}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20].upper()

    return f"CHUNK-{digest}"
