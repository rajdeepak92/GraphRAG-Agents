"""Identifier generation for externally visible ingestion records."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from multi_agentic_graph_rag.common_defs import IdentifierPrefix

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
    return f"{IdentifierPrefix.RUN.value}{timestamp}-{token}"


def document_id(project: str, logical_name: str) -> str:
    return (
        f"{IdentifierPrefix.DOC.value}{safe_slug(project)}-{stable_token(logical_name, length=10)}"
    )


def document_version_id(document_identifier: str, version: str, checksum: str) -> str:
    return f"DV-{stable_token(document_identifier, version, checksum, length=18)}"


def chunk_id(document_version_identifier: str, ordinal: int, text: str) -> str:
    return (
        f"CHUNK-{ordinal:04d}-{stable_token(document_version_identifier, ordinal, text, length=12)}"
    )


def fact_id(project: str, version: str, text: str, ordinal: int) -> str:
    return (
        f"{IdentifierPrefix.FACT.value}{stable_token(project, version, text, ordinal, length=14)}"
    )


def requirement_id(project: str, version: str, text: str, ordinal: int) -> str:
    return f"{IdentifierPrefix.REQ.value}{stable_token(project, version, text, ordinal, length=14)}"


def canonical_fact_id(project: str, document_identifier: str, normalized_text: str) -> str:
    return (
        f"{IdentifierPrefix.FACTCAN.value}"
        f"{stable_token(project, document_identifier, normalized_text, length=14)}"
    )


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
    return f"{IdentifierPrefix.FACT.value}{token}"


def requirement_lineage_id(
    project: str,
    document_identifier: str,
    requirement_key: str,
) -> str:
    return (
        f"{IdentifierPrefix.REQ.value}"
        f"{stable_token(project, document_identifier, requirement_key, length=14)}"
    )


def requirement_revision_id(requirement_identifier: str, normalized_statement: str) -> str:
    return (
        f"{IdentifierPrefix.REQREV.value}"
        f"{stable_token(requirement_identifier, normalized_statement, length=14)}"
    )


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
    return f"{IdentifierPrefix.REQEVID.value}{token}"


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
    return f"{IdentifierPrefix.US.value}{token}"


def test_scenario_id(
    project: str,
    story_identifier: str,
    normalized_title: str,
    ordinal: int,
) -> str:
    token = stable_token(
        project,
        story_identifier,
        normalized_title,
        ordinal,
        length=14,
    )
    return f"{IdentifierPrefix.SC.value}{token}"


test_scenario_id.__test__ = False  # type: ignore[attr-defined]


def user_story_evidence_id(
    *,
    story_identifier: str,
    document_version_identifier: str,
    chunk_identifier: str,
) -> str:
    token = stable_token(
        story_identifier,
        document_version_identifier,
        chunk_identifier,
        length=14,
    )
    return f"{IdentifierPrefix.USEVID.value}{token}"


def test_scenario_evidence_id(
    *,
    scenario_identifier: str,
    document_version_identifier: str,
    chunk_identifier: str,
) -> str:
    token = stable_token(
        scenario_identifier,
        document_version_identifier,
        chunk_identifier,
        length=14,
    )
    return f"{IdentifierPrefix.SCEVID.value}{token}"


test_scenario_evidence_id.__test__ = False  # type: ignore[attr-defined]


def text_unit_id(
    document_version_identifier: str,
    start_char: int,
    end_char: int,
    text: str,
) -> str:
    token = stable_token(document_version_identifier, start_char, end_char, text, length=14)
    return f"{IdentifierPrefix.TEXTUNIT.value}{token}"


def entity_id(project: str, normalized_name: str, entity_type: str) -> str:
    return (
        f"{IdentifierPrefix.ENTITY.value}"
        f"{stable_token(project, entity_type, normalized_name, length=14)}"
    )


def entity_mention_id(
    entity_identifier: str,
    chunk_identifier: str,
    surface_text: str,
) -> str:
    return (
        f"{IdentifierPrefix.MENTION.value}"
        f"{stable_token(entity_identifier, chunk_identifier, surface_text, length=14)}"
    )


def assertion_key(
    *,
    project: str,
    subject_entity_identifier: str,
    predicate: str,
    object_entity_identifier: str | None,
    object_literal: str | None,
    modality: str,
    polarity: str,
    condition: str | None,
) -> str:
    """Project-scoped semantic identity for an assertion.

    Modality is part of the identity: ``shall X`` and ``may X`` are different
    obligations and must not merge. Explicitness stays out; an explicit and an
    inferred statement of the same claim merge with explicit winning.
    """
    return stable_token(
        project,
        subject_entity_identifier,
        predicate,
        object_entity_identifier or "",
        object_literal or "",
        modality,
        polarity,
        condition or "",
        length=16,
    )


def assertion_id(assertion_key_token: str, document_version_identifier: str) -> str:
    return (
        f"{IdentifierPrefix.ASSERTION.value}"
        f"{stable_token(assertion_key_token, document_version_identifier, length=14)}"
    )


def assertion_lineage_key(
    *,
    project: str,
    subject_entity_identifier: str,
    predicate: str,
    object_entity_identifier: str | None,
) -> str:
    """Cross-version semantic-slot identity for an assertion.

    Deliberately coarser than :func:`assertion_key`: it fixes only the stable
    subject / predicate / object-entity identity, so a scalar value, modality,
    polarity, or normalized condition can change *within the same lineage* (a new
    revision of the same claim) instead of spawning an unrelated lineage. Assertions
    with a literal object (no object entity) share a lineage per subject+predicate,
    which is what lets ``threshold 70C`` -> ``threshold 80C`` supersede rather than
    duplicate.
    """
    return stable_token(
        project,
        subject_entity_identifier,
        predicate.upper(),
        object_entity_identifier or "",
        length=16,
    )


def assertion_evidence_id(
    *,
    assertion_identifier: str,
    chunk_identifier: str,
    quote: str,
    start_char: int,
    end_char: int,
) -> str:
    token = stable_token(
        assertion_identifier,
        chunk_identifier,
        quote,
        start_char,
        end_char,
        length=14,
    )
    return f"{IdentifierPrefix.ASTEVID.value}{token}"


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
    return f"{IdentifierPrefix.REQDELTA.value}{token}"
