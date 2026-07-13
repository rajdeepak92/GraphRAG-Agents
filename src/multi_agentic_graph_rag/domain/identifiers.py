"""Identifier generation for externally visible ingestion records."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from uuid_utils import uuid7

from multi_agentic_graph_rag.common_defs import IdentifierPrefix

_SAFE = re.compile(r"[^A-Za-z0-9]+")
_UUID7_ID = re.compile(
    r"^(?:REQ|REQREV|REQEVID)-[0-9A-F]{8}-[0-9A-F]{4}-7[0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$",
    re.I,
)


def stable_token(*parts: object, length: int = 16) -> str:
    """Execute the stable token operation within its declared architectural boundary.

    Args:
        parts (object): Parts required by the operation's typed contract.
        length (int): Length required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length].upper()


def safe_slug(value: str) -> str:
    """Execute the safe slug operation within its declared architectural boundary.

    Args:
        value (str): Value required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    slug = _SAFE.sub("-", value.strip()).strip("-").upper()
    return slug or "ITEM"


def run_id(project: str, document: Path, version: str) -> str:
    """Run id.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document (Path): Document required by the operation's typed contract.
        version (str): Document version label within the project scope.

    Returns:
        str: The typed result produced by the operation.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    token = stable_token(project, document.name, version, timestamp, uuid4().hex, length=8)
    return f"{IdentifierPrefix.RUN.value}{timestamp}-{token}"


def document_id(project: str, logical_name: str) -> str:
    """Execute the document id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        logical_name (str): Logical name required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
        f"{IdentifierPrefix.DOC.value}{safe_slug(project)}-{stable_token(logical_name, length=10)}"
    )


def document_version_id(document_identifier: str, version: str, checksum: str) -> str:
    """Execute the document version id operation within its declared architectural boundary.

    Args:
        document_identifier (str): Canonical document identifier used as a safe operational anchor.
        version (str): Document version label within the project scope.
        checksum (str): Checksum required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return f"DV-{stable_token(document_identifier, version, checksum, length=18)}"


def chunk_id(document_version_identifier: str, ordinal: int, text: str) -> str:
    """Execute the chunk id operation within its declared architectural boundary.

    Args:
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.
        ordinal (int): Bounded ordinal used for deterministic processing.
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
        f"CHUNK-{ordinal:04d}-{stable_token(document_version_identifier, ordinal, text, length=12)}"
    )


def fact_id(project: str, version: str, text: str, ordinal: int) -> str:
    """Execute the fact id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        version (str): Document version label within the project scope.
        text (str): Input text processed in memory and excluded from diagnostic logs.
        ordinal (int): Bounded ordinal used for deterministic processing.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
        f"{IdentifierPrefix.FACT.value}{stable_token(project, version, text, ordinal, length=14)}"
    )


def requirement_id(project: str, version: str, text: str, ordinal: int) -> str:
    """Execute the requirement id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        version (str): Document version label within the project scope.
        text (str): Input text processed in memory and excluded from diagnostic logs.
        ordinal (int): Bounded ordinal used for deterministic processing.

    Returns:
        str: The typed result produced by the operation.
    """
    return f"{IdentifierPrefix.REQ.value}{stable_token(project, version, text, ordinal, length=14)}"


def canonical_fact_id(project: str, document_identifier: str, normalized_text: str) -> str:
    """Execute the canonical fact id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_identifier (str): Canonical document identifier used as a safe operational anchor.
        normalized_text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the fact occurrence id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.
        chunk_identifier (str): Canonical chunk identifier used as a safe operational anchor.
        text (str): Input text processed in memory and excluded from diagnostic logs.
        quote (str): Quote required by the operation's typed contract.
        start_char (int): Start char required by the operation's typed contract.
        end_char (int): End char required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the requirement lineage id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        document_identifier (str): Canonical document identifier used as a safe operational anchor.
        requirement_key (str): Requirement key required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
        f"{IdentifierPrefix.REQ.value}"
        f"{stable_token(project, document_identifier, requirement_key, length=14)}"
    )


def new_requirement_id() -> str:
    """Allocate a globally unique, immutable canonical requirement identifier."""
    return f"{IdentifierPrefix.REQ.value}{str(uuid7()).upper()}"


def new_requirement_revision_id() -> str:
    """Allocate a globally unique identifier for a canonical requirement revision."""
    return f"{IdentifierPrefix.REQREV.value}{str(uuid7()).upper()}"


def new_requirement_evidence_id() -> str:
    """Allocate a globally unique identifier for one evidence occurrence."""
    return f"{IdentifierPrefix.REQEVID.value}{str(uuid7()).upper()}"


def requirement_evidence_occurrence_key(
    *,
    document_version_identifier: str,
    chunk_identifier: str,
    quote: str,
    start_char: int,
    end_char: int,
) -> str:
    """Stable lookup key used only to reuse an already-allocated evidence UUID."""
    return stable_token(
        document_version_identifier,
        chunk_identifier,
        quote,
        start_char,
        end_char,
        length=32,
    )


def is_requirement_uuid7(value: str) -> bool:
    """Return whether requirement uuid7.

    Args:
        value (str): Value required by the operation's typed contract.

    Returns:
        bool: The typed result produced by the operation.
    """
    return bool(_UUID7_ID.fullmatch(value.strip()))


def requirement_revision_id(requirement_identifier: str, normalized_statement: str) -> str:
    """Execute the requirement revision id operation within its declared architectural boundary.

    Args:
        requirement_identifier (str): Canonical requirement identifier used as a safe operational
                                      anchor.
        normalized_statement (str): Normalized statement required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the requirement evidence id operation within its declared architectural boundary.

    Args:
        requirement_identifier (str): Canonical requirement identifier used as a safe operational
                                      anchor.
        revision_identifier (str): Canonical revision identifier used as a safe operational anchor.
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.
        chunk_identifier (str): Canonical chunk identifier used as a safe operational anchor.
        quote (str): Quote required by the operation's typed contract.
        start_char (int): Start char required by the operation's typed contract.
        end_char (int): End char required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the user story id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        requirement_identifier (str): Canonical requirement identifier used as a safe operational
                                      anchor.
        normalized_title (str): Normalized title required by the operation's typed contract.
        ordinal (int): Bounded ordinal used for deterministic processing.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the test scenario id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        story_identifier (str): Canonical story identifier used as a safe operational anchor.
        normalized_title (str): Normalized title required by the operation's typed contract.
        ordinal (int): Bounded ordinal used for deterministic processing.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the user story evidence id operation within its declared architectural boundary.

    Args:
        story_identifier (str): Canonical story identifier used as a safe operational anchor.
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.
        chunk_identifier (str): Canonical chunk identifier used as a safe operational anchor.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the test scenario evidence id operation within its declared architectural boundary.

    Args:
        scenario_identifier (str): Canonical scenario identifier used as a safe operational anchor.
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.
        chunk_identifier (str): Canonical chunk identifier used as a safe operational anchor.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the text unit id operation within its declared architectural boundary.

    Args:
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.
        start_char (int): Start char required by the operation's typed contract.
        end_char (int): End char required by the operation's typed contract.
        text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
    token = stable_token(document_version_identifier, start_char, end_char, text, length=14)
    return f"{IdentifierPrefix.TEXTUNIT.value}{token}"


def entity_id(project: str, normalized_name: str, entity_type: str) -> str:
    """Execute the entity id operation within its declared architectural boundary.

    Args:
        project (str): Project scope that isolates persistence and retrieval.
        normalized_name (str): Normalized name required by the operation's typed contract.
        entity_type (str): Entity type required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
        f"{IdentifierPrefix.ENTITY.value}"
        f"{stable_token(project, entity_type, normalized_name, length=14)}"
    )


def entity_mention_id(
    entity_identifier: str,
    chunk_identifier: str,
    surface_text: str,
) -> str:
    """Execute the entity mention id operation within its declared architectural boundary.

    Args:
        entity_identifier (str): Canonical entity identifier used as a safe operational anchor.
        chunk_identifier (str): Canonical chunk identifier used as a safe operational anchor.
        surface_text (str): Input text processed in memory and excluded from diagnostic logs.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the assertion id operation within its declared architectural boundary.

    Args:
        assertion_key_token (str): Assertion key token required by the operation's typed contract.
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the assertion evidence id operation within its declared architectural boundary.

    Args:
        assertion_identifier (str): Canonical assertion identifier used as a safe operational
                                    anchor.
        chunk_identifier (str): Canonical chunk identifier used as a safe operational anchor.
        quote (str): Quote required by the operation's typed contract.
        start_char (int): Start char required by the operation's typed contract.
        end_char (int): End char required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
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
    """Execute the requirement delta event id operation within its declared architectural boundary.

    Args:
        event_type (str): Event type required by the operation's typed contract.
        requirement_identifier (str): Canonical requirement identifier used as a safe operational
                                      anchor.
        revision_identifier (str | None): Canonical revision identifier used as a safe operational
                                          anchor.
        previous_revision_identifier (str | None): Canonical previous revision identifier used as a
                                                   safe operational anchor.
        document_version_identifier (str): Canonical document version identifier used as a safe
                                           operational anchor.

    Returns:
        str: The typed result produced by the operation.
    """
    token = stable_token(
        event_type,
        requirement_identifier,
        revision_identifier or "",
        previous_revision_identifier or "",
        document_version_identifier,
        length=14,
    )
    return f"{IdentifierPrefix.REQDELTA.value}{token}"
