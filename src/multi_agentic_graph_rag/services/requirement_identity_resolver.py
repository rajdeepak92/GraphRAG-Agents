"""Deterministic requirement identity: signature, lineage, and revision.

This is the identity authority for requirement discovery. The LLM proposes atomic
requirement candidates; this module — never the LLM key — decides the permanent
lineage. A requirement's *lineage signature* preserves entity discriminators
(``Sensor-1`` vs ``Sensor-2``, ``RS485``, register/API names) while masking
mutable parameters (thresholds, timeouts, percentages, durations, dates, quoted
values), and it is scoped by the requirement *family* (its normalized type). Two
consequences follow, both deterministic:

* distinct atomic requirements (Sensor-1/2/3, or a Business Requirement vs an
  Acceptance Criterion with similar wording) get **distinct** lineages;
* the same obligation whose mutable value changed (70°C → 80°C) keeps the **same**
  lineage but a new revision (the revision is keyed on the full normalized text).

``requirement_key`` from the LLM is retained only as a non-authoritative hint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from multi_agentic_graph_rag.domain.identifiers import (
    requirement_lineage_id,
    requirement_revision_id,
)

_WHITESPACE = re.compile(r"\s+")
_QUOTED_VALUE = re.compile(r"(['\"])(?:\\.|(?!\1).)*\1")
# Mutable single-token parameters. Order matters: these are matched before a token
# is considered an entity discriminator, so "70c"/"80%"/"5s" mask, "sensor-1" stays.
_TEMPERATURE = re.compile(r"^\d+(?:\.\d+)?(?:degrees?|deg|°)?[cf]$")
_PERCENT = re.compile(r"^\d+(?:\.\d+)?%$")
_DURATION = re.compile(
    r"^\d+(?:\.\d+)?(?:ms|s|sec|secs|second|seconds|m|min|mins|minute|minutes|"
    r"h|hr|hrs|hour|hours|d|day|days)$"
)
_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$|^\d{1,2}/\d{1,2}/\d{2,4}$")
_PURE_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")
_HAS_LETTER = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"\d")
_TOKEN_STRIP = re.compile(r"^[^a-z0-9%°]+|[^a-z0-9%°]+$")

_MUTABLE = "{param}"
_QUOTED = "{quoted}"

IdentityDecision = Literal["EXACT", "SAME_LINEAGE_REVISION", "DISTINCT", "AMBIGUOUS"]


def normalize_requirement_family(requirement_type: str) -> str:
    """Normalize a requirement type into a lineage-scoping family token.

    Different semantic families (Business Requirement, Acceptance Criteria,
    Security Requirement, …) never share a lineage, so a same-wording Acceptance
    Criterion can never become a revision of a Business Requirement.
    """
    return _WHITESPACE.sub(" ", (requirement_type or "").strip().lower()) or "unspecified"


def _mask_token(token: str) -> str | None:
    """Return the signature form of one token, or ``None`` to drop it.

    A mutable parameter collapses to ``{param}``; an entity discriminator (a token
    mixing letters and digits, e.g. ``sensor-1``) is preserved verbatim; a plain
    word is kept; empty/punctuation-only tokens are dropped.
    """
    core = _TOKEN_STRIP.sub("", token)
    if not core:
        return None
    if (
        _PURE_NUMBER.match(core)
        or _TEMPERATURE.match(core)
        or _PERCENT.match(core)
        or _DURATION.match(core)
        or _DATE.match(core)
    ):
        return _MUTABLE
    # A token that mixes letters and digits is an identity discriminator
    # (Sensor-1, RS485, reg3, apiv2) and must be preserved to keep entities apart.
    if _HAS_LETTER.search(core) and _HAS_DIGIT.search(core):
        return core
    return core


def requirement_lineage_signature(statement: str, requirement_type: str) -> str:
    """Deterministic lineage signature: family + discriminator-preserving statement.

    Preserves entity discriminators, masks mutable parameters. Never derived from
    the LLM ``requirement_key``.
    """
    family = normalize_requirement_family(requirement_type)
    text = _QUOTED_VALUE.sub(f" {_QUOTED} ", (statement or "").lower())
    tokens = [_mask_token(token) for token in text.split()]
    masked = " ".join(token for token in tokens if token)
    masked = _WHITESPACE.sub(" ", masked).strip()
    return f"{family}::{masked}" if masked else family


def structured_requirement_signature(
    *,
    statement: str,
    requirement_type: str,
    actor: str,
    modality: str,
    action: str,
    object_text: str,
    condition: str,
    polarity: str,
    requirement_family: str,
    entity_discriminators: list[str],
    mutable_parameters: list[str],
) -> str:
    """Build the stable lineage signature from validated semantic slots."""
    if not actor.strip() or not action.strip():
        return requirement_lineage_signature(statement, requirement_type)

    def normalize(value: str) -> str:
        """Normalize normalize deterministically within the active scope.

        Args:
            value (str): Value required by the operation's typed contract.

        Returns:
            str: The typed result produced by the operation.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
        normalized = _WHITESPACE.sub(" ", value.strip().lower())
        for parameter in sorted(mutable_parameters, key=len, reverse=True):
            if parameter.strip():
                normalized = normalized.replace(parameter.strip().lower(), _MUTABLE)
        return normalized

    family = normalize_requirement_family(requirement_family or requirement_type)
    discriminators = ",".join(
        sorted({normalize(value) for value in entity_discriminators if value.strip()})
    )
    slots = (
        family,
        normalize(polarity or "positive"),
        normalize(actor),
        normalize(modality),
        normalize(action),
        normalize(object_text),
        normalize(condition),
        discriminators,
    )
    return "::".join(slots)


@dataclass(frozen=True)
class RequirementIdentity:
    """Coordinate requirement identity behavior within the services boundary."""

    requirement_id: str
    revision_id: str
    signature: str
    normalized_statement: str


def resolve_requirement_identity(
    *,
    project: str,
    document_id: str,
    statement: str,
    requirement_type: str,
    normalized_statement: str,
) -> RequirementIdentity:
    """Compute the deterministic (lineage, revision) identity for one requirement.

    Lineage is stable across document versions (it does not include the version),
    so an unchanged obligation reuses both ids; a mutable-value change reuses the
    lineage but yields a new revision because the revision is keyed on the full
    normalized statement.
    """
    signature = requirement_lineage_signature(statement, requirement_type)
    lineage = requirement_lineage_id(project, document_id, signature)
    revision = requirement_revision_id(lineage, normalized_statement)
    return RequirementIdentity(
        requirement_id=lineage,
        revision_id=revision,
        signature=signature,
        normalized_statement=normalized_statement,
    )
