"""Domain-agnostic normalization for knowledge-graph entities and predicates.

The graph schema stays stable across projects because the predicate is data:
free-form verb phrases are normalized into bounded UPPER_SNAKE tokens instead of
requiring a per-domain relationship vocabulary. Optional domain profiles can
layer alias hints on top later without changing this core.
"""

from __future__ import annotations

import re

MAX_PREDICATE_LENGTH = 64

_NON_ALNUM_RUN = re.compile(r"[^A-Za-z0-9]+")
_WORD = re.compile(r"[A-Za-z0-9]+")


def normalize_entity_name(text: str) -> str:
    """Lowercase, separator-free form used for entity identity and matching."""
    text = text.strip().lower().replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def normalize_entity_type(text: str) -> str:
    """Lowercase snake_case category; defaults to ``concept`` when unusable."""
    token = _NON_ALNUM_RUN.sub("_", text.strip().lower()).strip("_")
    return token or "concept"


def normalize_predicate(text: str) -> str:
    """Normalize a free-form verb phrase into a bounded UPPER_SNAKE predicate.

    Returns an empty string when nothing usable remains; callers treat that as
    a validation failure for the extracted assertion.
    """
    token = _NON_ALNUM_RUN.sub("_", text.strip().upper()).strip("_")
    return token[:MAX_PREDICATE_LENGTH].rstrip("_")


def normalize_literal(text: str) -> str:
    """Whitespace-collapsed lowercase literal form used for assertion identity."""
    return " ".join(text.strip().lower().split())


def normalize_condition(text: str) -> str:
    """Whitespace-collapsed lowercase condition form used for assertion identity."""
    return " ".join(text.strip().lower().split())


def acronym_of(name: str) -> str:
    """Initialism of a multi-word name (``operating data`` -> ``od``); empty otherwise."""
    words = _WORD.findall(name)
    if len(words) < 2:
        return ""
    return "".join(word[0] for word in words).lower()
