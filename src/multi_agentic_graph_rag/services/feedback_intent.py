"""Conservative guard for unsupported destructive feedback edits."""

from __future__ import annotations

import re

_DESTRUCTIVE_VERB_RE = re.compile(
    r"\b(delete|remove|drop|update|modify|rewrite|replace|edit|change)\b", re.I
)
_ARTIFACT_NOUN_RE = re.compile(r"\b(requirement|story|scenario|user story|test scenario)\b", re.I)
_ARTIFACT_ID_RE = re.compile(r"\b(?:REQ|US|SC)-[A-Z0-9][A-Z0-9-]*\b")


def classify_destructive_intent(comment: str) -> str | None:
    if not _DESTRUCTIVE_VERB_RE.search(comment):
        return None
    if _ARTIFACT_ID_RE.search(comment) or _ARTIFACT_NOUN_RE.search(comment):
        return "destructive feedback edits are not supported"
    return None
