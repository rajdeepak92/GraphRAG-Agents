"""Resolve human feedback comments to known generated-artifact anchors."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from re import Pattern

REQUIREMENT_ID_RE = re.compile(r"\bREQ-[A-Z0-9][A-Z0-9-]*\b")
STORY_ID_RE = re.compile(r"\bUS-[A-Z0-9][A-Z0-9-]*\b")


@dataclass(frozen=True)
class AnchorResolution:
    anchor_id: str | None
    method: str | None
    candidates: list[tuple[str, int]] = field(default_factory=list)
    decline_reason: str | None = None


def resolve_anchor(
    *,
    explicit_id: str | None,
    comment: str,
    id_pattern: Pattern[str],
    known_titles: dict[str, str],
    retrieval_candidates: list[tuple[str, int]],
) -> AnchorResolution:
    if explicit_id:
        if id_pattern.fullmatch(explicit_id) and explicit_id in known_titles:
            return AnchorResolution(anchor_id=explicit_id, method="flag")
        return AnchorResolution(
            anchor_id=None,
            method=None,
            decline_reason=f"unknown or invalid anchor id: {explicit_id}",
        )

    for candidate_id in id_pattern.findall(comment):
        if candidate_id in known_titles:
            return AnchorResolution(anchor_id=candidate_id, method="comment_id")

    if not retrieval_candidates:
        return AnchorResolution(
            anchor_id=None,
            method=None,
            decline_reason="could not ground feedback to a known artifact",
        )

    top_score = retrieval_candidates[0][1]
    top_candidates = [candidate for candidate in retrieval_candidates if candidate[1] == top_score]
    if len(top_candidates) == 1:
        return AnchorResolution(anchor_id=top_candidates[0][0], method="retrieval")

    candidate_ids = ", ".join(candidate_id for candidate_id, _ in top_candidates)
    return AnchorResolution(
        anchor_id=None,
        method=None,
        candidates=top_candidates,
        decline_reason=f"ambiguous feedback anchor candidates: {candidate_ids}",
    )
