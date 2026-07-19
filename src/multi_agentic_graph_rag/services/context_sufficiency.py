"""Sufficiency gate before planning/patching (plan §12.2).

Classifies whether the assembled context is sufficient to plan an
implementation, or whether the scenario is blocked on missing specification/data
or a missing framework capability. The gate never converts a blocker into an
assumption (plan §9).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SufficiencyStatus = Literal["SUFFICIENT", "SPEC_DATA_MISSING", "CAPABILITY_MISSING"]


@dataclass(frozen=True)
class SufficiencyInputs:
    """Deterministic signals gathered during retrieval."""

    bundle_resolved: bool
    all_actions_have_capability: bool
    unresolved_capability_ids: tuple[str, ...] = ()
    missing_data_reasons: tuple[str, ...] = ()
    worktree_matches_index: bool = True


@dataclass(frozen=True)
class SufficiencyResult:
    """Outcome of the sufficiency gate with structured reasons."""

    status: SufficiencyStatus
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_sufficient(self) -> bool:
        return self.status == "SUFFICIENT"


def assess_sufficiency(inputs: SufficiencyInputs) -> SufficiencyResult:
    """Return the deterministic sufficiency classification for a scenario."""
    reasons: list[str] = []
    if not inputs.worktree_matches_index:
        reasons.append("worktree hashes do not match the indexed snapshot; reindex required")
    if not inputs.bundle_resolved:
        reasons.append("resolved test-data bundle is not available")
    reasons.extend(inputs.missing_data_reasons)
    if reasons:
        return SufficiencyResult(status="SPEC_DATA_MISSING", reasons=tuple(reasons))

    if not inputs.all_actions_have_capability:
        capability_reasons = tuple(
            f"capability '{capability_id}' is unmapped"
            for capability_id in inputs.unresolved_capability_ids
        ) or ("one or more scenario actions have no mapped capability",)
        return SufficiencyResult(status="CAPABILITY_MISSING", reasons=capability_reasons)

    return SufficiencyResult(status="SUFFICIENT")


__all__ = [
    "SufficiencyInputs",
    "SufficiencyResult",
    "SufficiencyStatus",
    "assess_sufficiency",
]
