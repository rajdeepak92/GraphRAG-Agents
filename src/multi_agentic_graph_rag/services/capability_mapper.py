"""Deterministic capability-gap analysis for scenario actions (plan §11).

Each scenario action becomes a deterministic reuse/compose/wrap/implement/block
decision. The recommended order is reuse > compose > wrap > implement, and
production-code changes or new dependencies are never decided autonomously —
they surface as blockers requiring human approval.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multi_agentic_graph_rag.domain.codegen_schemas import CapabilityBinding, ReadinessStatus
from multi_agentic_graph_rag.domain.identifiers import make_capability_binding_id

# How well a candidate framework symbol matches the required capability contract.
MatchKind = ["exact", "compatible", "partial"]


@dataclass(frozen=True)
class CapabilityCandidate:
    """One framework symbol proposed for a capability, with its match quality."""

    symbol_id: str
    fqn: str
    match_kind: str  # one of MatchKind

    def __post_init__(self) -> None:
        if self.match_kind not in MatchKind:
            raise ValueError(f"invalid match_kind: {self.match_kind}")


@dataclass(frozen=True)
class CapabilityDecisionResult:
    """A capability binding plus any readiness blocker it produced."""

    binding: CapabilityBinding | None
    blocker: ReadinessStatus | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_ready(self) -> bool:
        return self.binding is not None and self.blocker is None


def decide_capability(
    *,
    scenario_id: str,
    scenario_action: str,
    capability_id: str,
    candidates: list[CapabilityCandidate],
    modification_authorized: bool,
    adapter_path: str | None = None,
) -> CapabilityDecisionResult:
    """Apply the deterministic capability decision table (plan §11)."""
    binding_id = make_capability_binding_id(scenario_id=scenario_id, capability_id=capability_id)
    candidate_fqns = [candidate.fqn for candidate in candidates]

    exact = [candidate for candidate in candidates if candidate.match_kind == "exact"]
    compatible = [candidate for candidate in candidates if candidate.match_kind == "compatible"]
    partial = [candidate for candidate in candidates if candidate.match_kind == "partial"]

    if exact:
        return _bind(
            binding_id,
            scenario_id,
            scenario_action,
            capability_id,
            candidate_fqns,
            decision="reuse",
            selected=exact[0].fqn,
            reason=f"exact match {exact[0].fqn} satisfies the capability contract",
        )
    if len(partial) >= 2:
        return _bind(
            binding_id,
            scenario_id,
            scenario_action,
            capability_id,
            candidate_fqns,
            decision="compose",
            selected=partial[0].fqn,
            reason="multiple partial symbols compose into the required behavior",
        )
    if compatible:
        return _bind(
            binding_id,
            scenario_id,
            scenario_action,
            capability_id,
            candidate_fqns,
            decision="wrap",
            selected=compatible[0].fqn,
            reason="compatible symbol requires an adapter for the project contract",
            adapter_path=adapter_path,
        )
    if modification_authorized:
        return _bind(
            binding_id,
            scenario_id,
            scenario_action,
            capability_id,
            candidate_fqns,
            decision="implement",
            selected=None,
            reason="no reusable symbol; implementing an authorized test-support helper",
            adapter_path=adapter_path,
        )
    return CapabilityDecisionResult(
        binding=None,
        blocker="BLOCKED_MISSING_CAPABILITY",
        reasons=(
            f"capability '{capability_id}' has no reusable symbol and modifications "
            "are not authorized",
        ),
    )


def _bind(
    binding_id: str,
    scenario_id: str,
    scenario_action: str,
    capability_id: str,
    candidate_fqns: list[str],
    *,
    decision: str,
    selected: str | None,
    reason: str,
    adapter_path: str | None = None,
) -> CapabilityDecisionResult:
    binding = CapabilityBinding(
        binding_id=binding_id,
        scenario_id=scenario_id,
        scenario_action=scenario_action,
        capability_id=capability_id,
        candidate_symbols=candidate_fqns,
        decision=decision,  # type: ignore[arg-type]
        selected_symbol=selected,
        adapter_to_create=adapter_path if decision in ("wrap", "implement") else None,
        reason=reason,
    )
    return CapabilityDecisionResult(binding=binding)


__all__ = ["CapabilityCandidate", "CapabilityDecisionResult", "MatchKind", "decide_capability"]
