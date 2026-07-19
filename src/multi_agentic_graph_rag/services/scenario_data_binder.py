"""Deterministic scenario-data binding selection (plan §8.4).

The binder joins a declarative scenario + execution profile to its approved
data. Cardinality is strict: zero applicable bindings blocks with
``BLOCKED_MISSING_DATA``; more than one equally-applicable binding blocks with
``BLOCKED_AMBIGUOUS_BINDING``. A ``selection_priority`` tie-break is honoured so
an explicit choice is not treated as ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agentic_graph_rag.domain.codegen_schemas import ReadinessStatus
from multi_agentic_graph_rag.domain.test_data_schemas import ScenarioDataBinding


@dataclass(frozen=True)
class BindingResolution:
    """Outcome of resolving one scenario/profile pair to a single binding."""

    scenario_id: str
    execution_profile_id: str
    variant_id: str
    status: ReadinessStatus
    binding: ScenarioDataBinding | None = None
    candidates: tuple[str, ...] = ()

    @property
    def is_ready(self) -> bool:
        return self.status == "READY" and self.binding is not None


def resolve_binding(
    *,
    scenario_id: str,
    execution_profile_id: str,
    variant_id: str = "default",
    bindings: list[ScenarioDataBinding],
) -> BindingResolution:
    """Select exactly one applicable approved binding or return a precise blocker."""
    applicable = [
        binding
        for binding in bindings
        if binding.scenario_id == scenario_id
        and binding.execution_profile_id == execution_profile_id
        and binding.variant_id == variant_id
        and binding.approval_status == "APPROVED"
    ]
    if not applicable:
        return BindingResolution(
            scenario_id=scenario_id,
            execution_profile_id=execution_profile_id,
            variant_id=variant_id,
            status="BLOCKED_MISSING_DATA",
        )
    if len(applicable) == 1:
        return _ready(scenario_id, execution_profile_id, variant_id, applicable[0])

    # More than one candidate: an explicit highest priority resolves the choice.
    highest = max(binding.selection_priority for binding in applicable)
    top = [binding for binding in applicable if binding.selection_priority == highest]
    if len(top) == 1:
        return _ready(scenario_id, execution_profile_id, variant_id, top[0])
    return BindingResolution(
        scenario_id=scenario_id,
        execution_profile_id=execution_profile_id,
        variant_id=variant_id,
        status="BLOCKED_AMBIGUOUS_BINDING",
        candidates=tuple(sorted(binding.binding_id for binding in top)),
    )


def _ready(
    scenario_id: str,
    execution_profile_id: str,
    variant_id: str,
    binding: ScenarioDataBinding,
) -> BindingResolution:
    return BindingResolution(
        scenario_id=scenario_id,
        execution_profile_id=execution_profile_id,
        variant_id=variant_id,
        status="READY",
        binding=binding,
        candidates=(binding.binding_id,),
    )


__all__ = ["BindingResolution", "resolve_binding"]
