"""Deterministic Stage-4 readiness gate (plan §9).

A scenario is READY only when every gate passes. The gate never converts a
blocker into an assumption; it returns the first blocking status with evidence
so the workflow can persist a precise blocker and exit.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field

from multi_agentic_graph_rag.domain.codegen_schemas import ReadinessStatus
from multi_agentic_graph_rag.domain.schemas import StrictModel


@dataclass(frozen=True)
class ReadinessInputs:
    """Computable readiness signals for one scenario/profile pair (plan §9)."""

    scenario_approved: bool
    traceability_valid: bool
    execution_profile_selected: bool
    framework_snapshot_ready: bool
    test_data_snapshot_ready: bool
    schema_conforms: bool
    binding_status: ReadinessStatus
    bundle_available: bool
    all_actions_mapped: bool
    all_inputs_resolved: bool
    oracles_present: bool
    setup_and_cleanup_defined: bool
    decisions_approved: bool
    framework_modification_required: bool
    framework_modifications_authorized: bool
    worktree_matches_index: bool
    dependencies_available: bool
    safety_satisfied: bool


class ReadinessReport(StrictModel):
    """Per-gate result and the overall readiness verdict."""

    scenario_id: str
    execution_profile_id: str
    status: ReadinessStatus
    failed_checks: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        return self.status == "READY"


@dataclass(frozen=True)
class _Gate:
    ok: bool
    status: ReadinessStatus
    name: str
    evidence: str


def evaluate_readiness(
    *,
    scenario_id: str,
    execution_profile_id: str,
    inputs: ReadinessInputs,
) -> ReadinessReport:
    """Evaluate ordered readiness gates and return the first blocker or READY."""
    if inputs.binding_status not in ("READY", "NOT_APPLICABLE"):
        return ReadinessReport(
            scenario_id=scenario_id,
            execution_profile_id=execution_profile_id,
            status=inputs.binding_status,
            failed_checks=["scenario_data_binding"],
            evidence=[f"binding resolution returned {inputs.binding_status}"],
        )

    ordered: list[_Gate] = [
        _Gate(
            inputs.framework_snapshot_ready,
            "BLOCKED_FRAMEWORK_STALE",
            "framework_snapshot_ready",
            "framework snapshot is not READY",
        ),
        _Gate(
            inputs.test_data_snapshot_ready,
            "BLOCKED_INVALID_TEST_DATA_DOCUMENT",
            "test_data_snapshot_ready",
            "test-data snapshot is not READY",
        ),
        _Gate(
            inputs.schema_conforms,
            "BLOCKED_INVALID_TEST_DATA_DOCUMENT",
            "schema_conforms",
            "workbook does not conform to the pinned schema",
        ),
        _Gate(
            inputs.bundle_available,
            "BLOCKED_MISSING_DATA",
            "bundle_available",
            "no checksummed resolved bundle was produced",
        ),
        _Gate(
            inputs.all_inputs_resolved,
            "BLOCKED_MISSING_DATA",
            "all_inputs_resolved",
            "one or more exact inputs are unresolved",
        ),
        _Gate(
            inputs.oracles_present,
            "BLOCKED_MISSING_DATA",
            "oracles_present",
            "no expected result maps to an observable oracle",
        ),
        _Gate(
            inputs.setup_and_cleanup_defined,
            "BLOCKED_MISSING_DATA",
            "setup_and_cleanup_defined",
            "setup, readiness, or cleanup is undefined",
        ),
        _Gate(
            inputs.all_actions_mapped,
            "BLOCKED_MISSING_CAPABILITY",
            "all_actions_mapped",
            "a scenario action has no mapped capability",
        ),
        _Gate(
            inputs.decisions_approved,
            "BLOCKED_AMBIGUOUS_SPEC",
            "decisions_approved",
            "an applicable decision is unapproved",
        ),
        _Gate(
            not inputs.framework_modification_required or inputs.framework_modifications_authorized,
            "BLOCKED_AUTHORIZATION",
            "framework_modifications_authorized",
            "a required framework modification is not authorized",
        ),
        _Gate(
            inputs.dependencies_available,
            "BLOCKED_DEPENDENCY",
            "dependencies_available",
            "a required dependency is unavailable",
        ),
        _Gate(
            inputs.worktree_matches_index,
            "BLOCKED_FRAMEWORK_STALE",
            "worktree_matches_index",
            "worktree hashes do not match the indexed snapshot",
        ),
        _Gate(
            inputs.safety_satisfied,
            "BLOCKED_AUTHORIZATION",
            "safety_satisfied",
            "safety constraints are not satisfied",
        ),
        _Gate(
            inputs.scenario_approved,
            "BLOCKED_MISSING_DATA",
            "scenario_approved",
            "scenario is not approved / checksum-invalid",
        ),
        _Gate(
            inputs.traceability_valid,
            "BLOCKED_MISSING_DATA",
            "traceability_valid",
            "parent story/requirement links are invalid",
        ),
        _Gate(
            inputs.execution_profile_selected,
            "BLOCKED_MISSING_DATA",
            "execution_profile_selected",
            "no execution profile is selected",
        ),
    ]
    for gate in ordered:
        if not gate.ok:
            return ReadinessReport(
                scenario_id=scenario_id,
                execution_profile_id=execution_profile_id,
                status=gate.status,
                failed_checks=[gate.name],
                evidence=[gate.evidence],
            )
    return ReadinessReport(
        scenario_id=scenario_id,
        execution_profile_id=execution_profile_id,
        status="READY",
    )


__all__ = ["ReadinessInputs", "ReadinessReport", "evaluate_readiness"]
