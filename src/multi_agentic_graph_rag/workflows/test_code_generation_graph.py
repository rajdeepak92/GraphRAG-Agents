"""Stage-4 test-code generation workflow spine (plan §12).

This module implements the *deterministic* part of the workflow that must be
reliable before any autonomous generation: scenario decomposition binding,
exact-data resolution, capability-gap analysis, context-manifest assembly, and
the readiness gate — exactly the ``--dry-run`` behavior of ``generate-test-code``
(plan §18). It performs no file edits.

The subsequent hash-guarded patch/validate/repair loop (§12.2, §20) is the
reasoning-model integration seam: it drives the ``PatchExecutor`` and
``ValidationRunner`` built in this change set. That loop is intentionally not
invoked here because it requires the live coding model; the deterministic
controls it depends on (worktree isolation, hash guards, validation, readiness)
are complete and tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CapabilityBinding,
    ContextManifest,
    ContextManifestItem,
    ReadinessStatus,
)
from multi_agentic_graph_rag.domain.schemas import StrictModel
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData
from multi_agentic_graph_rag.services.capability_mapper import (
    CapabilityCandidate,
    decide_capability,
)
from multi_agentic_graph_rag.services.context_assembler import RetrievedItem, assemble_manifest
from multi_agentic_graph_rag.services.readiness_gate import (
    ReadinessInputs,
    ReadinessReport,
    evaluate_readiness,
)
from multi_agentic_graph_rag.services.scenario_data_binder import (
    BindingResolution,
    resolve_binding,
)
from multi_agentic_graph_rag.services.test_data_bundle_builder import build_bundle


@dataclass(frozen=True)
class ScenarioAction:
    """One decomposed scenario action bound to a required capability (plan §10.5)."""

    action_text: str
    capability_id: str
    candidates: list[CapabilityCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioPlanRequest:
    """All deterministic inputs required to plan one scenario implementation."""

    scenario_id: str
    execution_profile_id: str
    codegen_run_id: str
    framework_snapshot: FrameworkSnapshot
    normalized: NormalizedTestData
    actions: list[ScenarioAction]
    scenario_approved: bool = True
    traceability_valid: bool = True
    decisions_approved: bool = True
    safety_satisfied: bool = True
    dependencies_available: bool = True
    modification_authorized: bool = False
    worktree_tree_hash: str = ""
    worktree_matches_index: bool = True
    token_budget: int = 24000
    sequence: int = 1


class ScenarioPlan(StrictModel):
    """Deterministic dry-run plan for one scenario (no files modified)."""

    scenario_id: str
    execution_profile_id: str
    binding_status: ReadinessStatus
    resolved_bundle_id: str | None = None
    resolved_bundle_checksum: str | None = None
    capability_bindings: list[CapabilityBinding] = Field(default_factory=list)
    unresolved_capabilities: list[str] = Field(default_factory=list)
    context_manifest_id: str | None = None
    readiness: ReadinessReport

    @property
    def is_ready(self) -> bool:
        return self.readiness.is_ready


def plan_scenario(request: ScenarioPlanRequest) -> ScenarioPlan:
    """Run the deterministic dry-run pipeline and return a scenario plan."""
    resolution = resolve_binding(
        scenario_id=request.scenario_id,
        execution_profile_id=request.execution_profile_id,
        bindings=request.normalized.bindings,
    )

    bundle_id: str | None = None
    bundle_checksum: str | None = None
    if resolution.is_ready and resolution.binding is not None:
        bundle = build_bundle(binding=resolution.binding, normalized=request.normalized)
        bundle_id = bundle.bundle_id
        bundle_checksum = bundle.checksum

    capability_bindings: list[CapabilityBinding] = []
    unresolved: list[str] = []
    for action in request.actions:
        decision = decide_capability(
            scenario_id=request.scenario_id,
            scenario_action=action.action_text,
            capability_id=action.capability_id,
            candidates=action.candidates,
            modification_authorized=request.modification_authorized,
        )
        if decision.binding is not None:
            capability_bindings.append(decision.binding)
        else:
            unresolved.append(action.capability_id)

    manifest = _build_manifest(request, capability_bindings)

    readiness = evaluate_readiness(
        scenario_id=request.scenario_id,
        execution_profile_id=request.execution_profile_id,
        inputs=ReadinessInputs(
            scenario_approved=request.scenario_approved,
            traceability_valid=request.traceability_valid,
            execution_profile_selected=bool(request.execution_profile_id),
            framework_snapshot_ready=request.framework_snapshot.status == "ready",
            test_data_snapshot_ready=True,
            schema_conforms=True,
            binding_status=resolution.status,
            bundle_available=bundle_id is not None,
            all_actions_mapped=not unresolved,
            all_inputs_resolved=bundle_id is not None,
            oracles_present=_bundle_has_oracles(resolution),
            setup_and_cleanup_defined=resolution.is_ready,
            decisions_approved=request.decisions_approved,
            framework_modification_required=any(
                binding.decision in ("implement", "modify_production")
                for binding in capability_bindings
            ),
            framework_modifications_authorized=request.modification_authorized,
            worktree_matches_index=request.worktree_matches_index,
            dependencies_available=request.dependencies_available,
            safety_satisfied=request.safety_satisfied,
        ),
    )

    return ScenarioPlan(
        scenario_id=request.scenario_id,
        execution_profile_id=request.execution_profile_id,
        binding_status=resolution.status,
        resolved_bundle_id=bundle_id,
        resolved_bundle_checksum=bundle_checksum,
        capability_bindings=capability_bindings,
        unresolved_capabilities=unresolved,
        context_manifest_id=manifest.manifest_id,
        readiness=readiness,
    )


def _bundle_has_oracles(resolution: BindingResolution) -> bool:
    return bool(resolution.binding and resolution.binding.oracle_ids)


def _build_manifest(
    request: ScenarioPlanRequest, capability_bindings: list[CapabilityBinding]
) -> ContextManifest:
    items: list[RetrievedItem] = [
        RetrievedItem(
            item=ContextManifestItem(
                source_type="scenario",
                source_id=request.scenario_id,
                revision=request.framework_snapshot.snapshot_id,
                content_hash="sha256:contract",
                retrieval_reason="immutable task contract",
                authoritative_source="postgresql",
            ),
            estimated_chars=800,
            tier=0,
        )
    ]
    for binding in capability_bindings:
        for symbol in binding.candidate_symbols:
            items.append(
                RetrievedItem(
                    item=ContextManifestItem(
                        source_type="code_symbol",
                        source_id=symbol,
                        revision=request.framework_snapshot.snapshot_id,
                        content_hash="sha256:pending",
                        retrieval_reason=f"candidate for {binding.capability_id}",
                        authoritative_source="repository",
                    ),
                    estimated_chars=1200,
                    tier=2,
                )
            )
    return assemble_manifest(
        codegen_run_id=request.codegen_run_id,
        scenario_id=request.scenario_id,
        sequence=request.sequence,
        framework_snapshot_id=request.framework_snapshot.snapshot_id,
        test_data_snapshot_id=request.normalized.snapshot_id,
        filesystem_snapshot_checksum=request.framework_snapshot.filesystem_checksum,
        candidates=items,
        token_budget=request.token_budget,
    )


__all__ = ["ScenarioAction", "ScenarioPlan", "ScenarioPlanRequest", "plan_scenario"]
