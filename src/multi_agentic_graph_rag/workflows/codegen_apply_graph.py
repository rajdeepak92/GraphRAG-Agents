"""Stage-4 per-scenario apply loop as a checkpointed LangGraph (plan §12.2, §21).

Wires the deterministic controls into a resumable state machine: plan and gate
readiness, request human approval via a LangGraph ``interrupt`` before any
production/dependency change, apply hash-guarded patches produced through the
``PatchProducer`` seam, validate statically, repair within a bounded budget, and
persist the test case or a precise blocker. Side effects before the interrupt
are idempotent so an interrupted node resumes safely (§12.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.codegen_schemas import CodegenBlocker, TestCaseRecord
from multi_agentic_graph_rag.domain.identifiers import new_blocker_id, new_test_case_id
from multi_agentic_graph_rag.services.patch_executor import PatchError, PatchExecutor
from multi_agentic_graph_rag.services.patch_producer import PatchProducer, PatchRequest
from multi_agentic_graph_rag.services.scenario_data_binder import resolve_binding
from multi_agentic_graph_rag.services.test_data_bundle_builder import build_bundle
from multi_agentic_graph_rag.services.validation_runner import ValidationRunner
from multi_agentic_graph_rag.workflows.test_code_generation_graph import (
    ScenarioPlanRequest,
    plan_scenario,
)


class Stage4ScenarioState(TypedDict, total=False):
    """Checkpointed per-scenario apply state (identifiers, not large bodies)."""

    scenario_id: str
    codegen_run_id: str
    status: str
    binding_status: str
    resolved_bundle_id: str | None
    resolved_bundle_checksum: str | None
    context_manifest_id: str | None
    requires_approval: bool
    approved: bool | None
    generated_files: list[str]
    generated_file_hashes: dict[str, str]
    validation_status: str
    repair_attempt: int
    diagnostics: list[str]
    blocker: dict[str, Any] | None
    test_case_id: str | None


@dataclass
class CodegenRuntime:
    """Non-serializable dependencies for the apply loop (held outside state)."""

    settings: AppSettings
    plan_request: ScenarioPlanRequest
    patch_producer: PatchProducer
    patch_executor: PatchExecutor
    validation_runner: ValidationRunner
    codegen_store: CodegenPostgresStore
    project: str
    test_name: str
    test_file: str
    test_function: str
    priority: str = "Medium"
    require_approval: bool = False
    max_repair_attempts: int = 2
    generation_model: str = "unset"
    prompt_revision: str = "stage4-apply-v1"
    checkpointer: Any = None


def build_scenario_codegen_graph(runtime: CodegenRuntime) -> Any:
    """Build the compiled, checkpointed per-scenario apply graph."""
    graph: StateGraph[Stage4ScenarioState] = StateGraph(Stage4ScenarioState)
    graph.add_node("plan", lambda state: _plan_node(state, runtime))
    graph.add_node("approval", lambda state: _approval_node(state, runtime))
    graph.add_node("apply", lambda state: _apply_node(state, runtime))
    graph.add_node("validate", lambda state: _validate_node(state, runtime))
    graph.add_node("persist", lambda state: _persist_node(state, runtime))
    graph.add_node("record_blocker", lambda state: _blocker_node(state, runtime))

    graph.set_entry_point("plan")
    graph.add_conditional_edges(
        "plan", _route_after_plan, {"approval": "approval", "blocked": "record_blocker"}
    )
    graph.add_conditional_edges(
        "approval", _route_after_approval, {"apply": "apply", "blocked": "record_blocker"}
    )
    graph.add_edge("apply", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"persist": "persist", "repair": "apply", "blocked": "record_blocker"},
    )
    graph.add_edge("persist", END)
    graph.add_edge("record_blocker", END)
    return graph.compile(checkpointer=runtime.checkpointer)


# --- nodes -------------------------------------------------------------------


def _plan_node(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    plan = plan_scenario(runtime.plan_request)
    runtime.codegen_store.save_context_manifest(runtime.project, _manifest(runtime, plan))
    requires_approval = runtime.require_approval or any(
        binding.decision in ("implement", "modify_production")
        for binding in plan.capability_bindings
    )
    update: Stage4ScenarioState = {
        "scenario_id": plan.scenario_id,
        "binding_status": plan.binding_status,
        "resolved_bundle_id": plan.resolved_bundle_id,
        "resolved_bundle_checksum": plan.resolved_bundle_checksum,
        "context_manifest_id": plan.context_manifest_id,
        "requires_approval": requires_approval,
        "repair_attempt": 0,
        "status": "PLANNED" if plan.is_ready else "BLOCKED",
    }
    if not plan.is_ready:
        update["blocker"] = {
            "blocker_type": plan.readiness.status,
            "evidence": "; ".join(plan.readiness.evidence) or "readiness gate failed",
        }
    return update


def _approval_node(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    if not state.get("requires_approval"):
        return {"approved": True}
    decision = interrupt(
        {
            "codegen_run_id": state.get("codegen_run_id"),
            "scenario_id": state.get("scenario_id"),
            "change_type": "test_support_or_production_change",
            "reason": "capability decision requires human approval",
            "requested_by_node": "approval",
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    if not approved:
        return {
            "approved": False,
            "status": "BLOCKED",
            "blocker": {
                "blocker_type": "BLOCKED_AUTHORIZATION",
                "evidence": "human approval was rejected",
            },
        }
    return {"approved": True}


def _apply_node(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    request = _patch_request(state, runtime)
    generated: list[str] = list(state.get("generated_files", []))
    hashes: dict[str, str] = dict(state.get("generated_file_hashes", {}))
    try:
        for op in runtime.patch_producer.produce(request):
            if op.kind == "create":
                file_hash = runtime.patch_executor.create_file(
                    op.relative_path, op.content, expected_absent=op.relative_path not in hashes
                )
            else:
                if op.expected_hash is None:
                    raise PatchError(f"patch op for {op.relative_path} lacks an expected_hash")
                file_hash = runtime.patch_executor.apply_patch(
                    op.relative_path, op.content, expected_hash=op.expected_hash
                )
            if op.relative_path not in generated:
                generated.append(op.relative_path)
            hashes[op.relative_path] = file_hash
    except PatchError as exc:
        return {
            "status": "BLOCKED",
            "blocker": {
                "blocker_type": "BLOCKED_AUTHORIZATION",
                "evidence": f"patch rejected: {exc}",
            },
        }
    return {
        "generated_files": generated,
        "generated_file_hashes": hashes,
        "status": "GENERATED",
    }


def _validate_node(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    if state.get("status") == "BLOCKED":
        return {}
    paths = list(state.get("generated_files", []))
    parse = runtime.validation_runner.parse_python(paths)
    if not parse.ok:
        return _validation_failure(state, runtime, "FAILED_VALIDATION", list(parse.errors))
    collect = runtime.validation_runner.run_command("collect", paths)
    if collect.unavailable:
        return _validation_failure(state, runtime, "GENERATED", ["pytest unavailable"])
    if not collect.ok:
        return _validation_failure(state, runtime, "GENERATED", [collect.stdout, collect.stderr])
    return {"validation_status": "COLLECTABLE", "diagnostics": [], "status": "VALIDATED"}


def _validation_failure(
    state: Stage4ScenarioState,
    runtime: CodegenRuntime,
    status: str,
    diagnostics: list[str],
) -> Stage4ScenarioState:
    """Route to bounded repair, or block once the repair budget is exhausted."""
    attempt = int(state.get("repair_attempt", 0))
    cleaned = [line for line in diagnostics if line]
    if attempt < runtime.max_repair_attempts:
        return {
            "validation_status": status,
            "diagnostics": cleaned,
            "repair_attempt": attempt + 1,
            "status": "REPAIR",
        }
    return {
        "validation_status": status,
        "diagnostics": cleaned,
        "status": "BLOCKED",
        "blocker": {
            "blocker_type": "BLOCKED_MISSING_CAPABILITY",
            "evidence": "; ".join(cleaned) or "validation failed after repair budget",
        },
    }


def _persist_node(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    test_case = TestCaseRecord(
        test_case_id=new_test_case_id(),
        test_case_revision=1,
        scenario_id=str(state.get("scenario_id")),
        execution_profile_id=runtime.plan_request.execution_profile_id,
        test_name=runtime.test_name,
        test_file=runtime.test_file,
        test_function=runtime.test_function,
        scenario_data_binding_id=_binding_id(runtime),
        resolved_test_data_bundle_id=str(state.get("resolved_bundle_id")),
        resolved_test_data_bundle_checksum=str(state.get("resolved_bundle_checksum")),
        priority=runtime.priority,  # type: ignore[arg-type]
        validation_status=str(state.get("validation_status", "GENERATED")),  # type: ignore[arg-type]
        generated_file_hashes=dict(state.get("generated_file_hashes", {})),
        context_manifest_id=str(state.get("context_manifest_id")),
        generation_model=runtime.generation_model,
        prompt_revision=runtime.prompt_revision,
    )
    runtime.codegen_store.save_test_case(runtime.project, test_case)
    return {"status": "COMPLETED", "test_case_id": test_case.test_case_id}


def _blocker_node(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    detail = state.get("blocker") or {
        "blocker_type": "BLOCKED_MISSING_DATA",
        "evidence": "unspecified blocker",
    }
    blocker = CodegenBlocker(
        blocker_id=new_blocker_id(),
        scenario_id=str(state.get("scenario_id")),
        blocker_type=detail["blocker_type"],
        evidence=str(detail.get("evidence", "")),
    )
    runtime.codegen_store.save_blocker(runtime.project, blocker)
    return {"status": "BLOCKED"}


# --- routing -----------------------------------------------------------------


def _route_after_plan(state: Stage4ScenarioState) -> str:
    return "blocked" if state.get("status") == "BLOCKED" else "approval"


def _route_after_approval(state: Stage4ScenarioState) -> str:
    return "blocked" if state.get("status") == "BLOCKED" else "apply"


def _route_after_validate(state: Stage4ScenarioState) -> str:
    status = state.get("status")
    if status == "VALIDATED":
        return "persist"
    if status == "REPAIR":
        return "repair"
    return "blocked"


# --- helpers -----------------------------------------------------------------


def _manifest(runtime: CodegenRuntime, plan: Any) -> Any:
    from multi_agentic_graph_rag.services.context_assembler import assemble_manifest

    return assemble_manifest(
        codegen_run_id=runtime.plan_request.codegen_run_id,
        scenario_id=plan.scenario_id,
        sequence=runtime.plan_request.sequence,
        framework_snapshot_id=runtime.plan_request.framework_snapshot.snapshot_id,
        test_data_snapshot_id=runtime.plan_request.normalized.snapshot_id,
        worktree_tree_hash=runtime.plan_request.worktree_tree_hash
        or runtime.plan_request.framework_snapshot.tree_hash,
        candidates=[],
        token_budget=runtime.plan_request.token_budget,
    )


def _patch_request(state: Stage4ScenarioState, runtime: CodegenRuntime) -> PatchRequest:
    resolution = resolve_binding(
        scenario_id=runtime.plan_request.scenario_id,
        execution_profile_id=runtime.plan_request.execution_profile_id,
        bindings=runtime.plan_request.normalized.bindings,
    )
    redacted: dict[str, Any] = {}
    if resolution.binding is not None:
        bundle = build_bundle(
            binding=resolution.binding, normalized=runtime.plan_request.normalized
        )
        redacted = bundle.model_dump(mode="json")
        redacted.pop("secret_refs", None)  # never send secret references to the model
    return PatchRequest(
        scenario_id=runtime.plan_request.scenario_id,
        execution_profile_id=runtime.plan_request.execution_profile_id,
        resolved_bundle_id=state.get("resolved_bundle_id"),
        context_manifest_id=state.get("context_manifest_id"),
        redacted_bundle=redacted,
        repair_diagnostics=tuple(state.get("diagnostics", [])),
    )


def _binding_id(runtime: CodegenRuntime) -> str:
    resolution = resolve_binding(
        scenario_id=runtime.plan_request.scenario_id,
        execution_profile_id=runtime.plan_request.execution_profile_id,
        bindings=runtime.plan_request.normalized.bindings,
    )
    return resolution.binding.binding_id if resolution.binding else ""


__all__ = ["CodegenRuntime", "Stage4ScenarioState", "build_scenario_codegen_graph"]
