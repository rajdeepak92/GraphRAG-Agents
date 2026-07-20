"""Durable no-Git Stage-4 generation graph for one frozen scenario.

The graph deliberately keeps the reasoning model behind ``PatchProducer`` and
owns every deterministic boundary around it: permanent identity, path naming,
journaled writes, bounded repair, validation, framework-graph publication, and
immutable persistence.  A scenario finishes before the parent may select the
next one.
"""

from __future__ import annotations

import ast
import hashlib
import ipaddress
import re
import sys
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CodegenBlocker,
    GeneratedPatchBundle,
    LogicalTestCaseKey,
    ProviderFingerprint,
    Stage4Request,
    Stage4TestCaseRecord,
    TcStep,
    TestCaseImplementationPlan,
)
from multi_agentic_graph_rag.domain.errors import InputManifestChanged
from multi_agentic_graph_rag.domain.identifiers import (
    make_logical_key_hash,
    make_provider_fingerprint_hash,
    normalize_project,
    stable_token,
    tc_stem,
)
from multi_agentic_graph_rag.domain.schemas import CanonicalTestScenario, canonical_checksum
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData
from multi_agentic_graph_rag.services.codegen_context_retriever import CodegenContextRetriever
from multi_agentic_graph_rag.services.direct_file_transaction import (
    DirectFileTransaction,
    JournalStatus,
    classify_path,
)
from multi_agentic_graph_rag.services.framework_indexer import (
    FrameworkIndexResult,
    publish_framework_building,
    verify_and_activate_framework_snapshot,
)
from multi_agentic_graph_rag.services.patch_producer import PatchProducer, PatchRequest
from multi_agentic_graph_rag.services.validation_runner import (
    GeneratedCaseValidationRequest,
    GeneratedValidationContext,
    ValidationRunner,
)

PROMPT_REVISION = "stage4-codegen-v1"
GENERATION_POLICY_VERSION = "stage4-optimized-masterplan-v1"
ScenarioTerminalStatus = Literal["ACCEPTED", "BLOCKED", "REVISION_REQUIRED"]
PublishBuilding = Callable[[list[str], str], FrameworkIndexResult]
VerifySnapshot = Callable[[str, dict[str, str]], FrameworkSnapshot]


class Stage4ScenarioState(TypedDict, total=False):
    """Checkpoint-safe state; model/file bodies are JSON-serializable dictionaries."""

    scenario_id: str
    variant_id: str
    tc_id: int
    logical_key_hash: str
    input_fingerprint: str
    status: str
    context: dict[str, Any]
    context_manifest_id: str
    context_manifest: dict[str, Any]
    plan: dict[str, Any]
    bundle: dict[str, Any]
    generated_files: list[str]
    generated_file_hashes: dict[str, str]
    journal_status: str
    repair_attempt: int
    diagnostics: list[str]
    validation_evidence: dict[str, Any]
    building_snapshot_id: str
    framework_snapshot_id: str
    blocker: dict[str, Any]
    accepted_record: dict[str, Any]


@dataclass(frozen=True)
class ScenarioExecutionResult:
    """Small adapter result consumed structurally by the serial parent runner."""

    status: ScenarioTerminalStatus
    scenario_id: str
    variant_id: str
    tc_id: int
    accepted_snapshot: FrameworkSnapshot | None = None
    blocker: CodegenBlocker | None = None

    @property
    def framework_snapshot(self) -> FrameworkSnapshot | None:
        """Compatibility alias for callers predating the parent-run protocol."""
        return self.accepted_snapshot


@dataclass
class CodegenRuntime:
    """Non-serializable dependencies and immutable inputs for one scenario."""

    settings: AppSettings
    request: Stage4Request
    scenario: CanonicalTestScenario
    variant_id: str
    normalized_test_data: NormalizedTestData
    framework_snapshot: FrameworkSnapshot
    provider_fingerprint: ProviderFingerprint
    patch_producer: PatchProducer
    context_retriever: CodegenContextRetriever
    validation_runner: ValidationRunner
    codegen_store: CodegenPostgresStore
    code_graph_store: CodeGraphStore
    graphify_out_dir: Path | None = None
    repository_id: str | None = None
    checkpointer: Any = None
    max_repair_attempts: int | None = None
    publish_building: PublishBuilding | None = None
    verify_snapshot: VerifySnapshot | None = None

    @property
    def repair_limit(self) -> int:
        return self.max_repair_attempts or self.settings.stage4.max_repair_attempts


def build_scenario_codegen_graph(runtime: CodegenRuntime) -> Any:
    """Compile the required per-case LangGraph with no approval or Git nodes."""

    graph: StateGraph[Stage4ScenarioState] = StateGraph(Stage4ScenarioState)
    graph.add_node("reserve_tc_id", lambda state: _reserve_tc_id(state, runtime))
    graph.add_node("retrieve_context", lambda state: _retrieve_context(state, runtime))
    graph.add_node("generate_plan", lambda state: _generate_plan(state, runtime))
    graph.add_node("generate_bundle", lambda state: _generate_bundle(state, runtime))
    graph.add_node("open_file_transaction", lambda state: _open_file_transaction(state, runtime))
    graph.add_node("apply_bundle", lambda state: _apply_bundle(state, runtime))
    graph.add_node("validate", lambda state: _validate(state, runtime))
    graph.add_node("repair", lambda state: _repair(state, runtime))
    graph.add_node("rollback", lambda state: _rollback(state, runtime))
    graph.add_node("record_blocker", lambda state: _record_blocker(state, runtime))
    graph.add_node("publish_kg", lambda state: _publish_kg(state, runtime))
    graph.add_node("verify_kg", lambda state: _verify_kg(state, runtime))
    graph.add_node("persist_case", lambda state: _persist_case(state, runtime))
    graph.add_node("seal_transaction", lambda state: _seal_transaction(state, runtime))

    graph.set_entry_point("reserve_tc_id")
    graph.add_conditional_edges(
        "reserve_tc_id",
        _route_after_reservation,
        {
            "continue": "retrieve_context",
            "accepted": END,
            "revision": "record_blocker",
        },
    )
    graph.add_edge("retrieve_context", "generate_plan")
    graph.add_edge("generate_plan", "generate_bundle")
    graph.add_edge("generate_bundle", "open_file_transaction")
    graph.add_edge("open_file_transaction", "apply_bundle")
    graph.add_edge("apply_bundle", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validation,
        {"valid": "publish_kg", "repair": "repair", "exhausted": "rollback"},
    )
    graph.add_conditional_edges(
        "repair",
        _route_after_repair,
        {"apply": "apply_bundle", "repair": "repair", "exhausted": "rollback"},
    )
    graph.add_edge("rollback", "record_blocker")
    graph.add_edge("record_blocker", END)
    graph.add_edge("publish_kg", "verify_kg")
    graph.add_edge("verify_kg", "persist_case")
    graph.add_edge("persist_case", "seal_transaction")
    graph.add_edge("seal_transaction", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _reserve_tc_id(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    scenario_id = runtime.scenario.scenario_id
    variant_id = runtime.variant_id or "default"
    logical_hash = make_logical_key_hash(
        project_name=runtime.request.project_name,
        scenario_id=scenario_id,
        execution_profile_id=runtime.request.execution_profile_id,
        variant_id=variant_id,
    )
    key = LogicalTestCaseKey(
        project_name=runtime.request.project_name,
        scenario_id=scenario_id,
        execution_profile_id=runtime.request.execution_profile_id,
        variant_id=variant_id,
        logical_key_hash=logical_hash,
    )
    tc_id, _created = runtime.codegen_store.allocate_or_get_tc_id(key)
    input_fingerprint = _input_fingerprint(runtime)
    existing = runtime.codegen_store.get_stage4_test_case(tc_id)
    if existing is not None and existing.status == "ACCEPTED":
        if existing.input_fingerprint != input_fingerprint or not _accepted_files_match(
            runtime.request.framework_path,
            existing.generated_file_hashes,
            existing.validation_evidence,
        ):
            return {
                "scenario_id": scenario_id,
                "variant_id": variant_id,
                "tc_id": tc_id,
                "logical_key_hash": logical_hash,
                "input_fingerprint": input_fingerprint,
                "status": "REVISION_REQUIRED",
                "blocker": {
                    "blocker_type": "BLOCKED_DEPENDENCY",
                    "error_code": "REVISION_REQUIRED",
                    "evidence": (
                        "The accepted logical key has changed inputs or generated-file "
                        "checksums; Stage 4 will not overwrite it."
                    ),
                },
            }
        runtime.codegen_store.save_stage4_test_case(existing, run_id=runtime.request.run_id)
        transaction = _transaction(runtime, tc_id)
        if transaction.journal.exists:
            transaction.begin()
            if transaction.journal.status is JournalStatus.OPEN:
                # Persistence and KG projection already succeeded.  The only
                # remaining crash window is journal sealing.
                transaction.reconcile(externally_finalized=True)
                _persist_journal(runtime, transaction)
        _save_idempotency(
            runtime,
            tc_id=tc_id,
            node_name="reserve_tc_id",
            input_value={
                "logical_key_hash": logical_hash,
                "input": input_fingerprint,
                "observed_status": "ACCEPTED",
            },
            result={"status": "ACCEPTED", "tc_id": tc_id},
        )
        return {
            "scenario_id": scenario_id,
            "variant_id": variant_id,
            "tc_id": tc_id,
            "logical_key_hash": logical_hash,
            "input_fingerprint": input_fingerprint,
            "status": "ACCEPTED",
            "framework_snapshot_id": existing.framework_snapshot_id or "",
            "accepted_record": existing.model_dump(mode="json"),
        }

    reserved = Stage4TestCaseRecord(
        tc_id=tc_id,
        project_name=runtime.request.project_name,
        scenario_id=scenario_id,
        execution_profile_id=runtime.request.execution_profile_id,
        variant_id=variant_id,
        logical_key_hash=logical_hash,
        status="RESERVED",
        story_ids=list(runtime.scenario.story_ids),
        requirement_ids=list(runtime.scenario.requirement_ids),
        test_data_snapshot_id=runtime.normalized_test_data.snapshot_id,
        provider_fingerprint=runtime.provider_fingerprint,
        prompt_revision=runtime.provider_fingerprint.prompt_revision,
        input_fingerprint=input_fingerprint,
    )
    runtime.codegen_store.save_stage4_test_case(reserved, run_id=runtime.request.run_id)
    _save_idempotency(
        runtime,
        tc_id=tc_id,
        node_name="reserve_tc_id",
        input_value={
            "logical_key_hash": logical_hash,
            "input": input_fingerprint,
            "observed_status": existing.status if existing is not None else None,
        },
        result={"status": "RESERVED", "tc_id": tc_id},
    )
    return {
        "scenario_id": scenario_id,
        "variant_id": variant_id,
        "tc_id": tc_id,
        "logical_key_hash": logical_hash,
        "input_fingerprint": input_fingerprint,
        "status": "RESERVED",
        "repair_attempt": 0,
        "diagnostics": [],
    }


def _retrieve_context(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    context = runtime.context_retriever.scenario_codegen_context(
        project=runtime.request.project_name,
        run_id=runtime.request.run_id,
        scenario_id=runtime.scenario.scenario_id,
        execution_profile_id=runtime.request.execution_profile_id,
        variant_id=runtime.variant_id,
        normalized_test_data=runtime.normalized_test_data,
        framework_snapshot=runtime.framework_snapshot,
        token_budget=runtime.settings.stage4.context_token_budget,
        symbol_search_limit=runtime.settings.stage4.symbol_search_limit,
        max_context_symbols=runtime.settings.stage4.max_context_symbols,
    )
    payload = context.to_prompt_payload()
    context_checksum = canonical_checksum({"context": payload})
    context_id = "CTX-" + stable_token(
        runtime.request.run_id,
        runtime.scenario.scenario_id,
        runtime.variant_id,
        context_checksum,
        length=22,
    )
    manifest = _context_manifest_payload(context_id, payload, context_checksum)
    return {
        "context": payload,
        "context_manifest_id": context_id,
        "context_manifest": manifest,
        "status": "CONTEXT_READY",
    }


def _generate_plan(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    request = _patch_request(state, runtime)
    plan = runtime.patch_producer.plan(request)
    _validate_plan(plan, runtime, int(state["tc_id"]))
    return {"plan": plan.model_dump(mode="json"), "status": "PLANNED"}


def _generate_bundle(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    plan = TestCaseImplementationPlan.model_validate(state["plan"])
    bundle = runtime.patch_producer.generate_bundle(_patch_request(state, runtime), plan)
    _validate_bundle(plan, bundle, runtime, int(state["tc_id"]))
    return {"bundle": bundle.model_dump(mode="json"), "status": "BUNDLE_READY"}


def _open_file_transaction(
    state: Stage4ScenarioState, runtime: CodegenRuntime
) -> Stage4ScenarioState:
    transaction = _transaction(runtime, int(state["tc_id"]))
    transaction.begin()
    if transaction.journal.status is JournalStatus.SEALED:
        raise RuntimeError("sealed transaction cannot be used for an unaccepted regeneration")
    if transaction.journal.status is JournalStatus.OPEN and transaction.journal.entries:
        # A non-finalized open journal belongs to an interrupted attempt.  Its
        # exact preimages are restored before a fresh bounded generation attempt.
        transaction.reconcile(externally_finalized=False)
        _persist_journal(runtime, transaction)
    if transaction.journal.status is JournalStatus.ROLLED_BACK:
        transaction.restart()
    _persist_journal(runtime, transaction)
    _save_idempotency(
        runtime,
        tc_id=int(state["tc_id"]),
        node_name="open_file_transaction",
        input_value={"bundle": state["bundle"]},
        result={"status": transaction.journal.status.value},
    )
    return {"journal_status": transaction.journal.status.value, "status": "WRITING"}


def _apply_bundle(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    plan = TestCaseImplementationPlan.model_validate(state["plan"])
    bundle = GeneratedPatchBundle.model_validate(state["bundle"])
    _validate_bundle(plan, bundle, runtime, int(state["tc_id"]))
    transaction = _transaction(runtime, int(state["tc_id"]))
    transaction.begin()
    try:
        for generated_file in bundle.files:
            transaction.write(generated_file.relative_path, generated_file.content)
        transaction.verify_open()
        _persist_journal(runtime, transaction)
        hashes = transaction.journal.expected_file_hashes()
        _save_idempotency(
            runtime,
            tc_id=int(state["tc_id"]),
            node_name="apply_bundle",
            input_value={"bundle": state["bundle"]},
            result={"files": hashes},
        )
    except Exception:
        if transaction.journal.status is JournalStatus.OPEN:
            transaction.rollback()
            _persist_journal(runtime, transaction)
        raise
    return {
        "generated_files": list(hashes),
        "generated_file_hashes": hashes,
        "journal_status": transaction.journal.status.value,
        "status": "GENERATED",
    }


def _validate(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    plan = TestCaseImplementationPlan.model_validate(state["plan"])
    bundle = GeneratedPatchBundle.model_validate(state["bundle"])
    roles = {item.role: item.relative_path for item in bundle.files}
    support = tuple(
        item.relative_path for item in bundle.files if item.role not in {"test", "robot"}
    )
    transaction = _transaction(runtime, int(state["tc_id"]))
    transaction.begin()
    request = GeneratedCaseValidationRequest(
        python_file=roles["test"],
        robot_file=roles["robot"],
        support_files=support,
        expected_hashes=transaction.journal.expected_file_hashes(),
        shared_preimages=transaction.journal.validation_preimages(),
        created_files=transaction.journal.created_files(),
        traceability_context={
            "scenario_id": runtime.scenario.scenario_id,
            "story_ids": runtime.scenario.story_ids,
            "requirement_ids": runtime.scenario.requirement_ids,
        },
        exact_test_data_context=state.get("context", {}).get("test_data_records", []),
        expected_step_count=len(plan.steps),
        critical_step_count=sum(1 for step in plan.steps if step.critical),
    )
    result = runtime.validation_runner.validate_generated_case(
        request,
        traceability_hook=lambda context: _traceability_check(context, runtime),
        exact_test_data_hook=lambda context: _exact_data_check(context, plan, state),
    )
    if result.ok:
        evidence = _validation_evidence(
            result,
            state["generated_file_hashes"],
            framework_root=runtime.request.framework_path,
        )
        _save_idempotency(
            runtime,
            tc_id=int(state["tc_id"]),
            node_name="validate",
            input_value={"files": state["generated_file_hashes"]},
            result={"status": "VALIDATED"},
        )
        return {
            "status": "VALIDATED",
            "diagnostics": [],
            "validation_evidence": evidence,
        }
    attempt = int(state.get("repair_attempt", 0))
    status = "REPAIR" if attempt < runtime.repair_limit else "VALIDATION_EXHAUSTED"
    return {"status": status, "diagnostics": list(result.diagnostics)}


def _repair(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    plan = TestCaseImplementationPlan.model_validate(state["plan"])
    current = GeneratedPatchBundle.model_validate(state["bundle"])
    try:
        repaired = runtime.patch_producer.repair_bundle(
            _patch_request(state, runtime),
            plan,
            current,
            tuple(state.get("diagnostics", [])),
        )
    except Exception:
        _rollback_open(runtime, int(state["tc_id"]))
        raise
    attempt = int(state.get("repair_attempt", 0)) + 1
    try:
        _validate_bundle(plan, repaired, runtime, int(state["tc_id"]))
    except ValueError as exc:
        return {
            "repair_attempt": attempt,
            "diagnostics": [f"REPAIRED_BUNDLE_POLICY_INVALID: {exc}"],
            "status": (
                "REPAIR_INVALID" if attempt < runtime.repair_limit else "VALIDATION_EXHAUSTED"
            ),
        }
    return {
        "bundle": repaired.model_dump(mode="json"),
        "repair_attempt": attempt,
        "status": "REPAIR_READY",
    }


def _rollback(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    transaction = _transaction(runtime, int(state["tc_id"]))
    transaction.begin()
    transaction.rollback()
    _persist_journal(runtime, transaction)
    _save_idempotency(
        runtime,
        tc_id=int(state["tc_id"]),
        node_name="rollback",
        input_value={"diagnostics": state.get("diagnostics", [])},
        result={"status": "ROLLED_BACK"},
    )
    first_code = _diagnostic_code(state.get("diagnostics", []))
    return {
        "journal_status": JournalStatus.ROLLED_BACK.value,
        "status": "BLOCKED",
        "blocker": {
            "blocker_type": "BLOCKED_DEPENDENCY",
            "error_code": first_code or "VALIDATION_FAILED",
            "evidence": "; ".join(state.get("diagnostics", []))[:4000]
            or "validation failed after the bounded repair budget",
        },
    }


def _publish_kg(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    tc_id = int(state["tc_id"])
    changed_files = list(state["generated_file_hashes"])
    attempt_id = _kg_attempt_id(runtime, tc_id)
    result: FrameworkIndexResult | None = None
    try:
        if runtime.publish_building is not None:
            result = runtime.publish_building(
                changed_files, runtime.normalized_test_data.snapshot_id
            )
        else:
            result = publish_framework_building(
                settings=runtime.settings,
                framework_path=runtime.request.framework_path,
                graphify_out_dir=runtime.graphify_out_dir,
                repository_id=runtime.repository_id
                or normalize_project(runtime.request.project_name),
                test_data_snapshot_id=runtime.normalized_test_data.snapshot_id,
                changed_files=changed_files,
            )
        runtime.codegen_store.save_kg_publication_attempt(
            attempt_id=attempt_id,
            project_name=runtime.request.project_name,
            run_id=runtime.request.run_id,
            tc_id=tc_id,
            status="BUILDING",
            previous_snapshot_id=runtime.framework_snapshot.snapshot_id,
            proposed_snapshot_id=result.snapshot.snapshot_id,
            payload={"changed_files": changed_files},
        )
        _save_idempotency(
            runtime,
            tc_id=tc_id,
            node_name="publish_kg",
            input_value={"files": state["generated_file_hashes"]},
            result={"snapshot_id": result.snapshot.snapshot_id},
        )
    except Exception as exc:
        if result is not None:
            runtime.code_graph_store.mark_snapshot_failed(result.snapshot.snapshot_id)
        with suppress(Exception):
            runtime.codegen_store.save_kg_publication_attempt(
                attempt_id=attempt_id,
                project_name=runtime.request.project_name,
                run_id=runtime.request.run_id,
                tc_id=tc_id,
                status="FAILED",
                previous_snapshot_id=runtime.framework_snapshot.snapshot_id,
                proposed_snapshot_id=(result.snapshot.snapshot_id if result is not None else None),
                payload={"error_type": type(exc).__name__},
            )
        _rollback_open(runtime, tc_id)
        raise
    return {"building_snapshot_id": result.snapshot.snapshot_id, "status": "KG_BUILDING"}


def _verify_kg(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    tc_id = int(state["tc_id"])
    snapshot_id = state["building_snapshot_id"]
    attempt_id = _kg_attempt_id(runtime, tc_id)
    snapshot: FrameworkSnapshot | None = None
    try:
        if runtime.verify_snapshot is not None:
            snapshot = runtime.verify_snapshot(snapshot_id, state["generated_file_hashes"])
        else:
            snapshot = verify_and_activate_framework_snapshot(
                settings=runtime.settings,
                snapshot_id=snapshot_id,
                expected_file_hashes=state["generated_file_hashes"],
            )
        runtime.codegen_store.save_kg_publication_attempt(
            attempt_id=attempt_id,
            project_name=runtime.request.project_name,
            run_id=runtime.request.run_id,
            tc_id=tc_id,
            status="READY",
            previous_snapshot_id=runtime.framework_snapshot.snapshot_id,
            proposed_snapshot_id=snapshot.snapshot_id,
            payload={"verified_files": state["generated_file_hashes"]},
        )
        _save_idempotency(
            runtime,
            tc_id=tc_id,
            node_name="verify_kg",
            input_value={"snapshot_id": snapshot_id},
            result={"status": "READY", "snapshot_id": snapshot.snapshot_id},
        )
    except Exception as exc:
        if snapshot is not None and snapshot.status == "ready" and snapshot.active:
            # The KG already represents the still-present files.  Keep the
            # journal open so durable resume can finish coordination/persistence.
            raise
        runtime.code_graph_store.mark_snapshot_failed(snapshot_id)
        with suppress(Exception):
            runtime.codegen_store.save_kg_publication_attempt(
                attempt_id=attempt_id,
                project_name=runtime.request.project_name,
                run_id=runtime.request.run_id,
                tc_id=tc_id,
                status="FAILED",
                previous_snapshot_id=runtime.framework_snapshot.snapshot_id,
                proposed_snapshot_id=snapshot_id,
                payload={"error_type": type(exc).__name__},
            )
        _rollback_open(runtime, tc_id)
        raise
    return {"framework_snapshot_id": snapshot.snapshot_id, "status": "KG_READY"}


def _persist_case(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    plan = TestCaseImplementationPlan.model_validate(state["plan"])
    bundle = GeneratedPatchBundle.model_validate(state["bundle"])
    by_role = {item.role: item.relative_path for item in bundle.files}
    helper_files = [
        item.relative_path for item in bundle.files if item.relative_path.startswith("test_lib/")
    ]
    record = Stage4TestCaseRecord(
        tc_id=int(state["tc_id"]),
        project_name=runtime.request.project_name,
        scenario_id=runtime.scenario.scenario_id,
        execution_profile_id=runtime.request.execution_profile_id,
        variant_id=runtime.variant_id,
        logical_key_hash=state["logical_key_hash"],
        status="ACCEPTED",
        story_ids=list(runtime.scenario.story_ids),
        requirement_ids=list(runtime.scenario.requirement_ids),
        tc_steps=[
            TcStep(
                step=item.step,
                description=item.description,
                action_ref=item.capability_id,
                critical=item.critical,
            )
            for item in plan.steps
        ],
        module=plan.module,
        test_file=by_role["test"],
        robot_file=by_role["robot"],
        helper_files=helper_files,
        generated_file_hashes=dict(state["generated_file_hashes"]),
        framework_snapshot_id=state["framework_snapshot_id"],
        test_data_snapshot_id=runtime.normalized_test_data.snapshot_id,
        provider_fingerprint=runtime.provider_fingerprint,
        prompt_revision=runtime.provider_fingerprint.prompt_revision,
        input_fingerprint=state["input_fingerprint"],
        context_manifest=dict(state.get("context_manifest", {})),
        validation_evidence=dict(state.get("validation_evidence", {})),
    )
    # Projection precedes authoritative acceptance.  Therefore a durable
    # ACCEPTED row implies the idempotent Neo4j projection was already attempted.
    runtime.code_graph_store.publish_test_case(record)
    runtime.codegen_store.save_stage4_test_case(record, run_id=runtime.request.run_id)
    _save_idempotency(
        runtime,
        tc_id=record.tc_id,
        node_name="persist_case",
        input_value={"record": record.model_dump(mode="json")},
        result={"status": "ACCEPTED", "tc_id": record.tc_id},
    )
    return {
        "accepted_record": record.model_dump(mode="json"),
        "status": "ACCEPTED",
    }


def _seal_transaction(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    transaction = _transaction(runtime, int(state["tc_id"]))
    transaction.begin()
    transaction.seal()
    _persist_journal(runtime, transaction)
    _save_idempotency(
        runtime,
        tc_id=int(state["tc_id"]),
        node_name="seal_transaction",
        input_value={"record": state["accepted_record"]},
        result={"status": "SEALED"},
    )
    return {"journal_status": JournalStatus.SEALED.value, "status": "ACCEPTED"}


def _record_blocker(state: Stage4ScenarioState, runtime: CodegenRuntime) -> Stage4ScenarioState:
    detail = state.get("blocker", {})
    error_code = detail.get("error_code")
    tc_id = state.get("tc_id")
    blocker = CodegenBlocker(
        blocker_id="BLK-"
        + stable_token(
            runtime.request.run_id,
            runtime.scenario.scenario_id,
            runtime.variant_id,
            tc_id or "unreserved",
            error_code or "BLOCKED",
            length=24,
        ),
        scenario_id=runtime.scenario.scenario_id,
        blocker_type=detail.get("blocker_type", "BLOCKED_DEPENDENCY"),
        evidence=detail.get("evidence", "Stage-4 generation was blocked"),
        run_id=runtime.request.run_id,
        tc_id=tc_id,
        logical_key_hash=state.get("logical_key_hash"),
        error_code=error_code,
    )
    runtime.codegen_store.save_blocker(
        runtime.request.project_name,
        blocker,
        run_id=runtime.request.run_id,
        tc_id=tc_id,
    )
    if state.get("status") != "REVISION_REQUIRED" and state.get("tc_id") is not None:
        existing = runtime.codegen_store.get_stage4_test_case(int(state["tc_id"]))
        if existing is not None and existing.status != "ACCEPTED":
            runtime.codegen_store.save_stage4_test_case(
                existing.model_copy(update={"status": "BLOCKED"}),
                run_id=runtime.request.run_id,
            )
    if tc_id is not None:
        _save_idempotency(
            runtime,
            tc_id=int(tc_id),
            node_name="record_blocker",
            input_value={"blocker": blocker.model_dump(mode="json")},
            result={"status": state.get("status", "BLOCKED"), "blocker_id": blocker.blocker_id},
        )
    return {"status": state.get("status", "BLOCKED"), "blocker": blocker.model_dump(mode="json")}


def _route_after_reservation(state: Stage4ScenarioState) -> str:
    if state.get("status") == "ACCEPTED":
        return "accepted"
    if state.get("status") == "REVISION_REQUIRED":
        return "revision"
    return "continue"


def _route_after_validation(state: Stage4ScenarioState) -> str:
    if state.get("status") == "VALIDATED":
        return "valid"
    if state.get("status") == "REPAIR":
        return "repair"
    return "exhausted"


def _route_after_repair(state: Stage4ScenarioState) -> str:
    if state.get("status") == "REPAIR_READY":
        return "apply"
    if state.get("status") == "REPAIR_INVALID":
        return "repair"
    return "exhausted"


def _patch_request(state: Stage4ScenarioState, runtime: CodegenRuntime) -> PatchRequest:
    context = state.get("context", {})
    return PatchRequest(
        scenario_id=runtime.scenario.scenario_id,
        execution_profile_id=runtime.request.execution_profile_id,
        variant_id=runtime.variant_id,
        tc_id=state.get("tc_id"),
        resolved_bundle_id=context.get("resolved_binding", {}).get("binding_id"),
        context_manifest_id=state.get("context_manifest_id"),
        redacted_bundle={
            "binding": context.get("resolved_binding", {}),
            "records": context.get("test_data_records", []),
            "secret_refs": context.get("secret_refs", []),
        },
        context=context,
        repair_diagnostics=tuple(state.get("diagnostics", [])),
    )


def _validate_plan(plan: TestCaseImplementationPlan, runtime: CodegenRuntime, tc_id: int) -> None:
    if plan.scenario_id != runtime.scenario.scenario_id:
        raise ValueError("implementation plan scenario_id mismatch")
    if re.fullmatch(r"[a-z][a-z0-9_]*", plan.module) is None:
        raise ValueError("implementation plan module must be lowercase and module-specific")
    if [item.step for item in plan.steps] != list(range(1, len(plan.steps) + 1)):
        raise ValueError("implementation plan steps must be contiguous from 1")
    stem = tc_stem(tc_id, plan.test_title)
    expected_test = f"tests/{plan.module}/{stem}.py"
    expected_robot = f"tests_robot/{plan.module}/{stem}.robot"
    planned_roles: dict[str, list[str]] = {}
    for planned in plan.files:
        classified = classify_path(planned.relative_path)
        if classified.module != plan.module:
            raise ValueError("planned file module differs from implementation module")
        planned_roles.setdefault(planned.role, []).append(planned.relative_path)
    if planned_roles.get("test") != [expected_test]:
        raise ValueError(f"plan must contain exactly {expected_test}")
    if planned_roles.get("robot") != [expected_robot]:
        raise ValueError(f"plan must contain exactly {expected_robot}")


def _validate_bundle(
    plan: TestCaseImplementationPlan,
    bundle: GeneratedPatchBundle,
    runtime: CodegenRuntime,
    tc_id: int,
) -> None:
    _validate_plan(plan, runtime, tc_id)
    if bundle.scenario_id != runtime.scenario.scenario_id or bundle.tc_id != tc_id:
        raise ValueError("generated bundle identity mismatch")
    if bundle.plan_checksum != canonical_checksum(plan):
        raise ValueError("generated bundle plan checksum mismatch")
    planned = {(item.role, item.relative_path) for item in plan.files if not item.reuse_existing}
    generated = {(item.role, item.relative_path) for item in bundle.files}
    if generated != planned or len(generated) != len(bundle.files):
        raise ValueError("generated bundle must exactly match non-reused planned files")
    roles = [item.role for item in bundle.files]
    if roles.count("test") != 1 or roles.count("robot") != 1:
        raise ValueError("generated bundle requires exactly one Python test and one Robot file")


def _input_fingerprint(runtime: CodegenRuntime) -> str:
    return canonical_checksum(
        {
            "scenario": runtime.scenario.model_dump(mode="json"),
            "execution_profile_id": runtime.request.execution_profile_id,
            "variant_id": runtime.variant_id,
            "test_data_checksum": runtime.normalized_test_data.checksum,
            "provider_fingerprint": runtime.provider_fingerprint.model_dump(mode="json"),
            "generation_policy_version": GENERATION_POLICY_VERSION,
        }
    )


def _accepted_files_match(
    root: Path,
    hashes: Mapping[str, str],
    validation_evidence: Mapping[str, Any] | None = None,
) -> bool:
    framework_root = root.resolve()
    shared_evidence = (validation_evidence or {}).get("shared_ast_fingerprints", {})
    for relative_path, expected in hashes.items():
        target = (framework_root / relative_path).resolve()
        try:
            target.relative_to(framework_root)
        except ValueError:
            return False
        if not target.is_file():
            return False
        actual = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            try:
                role = classify_path(relative_path).role
            except Exception:
                return False
            if role not in {"wrapper", "helper"}:
                return False
            recorded = (
                shared_evidence.get(relative_path) if isinstance(shared_evidence, Mapping) else None
            )
            if not isinstance(recorded, list):
                return False
            try:
                current = _module_statement_fingerprints(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, SyntaxError):
                return False
            if not set(str(value) for value in recorded).issubset(current):
                return False
    return bool(hashes)


def _module_statement_fingerprints(source: str) -> set[str]:
    """Fingerprint complete top-level AST statements for additive shared-file replay."""
    tree = ast.parse(source)
    return {
        canonical_checksum({"ast": ast.dump(node, include_attributes=False)}) for node in tree.body
    }


def _transaction(runtime: CodegenRuntime, tc_id: int) -> DirectFileTransaction:
    return DirectFileTransaction(
        framework_root=runtime.request.framework_path,
        project=normalize_project(runtime.request.project_name),
        run_id=runtime.request.run_id,
        tc_id=tc_id,
        journal_root=runtime.settings.paths.generated_dir,
    )


def _persist_journal(runtime: CodegenRuntime, transaction: DirectFileTransaction) -> None:
    runtime.codegen_store.save_file_journal_metadata(
        project_name=runtime.request.project_name,
        run_id=runtime.request.run_id,
        tc_id=transaction.tc_id,
        status=transaction.journal.status.value,
        journal_path=str(transaction.journal.journal_dir),
        payload=transaction.journal.metadata(),
    )


def _rollback_open(runtime: CodegenRuntime, tc_id: int) -> None:
    transaction = _transaction(runtime, tc_id)
    transaction.begin()
    if transaction.journal.status is JournalStatus.OPEN:
        transaction.rollback()
        _persist_journal(runtime, transaction)


def _save_idempotency(
    runtime: CodegenRuntime,
    *,
    tc_id: int,
    node_name: str,
    input_value: Any,
    result: dict[str, Any],
) -> None:
    runtime.codegen_store.save_idempotency_record(
        project_name=runtime.request.project_name,
        run_id=runtime.request.run_id,
        tc_id=tc_id,
        node_name=node_name,
        input_checksum=canonical_checksum({"input": input_value}),
        result_payload=result,
    )


def _kg_attempt_id(runtime: CodegenRuntime, tc_id: int) -> str:
    return "KGP-" + stable_token(runtime.request.run_id, tc_id, length=24)


def _traceability_check(
    _context: GeneratedValidationContext, runtime: CodegenRuntime
) -> Iterable[str]:
    issues: list[str] = []
    if not runtime.scenario.story_ids:
        issues.append("scenario has no parent story IDs")
    if not runtime.scenario.requirement_ids:
        issues.append("scenario has no parent requirement IDs")
    return issues


def _exact_data_check(
    context: GeneratedValidationContext,
    plan: TestCaseImplementationPlan,
    state: Stage4ScenarioState,
) -> Iterable[str]:
    records = state.get("context", {}).get("test_data_records", [])
    approved_values = tuple(_scalar_values(records))
    approved_secret_refs = {
        value
        for value in approved_values
        if isinstance(value, str) and value.startswith("secret://")
    }
    issues: list[str] = []
    for value in _scalar_values(plan.test_variables):
        if not _approved_exact(value, approved_values) and value not in {None, True, False}:
            issues.append(
                f"planned test variable is not present in the resolved data bundle: {value!r}"
            )
    for relative_path, source in context.sources.items():
        for secret_ref in set(re.findall(r"secret://[A-Za-z0-9_./:-]+", source)):
            if secret_ref not in approved_secret_refs:
                issues.append(f"unapproved secret reference in {relative_path}: {secret_ref}")
        for candidate in _ip_literals(source):
            if not _approved_exact(candidate, approved_values):
                issues.append(f"invented IP address in {relative_path}: {candidate}")
        if relative_path.endswith(".py"):
            issues.extend(
                f"invented exact test-data value in {relative_path}: {name}={value!r}"
                for name, value in _invented_exact_values(source, approved_values)
            )
    return issues


def _scalar_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _scalar_values(item)
    elif isinstance(value, list | tuple | set):
        for item in value:
            yield from _scalar_values(item)
    elif isinstance(value, str | int | float | bool) or value is None:
        yield value


def _ip_literals(source: str) -> set[str]:
    candidates = set(re.findall(r"(?<![A-Za-z0-9])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9])", source))
    valid: set[str] = set()
    for candidate in candidates:
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        valid.add(candidate)
    return valid


_EXACT_DATA_NAMES = (
    "address",
    "credential",
    "delay",
    "device_id",
    "duration",
    "host",
    "interval",
    "ip",
    "password",
    "port",
    "threshold",
    "timeout",
    "token",
    "username",
)

_STRUCTURAL_LITERAL_CALLS = frozenset(
    {
        "debug",
        "error",
        "exception",
        "info",
        "warning",
        "get",
        "getattr",
        "hasattr",
        "setattr",
        "pop",
        "setdefault",
        "startswith",
        "endswith",
        "replace",
        "split",
        "join",
        "format",
    }
)


def _invented_exact_values(
    source: str, approved_values: Iterable[Any]
) -> set[tuple[str, str | int | float]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[tuple[str, str | int | float]] = set()

    def check(name: str, value_node: ast.AST | None) -> None:
        normalized = name.casefold()
        if not any(token in normalized for token in _EXACT_DATA_NAMES):
            return
        if not isinstance(value_node, ast.Constant) or not isinstance(
            value_node.value, str | int | float
        ):
            return
        if not _approved_exact(value_node.value, approved_values):
            found.add((name, value_node.value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                check(_assignment_name(target), node.value)
        elif isinstance(node, ast.AnnAssign):
            check(_assignment_name(node.target), node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    check(key.value, value)
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg:
                    check(keyword.arg, keyword.value)
            function_name = _call_leaf_name(node.func)
            if function_name in {"sleep", "wait"} and node.args:
                check("timing_duration", node.args[0])
            if function_name.casefold() not in _STRUCTURAL_LITERAL_CALLS:
                for index, argument in enumerate(node.args):
                    if (
                        isinstance(argument, ast.Constant)
                        and not isinstance(argument.value, bool)
                        and isinstance(argument.value, str | int | float)
                        and not _approved_exact(argument.value, approved_values)
                    ):
                        found.add((f"{function_name}.arg{index}", argument.value))
    return found


def _assignment_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    ):
        return target.slice.value
    return ""


def _call_leaf_name(function: ast.AST) -> str:
    if isinstance(function, ast.Name):
        return function.id.casefold()
    if isinstance(function, ast.Attribute):
        return function.attr.casefold()
    return ""


def _approved_exact(value: Any, approved_values: Iterable[Any]) -> bool:
    return any(
        type(candidate) is type(value) and candidate == value for candidate in approved_values
    )


def _diagnostic_code(diagnostics: Iterable[str]) -> str | None:
    for diagnostic in diagnostics:
        match = re.match(r"([A-Z][A-Z0-9_]+)", diagnostic)
        if match:
            return match.group(1)
    return None


def _context_manifest_payload(
    context_id: str, payload: dict[str, Any], context_checksum: str
) -> dict[str, Any]:
    symbol_refs: list[dict[str, Any]] = []
    for group in ("matched_symbols", "test_lib_symbols", "neighbor_symbols"):
        for symbol in payload.get(group, []):
            if not isinstance(symbol, Mapping):
                continue
            symbol_refs.append(
                {
                    "group": group,
                    "symbol_id": symbol.get("symbol_id"),
                    "fqn": symbol.get("fqn"),
                    "relative_path": symbol.get("relative_path"),
                    "body_hash": symbol.get("body_hash"),
                }
            )
    manifest: dict[str, Any] = {
        "context_manifest_id": context_id,
        "context_checksum": context_checksum,
        "scenario_id": payload.get("scenario_id") or payload.get("scenario", {}).get("scenario_id"),
        "story_ids": [
            item.get("story_id") for item in payload.get("stories", []) if isinstance(item, Mapping)
        ],
        "requirement_ids": [
            item.get("requirement_id")
            for item in payload.get("requirements", [])
            if isinstance(item, Mapping)
        ],
        "evidence_chunk_ids": list(payload.get("evidence_chunk_ids", [])),
        "framework_snapshot": payload.get("framework_snapshot", {}),
        "test_data_snapshot": payload.get("test_data_snapshot", {}),
        "resolved_binding_id": payload.get("resolved_binding", {}).get("binding_id"),
        "symbol_refs": symbol_refs,
        "token_budget": payload.get("token_budget"),
        "selected_tokens": payload.get("selected_tokens"),
        "dropped_items": payload.get("dropped_items", []),
    }
    manifest["manifest_checksum"] = canonical_checksum({"manifest": manifest})
    return manifest


def _validation_evidence(
    result: Any,
    generated_file_hashes: Mapping[str, str],
    *,
    framework_root: Path,
) -> dict[str, Any]:
    shared_ast_fingerprints: dict[str, list[str]] = {}
    for relative_path in generated_file_hashes:
        try:
            role = classify_path(relative_path).role
        except Exception:
            continue
        if role not in {"wrapper", "helper"}:
            continue
        source = (framework_root / relative_path).read_text(encoding="utf-8")
        shared_ast_fingerprints[relative_path] = sorted(_module_statement_fingerprints(source))
    evidence: dict[str, Any] = {
        "pipeline": [
            "path_policy",
            "no_git_policy",
            "additive_ast",
            "python_parse_compile",
            "lifecycle_contract",
            "isolated_import",
            "robot_dryrun",
            "traceability_exact_data",
            "checksums",
        ],
        "commands": [
            {
                "name": command.name,
                "ok": command.ok,
                "returncode": command.returncode,
                "timed_out": command.timed_out,
                "argv": list(command.argv),
            }
            for command in result.commands
        ],
        "generated_file_hashes": dict(generated_file_hashes),
        "shared_ast_fingerprints": shared_ast_fingerprints,
        "diagnostics": list(result.diagnostics),
        "ok": bool(result.ok),
    }
    evidence["evidence_checksum"] = canonical_checksum({"evidence": evidence})
    return evidence


def make_provider_fingerprint(
    settings: AppSettings,
    provider: Literal["azure_openai", "gemini"],
    *,
    prompt_revision: str = PROMPT_REVISION,
) -> ProviderFingerprint:
    """Build the run pin without inspecting configuration for the other provider."""

    if provider == "azure_openai":
        model = settings.azure_openai.reasoning_deployment
        revision = None
        params: dict[str, Any] = {"structured_output": True, "sdk_retries": 0}
    else:
        model = settings.gemini.reasoning_model
        revision = None
        params = {"structured_output": True, "sdk_retries": 0}
    if not model:
        raise ValueError(f"selected Stage-4 provider {provider} has no configured model")
    params_checksum = canonical_checksum({"params": params})
    fingerprint_hash = make_provider_fingerprint_hash(
        provider=provider,
        model=model,
        model_revision=revision,
        generation_params_checksum=params_checksum,
        prompt_revision=prompt_revision,
    )
    return ProviderFingerprint(
        provider=provider,
        model=model,
        model_revision=revision,
        generation_params=params,
        prompt_revision=prompt_revision,
        fingerprint_hash=fingerprint_hash,
    )


class LangGraphScenarioRunner:
    """Production/integration adapter used by the serial parent run graph."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        request: Stage4Request,
        codegen_store: CodegenPostgresStore,
        code_graph_store: CodeGraphStore,
        patch_producer: PatchProducer,
        context_retriever: CodegenContextRetriever,
        checkpointer: Any = None,
    ) -> None:
        self.settings = settings
        self.request = request
        self.codegen_store = codegen_store
        self.code_graph_store = code_graph_store
        self.patch_producer = patch_producer
        self.context_retriever = context_retriever
        self.checkpointer = checkpointer
        self.provider_fingerprint = make_provider_fingerprint(settings, request.reasoning_provider)

    def run_scenario(self, context: Any) -> ScenarioExecutionResult:
        frozen_manifest = getattr(context, "frozen_manifest", None)
        if frozen_manifest is not None and not self.provider_fingerprint.matches(
            frozen_manifest.provider_fingerprint
        ):
            raise InputManifestChanged(
                "Stage-4 scenario runner provider pin differs from the frozen manifest"
            )
        scenario = context.scenario
        variant_id = context.variant_id
        normalized = context.normalized_test_data
        snapshot = context.framework_snapshot
        runtime = CodegenRuntime(
            settings=self.settings,
            request=self.request,
            scenario=scenario,
            variant_id=variant_id,
            normalized_test_data=normalized,
            framework_snapshot=snapshot,
            provider_fingerprint=self.provider_fingerprint,
            patch_producer=self.patch_producer,
            context_retriever=self.context_retriever,
            validation_runner=ValidationRunner(
                self.request.framework_path,
                python_executable=sys.executable,
                timeout=self.settings.runtime.timeout_seconds,
                # Robot dry-run is a locked Stage-4 acceptance gate.  The
                # settings schema also rejects attempts to disable it.
                robot_dryrun=True,
            ),
            codegen_store=self.codegen_store,
            code_graph_store=self.code_graph_store,
            checkpointer=self.checkpointer,
        )
        graph = build_scenario_codegen_graph(runtime)
        thread_id = (
            f"{normalize_project(self.request.project_name)}:{self.request.run_id}:"
            f"stage-4:{scenario.scenario_id}:{variant_id}"
        )
        final = graph.invoke(
            {
                "scenario_id": scenario.scenario_id,
                "variant_id": variant_id,
                "status": "STARTED",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        status = final.get("status", "BLOCKED")
        blocker_payload = final.get("blocker")
        blocker = CodegenBlocker.model_validate(blocker_payload) if blocker_payload else None
        ready = None
        if status == "ACCEPTED" and final.get("framework_snapshot_id"):
            ready = self.code_graph_store.get_snapshot(final["framework_snapshot_id"])
            if ready is None or not ready.active:
                if final.get("building_snapshot_id"):
                    raise RuntimeError(
                        "newly accepted case is not visible in an active READY snapshot"
                    )
                # An accepted replay may reference an older retained READY
                # snapshot. No files changed, so the parent must continue from
                # the active snapshot it supplied rather than reactivate history.
                ready = snapshot
        terminal: ScenarioTerminalStatus = (
            "ACCEPTED"
            if status == "ACCEPTED"
            else "REVISION_REQUIRED"
            if status == "REVISION_REQUIRED"
            else "BLOCKED"
        )
        return ScenarioExecutionResult(
            status=terminal,
            scenario_id=scenario.scenario_id,
            variant_id=variant_id,
            tc_id=int(final["tc_id"]),
            accepted_snapshot=ready,
            blocker=blocker,
        )


def build_default_scenario_runner(
    *,
    settings: AppSettings,
    request: Stage4Request,
    codegen_store: CodegenPostgresStore,
    code_graph_store: CodeGraphStore,
    checkpointer: Any = None,
) -> LangGraphScenarioRunner:
    """Construct production dependencies for the explicitly selected provider only."""

    from multi_agentic_graph_rag.db.neo4j_store import Neo4jStore
    from multi_agentic_graph_rag.db.postgres import PostgresStore
    from multi_agentic_graph_rag.llm_models.factory import create_stage4_reasoning_model
    from multi_agentic_graph_rag.services.llm_patch_producer import LLMPatchProducer

    fingerprint = make_provider_fingerprint(settings, request.reasoning_provider)
    model = create_stage4_reasoning_model(
        settings,
        provider=request.reasoning_provider,
        run_dir=(
            settings.paths.generated_dir
            / normalize_project(request.project_name)
            / request.run_id
            / "stage-4"
        ),
    )
    producer = LLMPatchProducer(model=model, provider_fingerprint=fingerprint)
    retriever = CodegenContextRetriever(
        lineage_store=PostgresStore(settings),
        evidence_store=Neo4jStore(settings),
        code_graph=code_graph_store,
        framework_root=request.framework_path,
    )
    return LangGraphScenarioRunner(
        settings=settings,
        request=request,
        codegen_store=codegen_store,
        code_graph_store=code_graph_store,
        patch_producer=producer,
        context_retriever=retriever,
        checkpointer=checkpointer,
    )


__all__ = [
    "GENERATION_POLICY_VERSION",
    "PROMPT_REVISION",
    "CodegenRuntime",
    "LangGraphScenarioRunner",
    "ScenarioExecutionResult",
    "Stage4ScenarioState",
    "build_default_scenario_runner",
    "build_scenario_codegen_graph",
    "make_provider_fingerprint",
]
