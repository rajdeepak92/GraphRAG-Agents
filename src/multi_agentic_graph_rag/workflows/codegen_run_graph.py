"""Frozen, serial parent orchestration for Stage-4 test-code generation.

The parent graph owns run-wide reproducibility and sequencing.  Per-scenario
generation is supplied through :class:`Stage4ScenarioRunner`, which keeps this
module independent of the code-writing implementation and makes readiness-only
runs incapable of accidentally invoking a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast

from langgraph.graph import END, StateGraph

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CodegenBlocker,
    FrozenRunManifest,
    FrozenScenarioEntry,
    ProviderFingerprint,
    ReasoningProviderName,
    Stage4Request,
    TestDataSnapshotRef,
)
from multi_agentic_graph_rag.domain.errors import ConfigurationError, InputManifestChanged
from multi_agentic_graph_rag.domain.identifiers import (
    make_checkpoint_thread_id,
    make_provider_fingerprint_hash,
    normalize_project,
)
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalTestScenario,
    TestScenariosArtifact,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData
from multi_agentic_graph_rag.services.checkpointing import workflow_checkpointer
from multi_agentic_graph_rag.services.framework_indexer import (
    FrameworkIndexResult,
    detect_graphify_capabilities,
    index_framework,
)
from multi_agentic_graph_rag.services.framework_snapshot import (
    compute_filesystem_checksum,
    validate_framework_path,
)
from multi_agentic_graph_rag.services.test_data_document_reader import read_document
from multi_agentic_graph_rag.services.test_data_ingestion import ingest_document

_RUN_ID = re.compile(r"[A-Za-z0-9._-]+")
_SUPPORTED_TEST_DATA_SUFFIXES = frozenset({".json", ".xlsx"})
_PROMPT_REVISION = "stage4-codegen-v1"
_GENERATION_POLICY_VERSION = "stage4-optimized-masterplan-v1"

ScenarioRunStatus = Literal["ACCEPTED", "BLOCKED", "REVISION_REQUIRED"]
Stage4ParentStatus = Literal["DRY_RUN_READY", "COMPLETED", "PARTIAL_FAILED"]


@dataclass(frozen=True)
class ScenarioRunContext:
    """Immutable input passed to exactly one serial per-scenario run."""

    request: Stage4Request
    scenario: CanonicalTestScenario
    variant_id: str
    normalized_test_data: NormalizedTestData
    framework_snapshot: FrameworkSnapshot
    frozen_manifest: FrozenRunManifest
    parent_thread_id: str
    thread_id: str


@dataclass(frozen=True)
class ScenarioRunResult:
    """Small result contract returned by a per-scenario implementation."""

    scenario_id: str
    status: ScenarioRunStatus
    tc_id: int
    accepted_snapshot: FrameworkSnapshot | None = None
    blocker: CodegenBlocker | None = None
    diagnostics: tuple[str, ...] = ()


class Stage4ScenarioRunner(Protocol):
    """Provider-neutral seam between the parent and per-scenario LangGraph."""

    def run_scenario(self, context: ScenarioRunContext) -> ScenarioRunResult:
        """Run one case to acceptance, blocker, or revision-required status."""


@dataclass(frozen=True)
class Stage4RunResult:
    """Public result returned by :func:`run_codegen_run`."""

    project_name: str
    run_id: str
    status: Stage4ParentStatus
    thread_id: str
    manifest_checksum: str
    accepted_tc_ids: tuple[int, ...]
    blocked_tc_ids: tuple[int, ...]
    revision_required_tc_ids: tuple[int, ...]
    framework_snapshot_id: str
    test_data_snapshot_id: str
    artifact_path: Path | None


class Stage4RunState(TypedDict, total=False):
    """Checkpoint-safe state for the required seven-node parent graph."""

    request: dict[str, Any]
    thread_id: str
    framework_path: str
    baseline_framework_checksum: str
    scenarios_artifact: dict[str, Any]
    normalized_test_data: dict[str, Any]
    provider_fingerprint: dict[str, Any]
    frozen_manifest: dict[str, Any]
    manifest_preexisting: bool
    framework_snapshot: dict[str, Any]
    scenario_results: list[dict[str, Any]]
    run_status: str
    artifact_path: str | None


@dataclass
class Stage4RunRuntime:
    """Non-serializable parent dependencies held outside LangGraph state."""

    settings: AppSettings
    stage123_store: PostgresStore
    codegen_store: CodegenPostgresStore
    code_graph_store: CodeGraphStore
    scenario_runner: Stage4ScenarioRunner
    checkpointer: Any = None
    prompt_revision: str = _PROMPT_REVISION
    generation_policy_version: str = _GENERATION_POLICY_VERSION
    graphify_check: Any = detect_graphify_capabilities
    framework_indexer: Any = index_framework
    provider_fingerprint_builder: Any = None

    def __post_init__(self) -> None:
        if self.provider_fingerprint_builder is None:
            self.provider_fingerprint_builder = build_provider_fingerprint


class _LazyDefaultScenarioRunner:
    """Construct the selected model and per-case adapter only after readiness passes."""

    def __init__(
        self,
        *,
        settings: AppSettings,
        request: Stage4Request,
        codegen_store: CodegenPostgresStore,
        code_graph_store: CodeGraphStore,
        checkpointer: Any,
    ) -> None:
        self._settings = settings
        self._request = request
        self._codegen_store = codegen_store
        self._code_graph_store = code_graph_store
        self._checkpointer = checkpointer
        self._delegate: Stage4ScenarioRunner | None = None

    def run_scenario(self, context: ScenarioRunContext) -> ScenarioRunResult:
        if self._delegate is None:
            from multi_agentic_graph_rag.workflows.codegen_apply_graph import (
                build_default_scenario_runner,
            )

            self._delegate = cast(
                Stage4ScenarioRunner,
                build_default_scenario_runner(
                    settings=self._settings,
                    request=self._request,
                    codegen_store=self._codegen_store,
                    code_graph_store=self._code_graph_store,
                    checkpointer=self._checkpointer,
                ),
            )
        delegate = self._delegate
        if delegate is None:  # defensive; construction either returns or raises
            raise RuntimeError("per-scenario runner construction did not return a runner")
        return delegate.run_scenario(context)


class _DryRunGuard:
    def run_scenario(self, _context: ScenarioRunContext) -> ScenarioRunResult:
        raise AssertionError("dry-run must not invoke the per-scenario runner")


def build_codegen_run_graph(runtime: Stage4RunRuntime) -> Any:
    """Build the required parent graph with a stable, serial scenario loop."""
    graph: StateGraph[Stage4RunState] = StateGraph(Stage4RunState)
    graph.add_node("validate_request", _validate_request)
    graph.add_node("validate_environment", lambda state: _validate_environment(state, runtime))
    graph.add_node("freeze_inputs", lambda state: _freeze_inputs(state, runtime))
    graph.add_node(
        "prepare_framework_and_test_data",
        lambda state: _prepare_framework_and_test_data(state, runtime),
    )
    graph.add_node("run_scenario_loop", lambda state: _run_scenario_loop(state, runtime))
    graph.add_node("validate_run_exit", _validate_run_exit)
    graph.add_node("write_final_artifacts", lambda state: _write_final_artifacts(state, runtime))
    graph.set_entry_point("validate_request")
    graph.add_edge("validate_request", "validate_environment")
    graph.add_edge("validate_environment", "freeze_inputs")
    graph.add_edge("freeze_inputs", "prepare_framework_and_test_data")
    graph.add_edge("prepare_framework_and_test_data", "run_scenario_loop")
    graph.add_edge("run_scenario_loop", "validate_run_exit")
    graph.add_edge("validate_run_exit", "write_final_artifacts")
    graph.add_edge("write_final_artifacts", END)
    return graph.compile(checkpointer=runtime.checkpointer)


def _validate_request(state: Stage4RunState) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    if _RUN_ID.fullmatch(request.run_id) is None:
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")
    suffix = request.test_data_document.suffix.lower()
    if suffix not in _SUPPORTED_TEST_DATA_SUFFIXES:
        raise ValueError("test_data_document must be .xlsx or .json")
    thread_id = make_checkpoint_thread_id(request.project_name, request.run_id, "stage-4")
    return {"request": request.model_dump(mode="json"), "thread_id": thread_id}


def _validate_environment(state: Stage4RunState, runtime: Stage4RunRuntime) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    framework_path = validate_framework_path(
        request.framework_path, runtime.settings.stage4.framework_allowed_roots
    )
    document = request.test_data_document.resolve(strict=True)
    if not document.is_file():
        raise ValueError(f"test-data document is not a file: {document}")
    fingerprint = runtime.provider_fingerprint_builder(
        runtime.settings, request.reasoning_provider, runtime.prompt_revision
    )
    runtime.graphify_check(runtime.settings.stage4.graphify_command)
    if not request.dry_run:
        runtime.codegen_store.ensure_schema()
        runtime.code_graph_store.ensure_schema()
    return {
        "framework_path": str(framework_path),
        "baseline_framework_checksum": compute_filesystem_checksum(framework_path),
        "provider_fingerprint": fingerprint.model_dump(mode="json"),
    }


def _freeze_inputs(state: Stage4RunState, runtime: Stage4RunRuntime) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    scenarios = runtime.stage123_store.load_test_scenarios(request.project_name, request.run_id)
    if scenarios is None:
        raise ValueError("frozen Stage-3 scenarios are unavailable for the selected project/run")
    _validate_scenario_artifact(request, scenarios)
    existing = runtime.codegen_store.load_frozen_manifest(request.project_name, request.run_id)
    if existing is not None:
        scenarios = _frozen_scenario_subset(scenarios, existing)

    unknown_variants = sorted(
        set(request.variant_selection) - {scenario.scenario_id for scenario in scenarios.scenarios}
    )
    if unknown_variants:
        raise ValueError(f"variant_selection references unknown scenarios: {unknown_variants}")

    raw_document = read_document(request.test_data_document)
    if raw_document.project != request.project_name:
        raise ValueError("test-data project does not match the Stage-4 request")
    ingestion = ingest_document(
        raw_document,
        scenario_ids={scenario.scenario_id for scenario in scenarios.scenarios},
    )
    if not ingestion.is_ready or ingestion.normalized is None:
        issues = "; ".join(
            f"{issue.issue_code}: {issue.message}" for issue in ingestion.report.issues
        )
        raise ValueError(f"test-data document is not READY: {issues or 'validation failed'}")
    normalized = ingestion.normalized
    resolved_variants = _resolve_variant_selection(request, scenarios, normalized)
    fingerprint = ProviderFingerprint.model_validate(state["provider_fingerprint"])
    manifest = _build_frozen_manifest(
        request=request,
        scenarios=scenarios,
        normalized=normalized,
        # Accepted cases legitimately mutate the framework during this run.
        # Preserve the originally frozen baseline on resume and validate the
        # current filesystem independently against a READY snapshot below.
        baseline_checksum=(
            existing.baseline_framework_checksum
            if existing is not None
            else state["baseline_framework_checksum"]
        ),
        provider_fingerprint=fingerprint,
        policy_version=runtime.generation_policy_version,
        resolved_variants=resolved_variants,
        frozen_at=existing.frozen_at if existing is not None else None,
    )
    if request.dry_run:
        if existing is not None and existing != manifest:
            raise InputManifestChanged(
                f"frozen input manifest changed for {request.project_name}/{request.run_id}"
            )
    else:
        runtime.codegen_store.save_or_validate_frozen_manifest(manifest)
    return {
        "scenarios_artifact": scenarios.model_dump(mode="json"),
        "normalized_test_data": normalized.model_dump(mode="json"),
        "frozen_manifest": manifest.model_dump(mode="json"),
        "manifest_preexisting": existing is not None,
    }


def _prepare_framework_and_test_data(
    state: Stage4RunState, runtime: Stage4RunRuntime
) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    normalized = NormalizedTestData.model_validate(state["normalized_test_data"])
    framework_path = Path(state["framework_path"])
    snapshot = runtime.code_graph_store.find_ready_snapshot(
        canonical_path=str(framework_path),
        filesystem_checksum=state["baseline_framework_checksum"],
    )
    manifest = FrozenRunManifest.model_validate(state["frozen_manifest"])

    if request.dry_run:
        if snapshot is None or not snapshot.active:
            raise ValueError(
                "dry-run requires an active READY framework snapshot for the exact filesystem"
            )
        return {"framework_snapshot": snapshot.model_dump(mode="json")}

    snapshot_ref = TestDataSnapshotRef(
        snapshot_id=normalized.snapshot_id,
        project=normalized.project,
        schema_version=normalized.schema_version,
        workbook_checksum=normalized.workbook_checksum,
        normalized_checksum=normalized.checksum,
        decision_revision=normalized.decision_revision,
        status="ready",
    )
    runtime.codegen_store.save_test_data_snapshot(snapshot_ref, normalized.model_dump(mode="json"))
    runtime.code_graph_store.publish_test_data(normalized)

    if snapshot is None or not snapshot.active:
        if (
            state.get("manifest_preexisting", False)
            and state["baseline_framework_checksum"] != manifest.baseline_framework_checksum
        ):
            raise InputManifestChanged(
                "resumed framework content is not represented by the active READY snapshot"
            )
        indexed = cast(
            FrameworkIndexResult,
            runtime.framework_indexer(
                settings=runtime.settings,
                framework_path=framework_path,
                graphify_out_dir=framework_path / "graphify-out",
                test_data_snapshot_id=normalized.snapshot_id,
            ),
        )
        snapshot = indexed.snapshot
    if snapshot.status != "ready" or not snapshot.active:
        raise ValueError("framework snapshot is not READY and active")
    runtime.codegen_store.save_codegen_run(
        codegen_run_id=_codegen_run_id(request),
        project=request.project_name,
        run_id=request.run_id,
        framework_snapshot_id=snapshot.snapshot_id,
        test_data_snapshot_id=normalized.snapshot_id,
        execution_profile_id=request.execution_profile_id,
        status="RUNNING",
        payload={"frozen_manifest_checksum": state["frozen_manifest"]["checksum"]},
    )
    return {"framework_snapshot": snapshot.model_dump(mode="json")}


def _run_scenario_loop(state: Stage4RunState, runtime: Stage4RunRuntime) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    if request.dry_run:
        return {"scenario_results": []}

    scenarios = TestScenariosArtifact.model_validate(state["scenarios_artifact"])
    normalized = NormalizedTestData.model_validate(state["normalized_test_data"])
    manifest = FrozenRunManifest.model_validate(state["frozen_manifest"])
    variants = {entry.scenario_id: entry.variant_id for entry in manifest.scenarios}
    snapshot = FrameworkSnapshot.model_validate(state["framework_snapshot"])
    parent_thread_id = state["thread_id"]
    results: list[dict[str, Any]] = []

    for scenario in scenarios.scenarios:
        variant_id = variants[scenario.scenario_id]
        context = ScenarioRunContext(
            request=request,
            scenario=scenario,
            variant_id=variant_id,
            normalized_test_data=normalized,
            framework_snapshot=snapshot,
            frozen_manifest=manifest,
            parent_thread_id=parent_thread_id,
            thread_id=f"{parent_thread_id}:{scenario.scenario_id}:{variant_id}",
        )
        result = runtime.scenario_runner.run_scenario(context)
        _validate_scenario_result(result, scenario.scenario_id)
        results.append(_scenario_result_payload(result))
        if result.status == "ACCEPTED":
            assert result.accepted_snapshot is not None
            snapshot = result.accepted_snapshot
    return {
        "scenario_results": results,
        "framework_snapshot": snapshot.model_dump(mode="json"),
    }


def _validate_run_exit(state: Stage4RunState) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    if request.dry_run:
        return {"run_status": "DRY_RUN_READY"}
    manifest = FrozenRunManifest.model_validate(state["frozen_manifest"])
    results = state.get("scenario_results", [])
    expected_ids = [entry.scenario_id for entry in manifest.scenarios]
    actual_ids = [str(result["scenario_id"]) for result in results]
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"Stage-4 serial exit mismatch: expected {expected_ids}, received {actual_ids}"
        )
    status: Stage4ParentStatus = (
        "COMPLETED"
        if all(result["status"] == "ACCEPTED" for result in results)
        else "PARTIAL_FAILED"
    )
    return {"run_status": status}


def _write_final_artifacts(state: Stage4RunState, runtime: Stage4RunRuntime) -> Stage4RunState:
    request = Stage4Request.model_validate(state["request"])
    if request.dry_run:
        return {"artifact_path": None}
    snapshot = FrameworkSnapshot.model_validate(state["framework_snapshot"])
    normalized = NormalizedTestData.model_validate(state["normalized_test_data"])
    status = state["run_status"]
    runtime.codegen_store.save_codegen_run(
        codegen_run_id=_codegen_run_id(request),
        project=request.project_name,
        run_id=request.run_id,
        framework_snapshot_id=snapshot.snapshot_id,
        test_data_snapshot_id=normalized.snapshot_id,
        execution_profile_id=request.execution_profile_id,
        status=status,
        payload={
            "frozen_manifest_checksum": state["frozen_manifest"]["checksum"],
            "scenario_results": state.get("scenario_results", []),
        },
    )
    artifact = runtime.codegen_store.write_test_cases_artifact(
        request.project_name,
        request.run_id,
        generated_dir=runtime.settings.paths.generated_dir,
    )
    return {"artifact_path": str(artifact)}


def build_provider_fingerprint(
    settings: AppSettings,
    provider: ReasoningProviderName,
    prompt_revision: str = _PROMPT_REVISION,
) -> ProviderFingerprint:
    """Validate and fingerprint only the explicitly selected provider."""
    if provider == "azure_openai":
        selected = settings.azure_openai
        if not selected.endpoint or not selected.api_key or not selected.reasoning_deployment:
            raise ConfigurationError(
                "azure_openai Stage 4 mode requires endpoint, authentication, and deployment"
            )
        if find_spec("openai") is None:
            raise ConfigurationError("azure_openai Stage 4 mode requires the azure extra")
        model = selected.reasoning_deployment
        revision = None
        params: dict[str, Any] = {"structured_output": True, "sdk_retries": 0}
    elif provider == "gemini":
        selected_gemini = settings.gemini
        if not selected_gemini.api_key:
            raise ConfigurationError("gemini Stage 4 mode requires GEMINI_API_KEY")
        if not selected_gemini.reasoning_model:
            raise ConfigurationError(
                "gemini Stage 4 mode requires an explicitly configured reasoning model"
            )
        if find_spec("google.genai") is None:
            raise ConfigurationError("gemini Stage 4 mode requires the gemini extra")
        model = selected_gemini.reasoning_model
        revision = None
        params = {"structured_output": True, "sdk_retries": 0}
    else:
        raise ConfigurationError(f"Unsupported Stage 4 reasoning provider: {provider}")
    return _make_provider_fingerprint(
        provider=provider,
        model=model,
        revision=revision,
        params=params,
        prompt_revision=prompt_revision,
    )


def _make_provider_fingerprint(
    *,
    provider: ReasoningProviderName,
    model: str,
    revision: str | None,
    params: dict[str, Any],
    prompt_revision: str,
) -> ProviderFingerprint:
    return ProviderFingerprint(
        provider=provider,
        model=model,
        model_revision=revision,
        generation_params=params,
        prompt_revision=prompt_revision,
        fingerprint_hash=make_provider_fingerprint_hash(
            provider=provider,
            model=model,
            model_revision=revision,
            generation_params_checksum=canonical_checksum({"params": params}),
            prompt_revision=prompt_revision,
        ),
    )


def _validate_scenario_artifact(request: Stage4Request, scenarios: TestScenariosArtifact) -> None:
    if scenarios.project != request.project_name or scenarios.run_id != request.run_id:
        raise ValueError("Stage-3 scenario artifact project/run scope mismatch")
    ids = [scenario.scenario_id for scenario in scenarios.scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage-3 scenario artifact contains duplicate scenario IDs")


def _frozen_scenario_subset(
    current: TestScenariosArtifact, manifest: FrozenRunManifest
) -> TestScenariosArtifact:
    """Keep the immutable frozen order and ignore Stage-3 scenarios added later."""
    by_id = {scenario.scenario_id: scenario for scenario in current.scenarios}
    frozen_ids = [entry.scenario_id for entry in manifest.scenarios]
    missing = [scenario_id for scenario_id in frozen_ids if scenario_id not in by_id]
    if missing:
        raise InputManifestChanged(f"frozen Stage-3 scenarios are missing on resume: {missing}")
    draft = TestScenariosArtifact.model_construct(
        project=current.project,
        run_id=current.run_id,
        scenarios=[by_id[scenario_id] for scenario_id in frozen_ids],
        checksum="",
    )
    return TestScenariosArtifact.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


def _resolve_variant_selection(
    request: Stage4Request,
    scenarios: TestScenariosArtifact,
    normalized: NormalizedTestData,
) -> dict[str, str]:
    """Resolve one deterministic, approved data variant for every frozen scenario.

    An explicit request value must name an available variant.  Without one, a
    sole approved scenario/profile variant is selected (including a non-default
    name).  Multiple variants are intentionally ambiguous and must be selected
    explicitly before the immutable manifest is written.
    """
    resolved: dict[str, str] = {}
    for scenario in scenarios.scenarios:
        available = sorted(
            {
                binding.variant_id or "default"
                for binding in normalized.bindings
                if binding.scenario_id == scenario.scenario_id
                and binding.execution_profile_id == request.execution_profile_id
                and binding.approval_status == "APPROVED"
            }
        )
        explicit = request.variant_selection.get(scenario.scenario_id)
        if explicit is not None:
            if explicit not in available:
                raise ValueError(
                    "variant_selection references an unavailable approved binding: "
                    f"{scenario.scenario_id}={explicit}"
                )
            resolved[scenario.scenario_id] = explicit
            continue
        if len(available) == 1:
            resolved[scenario.scenario_id] = available[0]
            continue
        if not available:
            raise ValueError(
                "no approved test-data binding exists for scenario/profile: "
                f"{scenario.scenario_id}/{request.execution_profile_id}"
            )
        raise ValueError(
            "multiple test-data variants require an explicit variant_selection: "
            f"{scenario.scenario_id}={available}"
        )
    return resolved


def _build_frozen_manifest(
    *,
    request: Stage4Request,
    scenarios: TestScenariosArtifact,
    normalized: NormalizedTestData,
    baseline_checksum: str,
    provider_fingerprint: ProviderFingerprint,
    policy_version: str,
    resolved_variants: dict[str, str],
    frozen_at: Any,
) -> FrozenRunManifest:
    fields: dict[str, Any] = {
        "project_name": request.project_name,
        "run_id": request.run_id,
        "scenarios": [
            FrozenScenarioEntry(
                scenario_id=scenario.scenario_id,
                scenario_checksum=canonical_checksum(scenario),
                story_ids=list(scenario.story_ids),
                requirement_ids=list(scenario.requirement_ids),
                variant_id=resolved_variants[scenario.scenario_id],
            )
            for scenario in scenarios.scenarios
        ],
        "execution_profile_id": request.execution_profile_id,
        "test_data_snapshot_checksum": normalized.checksum,
        "baseline_framework_checksum": baseline_checksum,
        "provider_fingerprint": provider_fingerprint,
        "generation_policy_version": policy_version,
        "checksum": "",
    }
    if frozen_at is not None:
        fields["frozen_at"] = frozen_at
    draft = FrozenRunManifest.model_construct(**fields)
    return FrozenRunManifest.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


def _validate_scenario_result(result: ScenarioRunResult, scenario_id: str) -> None:
    if result.scenario_id != scenario_id:
        raise RuntimeError(
            f"per-scenario runner returned {result.scenario_id!r} for {scenario_id!r}"
        )
    if not 100001 <= result.tc_id <= 999999:
        raise RuntimeError(f"per-scenario runner returned invalid TC ID {result.tc_id}")
    if result.status == "ACCEPTED":
        snapshot = result.accepted_snapshot
        if snapshot is None or snapshot.status != "ready" or not snapshot.active:
            raise RuntimeError("accepted scenario must return its active READY framework snapshot")


def _scenario_result_payload(result: ScenarioRunResult) -> dict[str, Any]:
    return {
        "scenario_id": result.scenario_id,
        "status": result.status,
        "tc_id": result.tc_id,
        "accepted_snapshot_id": (
            result.accepted_snapshot.snapshot_id if result.accepted_snapshot else None
        ),
        "blocker": result.blocker.model_dump(mode="json") if result.blocker else None,
        "diagnostics": list(getattr(result, "diagnostics", ())),
    }


def _codegen_run_id(request: Stage4Request) -> str:
    return f"{normalize_project(request.project_name)}:{request.run_id}:stage-4"


def build_default_codegen_runtime(
    settings: AppSettings,
    request: Stage4Request,
    *,
    checkpointer: Any = None,
) -> Stage4RunRuntime:
    """Create production stores and a lazily initialized per-scenario adapter."""
    codegen_store = CodegenPostgresStore(settings)
    code_graph_store = CodeGraphStore(settings)
    runner: Stage4ScenarioRunner
    if request.dry_run:
        runner = _DryRunGuard()
    else:
        runner = _LazyDefaultScenarioRunner(
            settings=settings,
            request=request,
            codegen_store=codegen_store,
            code_graph_store=code_graph_store,
            checkpointer=checkpointer,
        )
    return Stage4RunRuntime(
        settings=settings,
        stage123_store=PostgresStore(settings),
        codegen_store=codegen_store,
        code_graph_store=code_graph_store,
        scenario_runner=runner,
        checkpointer=checkpointer,
    )


def run_codegen_run(
    request: Stage4Request,
    *,
    settings: AppSettings | None = None,
    runtime: Stage4RunRuntime | None = None,
) -> Stage4RunResult:
    """Run the Stage-4 parent graph under its stable durable thread ID."""
    selected_settings = settings or (runtime.settings if runtime is not None else load_config())
    thread_id = make_checkpoint_thread_id(request.project_name, request.run_id, "stage-4")

    if runtime is not None:
        return _invoke_parent(request, runtime, thread_id)
    if request.dry_run:
        # A readiness run is deliberately checkpoint-free: even checkpoint
        # schema setup would violate the write-free dry-run contract.
        dry_runtime = build_default_codegen_runtime(selected_settings, request, checkpointer=None)
        return _invoke_parent(request, dry_runtime, thread_id)
    with workflow_checkpointer(selected_settings) as checkpointer:
        default_runtime = build_default_codegen_runtime(
            selected_settings, request, checkpointer=checkpointer
        )
        return _invoke_parent(request, default_runtime, thread_id)


def _invoke_parent(
    request: Stage4Request, runtime: Stage4RunRuntime, thread_id: str
) -> Stage4RunResult:
    graph = build_codegen_run_graph(runtime)
    final = graph.invoke(
        {"request": request.model_dump(mode="json")},
        config={"configurable": {"thread_id": thread_id}},
    )
    results = final.get("scenario_results", [])
    accepted = tuple(int(row["tc_id"]) for row in results if row["status"] == "ACCEPTED")
    blocked = tuple(int(row["tc_id"]) for row in results if row["status"] == "BLOCKED")
    revisions = tuple(int(row["tc_id"]) for row in results if row["status"] == "REVISION_REQUIRED")
    snapshot = FrameworkSnapshot.model_validate(final["framework_snapshot"])
    normalized = NormalizedTestData.model_validate(final["normalized_test_data"])
    manifest = FrozenRunManifest.model_validate(final["frozen_manifest"])
    return Stage4RunResult(
        project_name=request.project_name,
        run_id=request.run_id,
        status=cast(Stage4ParentStatus, final["run_status"]),
        thread_id=thread_id,
        manifest_checksum=manifest.checksum,
        accepted_tc_ids=accepted,
        blocked_tc_ids=blocked,
        revision_required_tc_ids=revisions,
        framework_snapshot_id=snapshot.snapshot_id,
        test_data_snapshot_id=normalized.snapshot_id,
        artifact_path=Path(final["artifact_path"]) if final.get("artifact_path") else None,
    )


__all__ = [
    "ScenarioRunContext",
    "ScenarioRunResult",
    "Stage4RunResult",
    "Stage4RunRuntime",
    "Stage4ScenarioRunner",
    "build_codegen_run_graph",
    "build_default_codegen_runtime",
    "build_provider_fingerprint",
    "run_codegen_run",
]
