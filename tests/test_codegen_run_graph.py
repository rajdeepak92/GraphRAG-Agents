"""Parent Stage-4 run graph and CLI contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

import multi_agentic_graph_rag.cli as cli_module
from multi_agentic_graph_rag.cli import app
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.codegen_schemas import (
    FrozenRunManifest,
    ProviderFingerprint,
    Stage4Request,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.identifiers import make_provider_fingerprint_hash
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalTestScenario,
    TestScenariosArtifact,
    Traceability,
)
from multi_agentic_graph_rag.workflows.codegen_run_graph import (
    ScenarioRunContext,
    ScenarioRunResult,
    Stage4RunResult,
    Stage4RunRuntime,
    run_codegen_run,
)


def _scenario(identifier: str, title: str) -> CanonicalTestScenario:
    return CanonicalTestScenario(
        source_req_id=None,
        source_req_id_type="generated",
        scenario_id=identifier,
        story_ids=[f"US-{identifier}-1", f"US-{identifier}-2"],
        requirement_ids=[f"REQ-{identifier}-1", f"REQ-{identifier}-2"],
        title=title,
        description=f"Exercise {title}",
        scenario_type="Positive",
        priority="High",
        preconditions=["Fixture is ready"],
        action="Perform the approved action",
        expected_result="The approved oracle succeeds",
        covered_acceptance_criterion_ids=[f"AC-{identifier}"],
        confidence=0.95,
        traceability=Traceability(
            evidence_chunk_ids=[f"CHK-{identifier}"],
            entity_ids=[],
            relationship_ids=[],
        ),
    )


def _artifact(scenarios: list[CanonicalTestScenario]) -> TestScenariosArtifact:
    draft = TestScenariosArtifact.model_construct(
        project="demo",
        run_id="RUN-001",
        scenarios=scenarios,
        checksum="",
    )
    return TestScenariosArtifact.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


def _record(
    record_id: str, record_type: str, payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_status": "APPROVED",
        "name": record_id,
        "owner": "qa",
        "valid_from_revision": "r1",
        "record_type": record_type,
        "natural_key": record_id.lower(),
        "source_sheet": record_type,
        "source_row": 2,
        "payload": payload or {},
    }


def _write_test_data(path: Path, scenario_ids: list[str]) -> None:
    payload = {
        "manifest": {
            "project": "demo",
            "schema_version": "1.0",
            "workbook_checksum": "sha256:workbook",
            "decision_revision": "r1",
        },
        "records": [
            _record("EP-1", "ExecutionProfile"),
            _record("FIX-1", "Fixture"),
            _record("CLEAN-1", "Cleanup"),
            _record("ORA-1", "Oracle", {"predicate": {"equals": True}}),
        ],
        "bindings": [
            {
                "binding_id": f"BND-{scenario_id}",
                "scenario_id": scenario_id,
                "execution_profile_id": "EP-1",
                "fixture_id": "FIX-1",
                "cleanup_id": "CLEAN-1",
                "oracle_ids": ["ORA-1"],
                "approval_status": "APPROVED",
            }
            for scenario_id in scenario_ids
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _snapshot(identifier: str, root: Path, checksum: str) -> FrameworkSnapshot:
    return FrameworkSnapshot(
        snapshot_id=identifier,
        repository_id="repo",
        canonical_path=str(root.resolve()),
        filesystem_checksum=checksum,
        extractor_version="test",
        extractor_config_hash="cfg",
        status="ready",
        active=True,
    )


def _fingerprint(_settings: Any, provider: str, prompt_revision: str) -> ProviderFingerprint:
    params = {"temperature": 0.0}
    return ProviderFingerprint(
        provider=provider,
        model="selected-model",
        generation_params=params,
        prompt_revision=prompt_revision,
        fingerprint_hash=make_provider_fingerprint_hash(
            provider=provider,
            model="selected-model",
            model_revision=None,
            generation_params_checksum=canonical_checksum({"params": params}),
            prompt_revision=prompt_revision,
        ),
    )


class _Stage123Store:
    def __init__(self, artifact: TestScenariosArtifact) -> None:
        self.artifact = artifact

    def load_test_scenarios(self, _project: str, _run_id: str) -> TestScenariosArtifact:
        return self.artifact


class _CodegenStore:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.manifest: FrozenRunManifest | None = None
        self.schema_calls = 0
        self.test_data_calls = 0
        self.run_statuses: list[str] = []
        self.artifact_calls = 0

    def ensure_schema(self) -> None:
        self.schema_calls += 1

    def load_frozen_manifest(self, _project: str, _run_id: str) -> FrozenRunManifest | None:
        return self.manifest

    def save_or_validate_frozen_manifest(self, manifest: FrozenRunManifest) -> bool:
        if self.manifest is not None and self.manifest != manifest:
            raise AssertionError("test fake received changed frozen manifest")
        created = self.manifest is None
        self.manifest = manifest
        return created

    def save_test_data_snapshot(self, _snapshot: Any, _payload: dict[str, Any]) -> bool:
        self.test_data_calls += 1
        return True

    def save_codegen_run(self, **kwargs: Any) -> None:
        self.run_statuses.append(str(kwargs["status"]))

    def write_test_cases_artifact(self, project: str, run_id: str, *, generated_dir: Path) -> Path:
        self.artifact_calls += 1
        path = generated_dir / project / run_id / "test-cases" / "test_cases.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
        return path


class _CodeGraphStore:
    def __init__(self, snapshot: FrameworkSnapshot) -> None:
        self.snapshot = snapshot
        self.schema_calls = 0
        self.test_data_calls = 0

    def ensure_schema(self) -> None:
        self.schema_calls += 1

    def find_ready_snapshot(self, **_kwargs: str) -> FrameworkSnapshot:
        return self.snapshot

    def publish_test_data(self, _normalized: Any) -> None:
        self.test_data_calls += 1


class _SerialRunner:
    def __init__(self, root: Path, checksum: str) -> None:
        self.root = root
        self.checksum = checksum
        self.contexts: list[ScenarioRunContext] = []

    def run_scenario(self, context: ScenarioRunContext) -> ScenarioRunResult:
        self.contexts.append(context)
        if len(self.contexts) == 1:
            return ScenarioRunResult(
                scenario_id=context.scenario.scenario_id,
                status="ACCEPTED",
                tc_id=100001,
                accepted_snapshot=_snapshot("FWS-AFTER-ONE", self.root, self.checksum),
            )
        return ScenarioRunResult(
            scenario_id=context.scenario.scenario_id,
            status="BLOCKED",
            tc_id=100002,
            diagnostics=("bounded validation repairs exhausted",),
        )


def _runtime(
    tmp_path: Path, *, dry_run: bool = False
) -> tuple[Stage4Request, Stage4RunRuntime, Any]:
    framework = tmp_path / "framework"
    framework.mkdir()
    (framework / "library.py").write_text("def existing():\n    return True\n", encoding="utf-8")
    test_data = tmp_path / "test-data.json"
    scenario_ids = ["TS-1", "TS-2"]
    _write_test_data(test_data, scenario_ids)
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.neo4j.mode = "local_json"
    settings.paths.generated_dir = tmp_path / "generated"
    settings.stage4.framework_allowed_roots = [tmp_path]
    from multi_agentic_graph_rag.services.framework_snapshot import compute_filesystem_checksum

    checksum = compute_filesystem_checksum(framework)
    graph_store = _CodeGraphStore(_snapshot("FWS-BASE", framework, checksum))
    codegen_store = _CodegenStore(settings.paths.generated_dir)
    runner = _SerialRunner(framework, checksum)
    request = Stage4Request(
        project_name="demo",
        run_id="RUN-001",
        framework_path=framework,
        test_data_document=test_data,
        execution_profile_id="EP-1",
        reasoning_provider="azure_openai",
        dry_run=dry_run,
    )
    runtime = Stage4RunRuntime(
        settings=settings,
        stage123_store=_Stage123Store(
            _artifact([_scenario("TS-1", "First"), _scenario("TS-2", "Second")])
        ),  # type: ignore[arg-type]
        codegen_store=codegen_store,  # type: ignore[arg-type]
        code_graph_store=graph_store,  # type: ignore[arg-type]
        scenario_runner=runner,
        graphify_check=lambda _command: object(),
        provider_fingerprint_builder=_fingerprint,
    )
    return request, runtime, runner


def test_parent_runs_scenarios_serially_and_updates_context_snapshot(tmp_path: Path) -> None:
    request, runtime, runner = _runtime(tmp_path)
    result = run_codegen_run(request, runtime=runtime)

    assert result.status == "PARTIAL_FAILED"
    assert result.accepted_tc_ids == (100001,)
    assert result.blocked_tc_ids == (100002,)
    assert [item.scenario.scenario_id for item in runner.contexts] == ["TS-1", "TS-2"]
    assert runner.contexts[0].framework_snapshot.snapshot_id == "FWS-BASE"
    assert runner.contexts[1].framework_snapshot.snapshot_id == "FWS-AFTER-ONE"
    assert runner.contexts[0].thread_id == "demo:RUN-001:stage-4:TS-1:default"
    assert runtime.codegen_store.manifest.scenarios[0].story_ids == ["US-TS-1-1", "US-TS-1-2"]
    assert runtime.codegen_store.run_statuses == ["RUNNING", "PARTIAL_FAILED"]
    assert result.artifact_path is not None and result.artifact_path.exists()


def test_partial_run_resume_preserves_frozen_baseline_after_accepted_files(
    tmp_path: Path,
) -> None:
    request, runtime, runner = _runtime(tmp_path)
    run_codegen_run(request, runtime=runtime)
    assert runtime.codegen_store.manifest is not None
    frozen_baseline = runtime.codegen_store.manifest.baseline_framework_checksum

    accepted_file = request.framework_path / "tests" / "sensor" / "Tc100001Accepted.py"
    accepted_file.parent.mkdir(parents=True)
    accepted_file.write_text("class Tc100001Accepted:\n    pass\n", encoding="utf-8")
    from multi_agentic_graph_rag.services.framework_snapshot import compute_filesystem_checksum

    current_checksum = compute_filesystem_checksum(request.framework_path)
    runtime.code_graph_store.snapshot = _snapshot(
        "FWS-ACTIVE-RESUME", request.framework_path, current_checksum
    )

    run_codegen_run(request, runtime=runtime)

    assert runtime.codegen_store.manifest.baseline_framework_checksum == frozen_baseline
    assert runner.contexts[2].framework_snapshot.snapshot_id == "FWS-ACTIVE-RESUME"


def test_resume_ignores_stage3_scenarios_added_after_freeze(tmp_path: Path) -> None:
    request, runtime, runner = _runtime(tmp_path)
    run_codegen_run(request, runtime=runtime)
    runtime.stage123_store.artifact = _artifact(
        [_scenario("TS-1", "First"), _scenario("TS-2", "Second"), _scenario("TS-3", "New")]
    )

    run_codegen_run(request, runtime=runtime)

    assert runtime.codegen_store.manifest is not None
    assert [entry.scenario_id for entry in runtime.codegen_store.manifest.scenarios] == [
        "TS-1",
        "TS-2",
    ]
    assert [context.scenario.scenario_id for context in runner.contexts] == [
        "TS-1",
        "TS-2",
        "TS-1",
        "TS-2",
    ]


def test_dry_run_is_read_only_and_never_invokes_scenario_runner(tmp_path: Path) -> None:
    request, runtime, runner = _runtime(tmp_path, dry_run=True)
    result = run_codegen_run(request, runtime=runtime)

    assert result.status == "DRY_RUN_READY"
    assert runner.contexts == []
    assert runtime.codegen_store.schema_calls == 0
    assert runtime.codegen_store.test_data_calls == 0
    assert runtime.codegen_store.run_statuses == []
    assert runtime.codegen_store.artifact_calls == 0
    assert runtime.code_graph_store.schema_calls == 0
    assert runtime.code_graph_store.test_data_calls == 0
    assert result.artifact_path is None


def test_parent_derives_the_only_approved_non_default_variant(tmp_path: Path) -> None:
    request, runtime, runner = _runtime(tmp_path)
    payload = json.loads(request.test_data_document.read_text(encoding="utf-8"))
    for binding in payload["bindings"]:
        binding["variant_id"] = "hot"
    request.test_data_document.write_text(json.dumps(payload), encoding="utf-8")

    run_codegen_run(request, runtime=runtime)

    assert [context.variant_id for context in runner.contexts] == ["hot", "hot"]
    assert runtime.codegen_store.manifest is not None
    assert [entry.variant_id for entry in runtime.codegen_store.manifest.scenarios] == [
        "hot",
        "hot",
    ]


def test_parent_requires_explicit_selection_when_multiple_variants_exist(tmp_path: Path) -> None:
    request, runtime, _runner = _runtime(tmp_path)
    payload = json.loads(request.test_data_document.read_text(encoding="utf-8"))
    second = dict(payload["bindings"][0])
    second["binding_id"] = "BND-TS-1-HOT"
    second["variant_id"] = "hot"
    payload["bindings"].append(second)
    request.test_data_document.write_text(json.dumps(payload), encoding="utf-8")

    try:
        run_codegen_run(request, runtime=runtime)
    except ValueError as exc:
        assert "explicit variant_selection" in str(exc)
    else:  # pragma: no cover - guards the frozen-input policy itself
        raise AssertionError("ambiguous variants were accepted")


def _dry_run_result(request: Stage4Request) -> Stage4RunResult:
    return Stage4RunResult(
        project_name=request.project_name,
        run_id=request.run_id,
        status="DRY_RUN_READY",
        thread_id="demo:RUN-001:stage-4",
        manifest_checksum="sha256:manifest",
        accepted_tc_ids=(),
        blocked_tc_ids=(),
        revision_required_tc_ids=(),
        framework_snapshot_id="FWS-1",
        test_data_snapshot_id="TDS-1",
        artifact_path=None,
    )


def test_generate_test_code_cli_has_locked_flags(monkeypatch: Any, tmp_path: Path) -> None:
    captured: list[Stage4Request] = []
    settings = load_config()
    settings.stage4.reasoning_provider = "azure_openai"

    def fake_run(request: Stage4Request, **_kwargs: Any) -> Stage4RunResult:
        captured.append(request)
        return _dry_run_result(request)

    monkeypatch.setattr(cli_module, "load_config", lambda: settings)
    monkeypatch.setattr(cli_module, "run_codegen_run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "generate-test-code",
            "--project",
            "demo",
            "--run-id",
            "RUN-001",
            "--framework-path",
            str(tmp_path / "framework"),
            "--execution-profile",
            "EP-1",
            "--test-data",
            str(tmp_path / "data.xlsx"),
            "--reasoning-provider",
            "huggingface",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured[0].reasoning_provider == "huggingface"
    assert captured[0].dry_run is True
    assert '"status": "DRY_RUN_READY"' in result.stdout


def test_generate_test_code_cli_uses_configured_provider_by_default(
    monkeypatch: Any, tmp_path: Path
) -> None:
    captured: list[Stage4Request] = []
    settings = load_config()
    settings.stage4.reasoning_provider = "huggingface"

    def fake_run(request: Stage4Request, **_kwargs: Any) -> Stage4RunResult:
        captured.append(request)
        return _dry_run_result(request)

    monkeypatch.setattr(cli_module, "load_config", lambda: settings)
    monkeypatch.setattr(cli_module, "run_codegen_run", fake_run)
    result = CliRunner().invoke(
        app,
        [
            "generate-test-code",
            "--project",
            "demo",
            "--run-id",
            "RUN-001",
            "--framework-path",
            str(tmp_path / "framework"),
            "--execution-profile",
            "EP-1",
            "--test-data",
            str(tmp_path / "data.xlsx"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured[0].reasoning_provider == "huggingface"
