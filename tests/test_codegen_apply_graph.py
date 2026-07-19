"""Optimized per-scenario Stage-4 graph tests."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.codegen_schemas import (
    GeneratedFileContent,
    GeneratedPatchBundle,
    LogicalTestCaseKey,
    PlannedFile,
    PlannedStep,
    Stage4Request,
    TestCaseImplementationPlan,
)
from multi_agentic_graph_rag.domain.identifiers import make_logical_key_hash
from multi_agentic_graph_rag.domain.schemas import (
    CanonicalTestScenario,
    Traceability,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData
from multi_agentic_graph_rag.services.direct_file_transaction import (
    DirectFileTransaction,
    JournalStatus,
)
from multi_agentic_graph_rag.services.framework_indexer import FrameworkIndexResult
from multi_agentic_graph_rag.services.patch_producer import StaticPatchProducer
from multi_agentic_graph_rag.services.test_data_document_reader import read_document
from multi_agentic_graph_rag.services.test_data_ingestion import ingest_document
from multi_agentic_graph_rag.services.validation_runner import ValidationRunner
from multi_agentic_graph_rag.workflows.codegen_apply_graph import (
    CodegenRuntime,
    build_scenario_codegen_graph,
    make_provider_fingerprint,
)

STEM = "Tc100001ValidateTemperatureSensorThreshold"
PYTHON_PATH = f"tests/sensor/{STEM}.py"
ROBOT_PATH = f"tests_robot/sensor/{STEM}.robot"
HELPER_PATH = "test_lib/sensor/sensor_helpers.py"

VALID_PYTHON = f"""from __future__ import annotations

import logging


class {STEM}:
    test_variables: dict[str, object] = {{}}

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.test_variables = {{}}

    def test_setup(self) -> bool:
        self.logger.info("Setup stage started")
        try:
            self.logger.info("Setup stage passed")
            return True
        except Exception:
            self.logger.exception("Setup stage failed")
            return False

    def execute_test(self) -> bool:
        step_results: list[bool] = []
        self.logger.info("Execution step 1 started")
        try:
            step_result = bool(True)
            step_results.append(step_result)
            if not step_result:
                self.logger.error("Critical execution step 1 failed")
                return False
        except Exception:
            self.logger.exception("Execution step 1 raised")
            return False
        return bool(step_results) and all(step_results)

    def test_teardown(self) -> bool:
        self.logger.info("Teardown stage started")
        try:
            self.logger.info("Teardown stage passed")
            return True
        except Exception:
            self.logger.exception("Teardown stage failed")
            return False

    def run_test(self) -> bool:
        if not self.test_setup():
            return False

        execution_ok = False
        teardown_ok = False
        try:
            execution_ok = bool(self.execute_test())
        except Exception:
            self.logger.exception("Execution stage failed")
            execution_ok = False
        finally:
            try:
                teardown_ok = bool(self.test_teardown())
            except Exception:
                self.logger.exception("Teardown stage failed")
                teardown_ok = False

        return execution_ok and teardown_ok


if __name__ == "__main__":
    result = {STEM}().run_test()
    raise SystemExit(0 if result else 1)
"""

INVALID_PYTHON = f"class {STEM}(:\n    pass\n"

VALID_ROBOT = f"""*** Settings ***
Library    tests.sensor.{STEM}

*** Test Cases ***
TC100001 Validate Temperature Sensor Threshold
    ${{result}}=    Run Test
    Should Be True    ${{result}}
"""

VALID_HELPER = "def normalize_sensor_result(value: object) -> bool:\n    return bool(value)\n"


class _RetrievedContext:
    def __init__(self, normalized: NormalizedTestData) -> None:
        self.normalized = normalized

    def to_prompt_payload(self) -> dict[str, Any]:
        return {
            "scenario": {"scenario_id": "TS-1"},
            "stories": [{"story_id": "US-1"}, {"story_id": "US-2"}],
            "requirements": [{"requirement_id": "REQ-1"}, {"requirement_id": "REQ-2"}],
            "resolved_binding": {"binding_id": "BND-1"},
            "test_data_records": [
                record.model_dump(mode="json") for record in self.normalized.records
            ],
            "secret_refs": [],
            "framework_snapshot": {"snapshot_id": "FWS-BASE"},
        }


class _ContextRetriever:
    def scenario_codegen_context(self, **kwargs: Any) -> _RetrievedContext:
        return _RetrievedContext(kwargs["normalized_test_data"])


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.neo4j.mode = "local_json"
    settings.paths.generated_dir = tmp_path / "generated"
    settings.stage4.codegen_local_path = tmp_path / "codegen.jsonl"
    settings.stage4.code_graph_local_path = tmp_path / "code_graph.jsonl"
    settings.stage4.framework_allowed_roots = [tmp_path / "framework"]
    settings.azure_openai.reasoning_deployment = "stage4-test-deployment"
    return settings


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


def _normalized() -> NormalizedTestData:
    document = read_document(
        {
            "manifest": {
                "project": "alpha",
                "schema_version": "1.0",
                "workbook_checksum": "sha256:wb",
                "decision_revision": "r1",
            },
            "records": [
                _record("PROFILE-1", "ExecutionProfile"),
                _record("FIX-1", "Fixture"),
                _record("CLEAN-1", "Cleanup"),
                _record("ORA-1", "Oracle", {"predicate": {"equals": 200}}),
            ],
            "bindings": [
                {
                    "binding_id": "BND-1",
                    "scenario_id": "TS-1",
                    "execution_profile_id": "PROFILE-1",
                    "fixture_id": "FIX-1",
                    "cleanup_id": "CLEAN-1",
                    "oracle_ids": ["ORA-1"],
                    "approval_status": "APPROVED",
                }
            ],
        }
    )
    result = ingest_document(document, scenario_ids={"TS-1"})
    assert result.normalized is not None
    return result.normalized


def _scenario() -> CanonicalTestScenario:
    return CanonicalTestScenario(
        source_req_id=None,
        source_req_id_type="generated",
        scenario_id="TS-1",
        story_ids=["US-1", "US-2"],
        requirement_ids=["REQ-1", "REQ-2"],
        title="Validate Temperature Sensor Threshold",
        description="Validate the approved threshold behavior.",
        scenario_type="Positive",
        priority="High",
        preconditions=["Approved fixture is available"],
        action="Read and validate the sensor threshold",
        expected_result="The threshold is accepted",
        covered_acceptance_criterion_ids=["AC-1"],
        confidence=1.0,
        traceability=Traceability(evidence_chunk_ids=["CHK-1"], entity_ids=[], relationship_ids=[]),
    )


def _snapshot(snapshot_id: str = "FWS-BASE", *, status: str = "ready") -> FrameworkSnapshot:
    return FrameworkSnapshot(
        snapshot_id=snapshot_id,
        repository_id="alpha",
        canonical_path="framework",
        filesystem_checksum=f"sha256:{snapshot_id.lower()}",
        extractor_version="graphify-test",
        extractor_config_hash="sha256:cfg",
        status=status,
        active=status == "ready",
    )


def _plan() -> TestCaseImplementationPlan:
    return TestCaseImplementationPlan(
        scenario_id="TS-1",
        module="sensor",
        test_title="Validate Temperature Sensor Threshold",
        steps=[PlannedStep(step=1, description="Validate threshold", critical=True)],
        files=[
            PlannedFile(role="test", relative_path=PYTHON_PATH),
            PlannedFile(role="robot", relative_path=ROBOT_PATH),
        ],
    )


def _generated(role: str, path: str, content: str) -> GeneratedFileContent:
    digest = "sha256:" + hashlib.sha256(content.encode()).hexdigest()
    return GeneratedFileContent(role=role, relative_path=path, content=content, content_hash=digest)


def _plan_with_helper() -> TestCaseImplementationPlan:
    return _plan().model_copy(
        update={
            "files": [
                *_plan().files,
                PlannedFile(role="helper", relative_path=HELPER_PATH),
            ]
        }
    )


def _bundle(python: str = VALID_PYTHON) -> GeneratedPatchBundle:
    plan = _plan()
    return GeneratedPatchBundle(
        scenario_id="TS-1",
        tc_id=100001,
        plan_checksum=canonical_checksum(plan),
        files=[
            _generated("test", PYTHON_PATH, python),
            _generated("robot", ROBOT_PATH, VALID_ROBOT),
        ],
    )


def _bundle_with_helper() -> GeneratedPatchBundle:
    plan = _plan_with_helper()
    return GeneratedPatchBundle(
        scenario_id="TS-1",
        tc_id=100001,
        plan_checksum=canonical_checksum(plan),
        files=[
            _generated("test", PYTHON_PATH, VALID_PYTHON),
            _generated("robot", ROBOT_PATH, VALID_ROBOT),
            _generated("helper", HELPER_PATH, VALID_HELPER),
        ],
    )


def _runtime(
    tmp_path: Path,
    producer: StaticPatchProducer,
    *,
    verify_failure: bool = False,
    publish_calls: list[list[str]] | None = None,
) -> CodegenRuntime:
    framework = tmp_path / "framework"
    framework.mkdir(exist_ok=True)
    settings = _settings(tmp_path)
    normalized = _normalized()
    scenario = _scenario()
    request = Stage4Request(
        project_name="alpha",
        run_id="RUN-001",
        framework_path=framework,
        test_data_document=tmp_path / "test-data.json",
        execution_profile_id="PROFILE-1",
        reasoning_provider="azure_openai",
    )
    store = CodegenPostgresStore(settings)
    store.ensure_schema()
    logical_hash = make_logical_key_hash(
        project_name="alpha",
        scenario_id="TS-1",
        execution_profile_id="PROFILE-1",
        variant_id="default",
    )
    store.mirror_tc_reservation(
        100001,
        LogicalTestCaseKey(
            project_name="alpha",
            scenario_id="TS-1",
            execution_profile_id="PROFILE-1",
            variant_id="default",
            logical_key_hash=logical_hash,
        ),
    )

    def publish(changed: list[str], _test_data_id: str) -> FrameworkIndexResult:
        if publish_calls is not None:
            publish_calls.append(changed)
        return FrameworkIndexResult(
            snapshot=_snapshot("FWS-NEXT", status="building"),
            file_count=len(changed),
            symbol_count=1,
            edge_count=0,
            dependency_count=0,
            graphify_version="test",
        )

    def verify(_snapshot_id: str, _hashes: dict[str, str]) -> FrameworkSnapshot:
        if verify_failure:
            raise RuntimeError("injected KG verification failure")
        return _snapshot("FWS-NEXT")

    return CodegenRuntime(
        settings=settings,
        request=request,
        scenario=scenario,
        variant_id="default",
        normalized_test_data=normalized,
        framework_snapshot=_snapshot(),
        provider_fingerprint=make_provider_fingerprint(settings, "azure_openai"),
        patch_producer=producer,
        context_retriever=_ContextRetriever(),  # type: ignore[arg-type]
        validation_runner=ValidationRunner(
            framework,
            python_executable=sys.executable,
            robot_dryrun=False,
        ),
        codegen_store=store,
        code_graph_store=CodeGraphStore(settings),
        publish_building=publish,
        verify_snapshot=verify,
    )


def _invoke(runtime: CodegenRuntime) -> dict[str, Any]:
    graph = build_scenario_codegen_graph(runtime)
    return graph.invoke({"scenario_id": "TS-1", "variant_id": "default", "status": "STARTED"})


def _case_transaction(runtime: CodegenRuntime) -> DirectFileTransaction:
    return DirectFileTransaction(
        framework_root=runtime.request.framework_path,
        project=runtime.request.project_name,
        run_id=runtime.request.run_id,
        tc_id=100001,
        journal_root=runtime.settings.paths.generated_dir,
    )


def test_complete_case_is_written_validated_published_persisted_and_sealed(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle()),
        publish_calls=calls,
    )

    final = _invoke(runtime)

    assert final["status"] == "ACCEPTED"
    assert final["tc_id"] == 100001
    assert (runtime.request.framework_path / PYTHON_PATH).is_file()
    assert (runtime.request.framework_path / ROBOT_PATH).is_file()
    assert calls == [[PYTHON_PATH, ROBOT_PATH]]


def test_interrupted_open_journal_is_reconciled_before_regeneration(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle()),
    )
    interrupted = _case_transaction(runtime)
    interrupted.begin()
    interrupted.write(PYTHON_PATH, VALID_PYTHON)
    assert interrupted.journal.status is JournalStatus.OPEN

    final = _invoke(runtime)

    assert final["status"] == "ACCEPTED"
    reconciled = _case_transaction(runtime)
    reconciled.begin()
    assert reconciled.journal.status is JournalStatus.SEALED
    assert reconciled.journal.attempt == 2


def test_accepted_fast_path_seals_a_crash_window_journal(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle()),
    )
    assert _invoke(runtime)["status"] == "ACCEPTED"
    transaction = _case_transaction(runtime)
    transaction.begin()
    transaction.journal.status = JournalStatus.OPEN
    transaction.journal.persist()

    resumed = _invoke(runtime)

    assert resumed["status"] == "ACCEPTED"
    transaction = _case_transaction(runtime)
    transaction.begin()
    assert transaction.journal.status is JournalStatus.SEALED
    record = runtime.codegen_store.get_stage4_test_case(100001)
    assert record is not None and record.status == "ACCEPTED"
    assert record.story_ids == ["US-1", "US-2"]
    assert record.requirement_ids == ["REQ-1", "REQ-2"]
    assert record.context_manifest["context_manifest_id"].startswith("CTX-")
    assert record.validation_evidence["ok"] is True
    assert record.validation_evidence["evidence_checksum"].startswith("sha256:")
    journal = runtime.codegen_store.load_file_journal_metadata("alpha", "RUN-001", 100001)
    assert journal is not None and journal["status"] == "SEALED"


def test_new_test_lib_helper_is_accepted_with_created_file_evidence(tmp_path: Path) -> None:
    plan = _plan_with_helper()
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=plan, bundle=_bundle_with_helper()),
    )

    final = _invoke(runtime)

    assert final["status"] == "ACCEPTED"
    assert (runtime.request.framework_path / HELPER_PATH).read_text(
        encoding="utf-8"
    ) == VALID_HELPER
    record = runtime.codegen_store.get_stage4_test_case(100001)
    assert record is not None
    assert HELPER_PATH in record.helper_files
    assert record.validation_evidence["shared_ast_fingerprints"][HELPER_PATH]

    helper = runtime.request.framework_path / HELPER_PATH
    helper.write_text(
        VALID_HELPER + "\n\ndef helper_added_by_later_case() -> bool:\n    return True\n",
        encoding="utf-8",
    )
    assert _invoke(runtime)["status"] == "ACCEPTED"

    helper.write_text(
        helper.read_text(encoding="utf-8").replace("return bool(value)", "return False"),
        encoding="utf-8",
    )
    assert _invoke(runtime)["status"] == "REVISION_REQUIRED"


def test_validation_exhaustion_rolls_back_only_current_case_and_never_publishes(
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    invalid = _bundle(INVALID_PYTHON)
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(
            implementation_plan=_plan(),
            bundle=invalid,
            repairs=[invalid, invalid],
        ),
        publish_calls=calls,
    )

    final = _invoke(runtime)

    assert final["status"] == "BLOCKED"
    assert final["repair_attempt"] == 2
    assert not (runtime.request.framework_path / PYTHON_PATH).exists()
    assert not (runtime.request.framework_path / ROBOT_PATH).exists()
    assert calls == []
    journal = runtime.codegen_store.load_file_journal_metadata("alpha", "RUN-001", 100001)
    assert journal is not None and journal["status"] == "ROLLED_BACK"


def test_policy_invalid_repair_bundles_consume_budget_then_roll_back(tmp_path: Path) -> None:
    invalid_source = _bundle(INVALID_PYTHON)
    wrong_identity = _bundle().model_copy(update={"tc_id": 100002})
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(
            implementation_plan=_plan(),
            bundle=invalid_source,
            repairs=[wrong_identity, wrong_identity],
        ),
    )

    final = _invoke(runtime)

    assert final["status"] == "BLOCKED"
    assert final["repair_attempt"] == 2
    assert any("REPAIRED_BUNDLE_POLICY_INVALID" in item for item in final["diagnostics"])
    assert not (runtime.request.framework_path / PYTHON_PATH).exists()


def test_exact_data_guard_rejects_an_invented_port(tmp_path: Path) -> None:
    invented = VALID_PYTHON.replace(
        "step_results: list[bool] = []",
        "port = 65000\n        step_results: list[bool] = []",
    )
    invalid = _bundle(invented)
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(
            implementation_plan=_plan(), bundle=invalid, repairs=[invalid, invalid]
        ),
    )

    final = _invoke(runtime)

    assert final["status"] == "BLOCKED"
    assert any("port=65000" in diagnostic for diagnostic in final["diagnostics"])
    assert not (runtime.request.framework_path / PYTHON_PATH).exists()


def test_exact_data_guard_rejects_invented_positional_host_and_port(tmp_path: Path) -> None:
    invented = VALID_PYTHON.replace(
        "step_results: list[bool] = []",
        'connect("invented-host", 65000)\n        step_results: list[bool] = []',
    )
    invalid = _bundle(invented)
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(
            implementation_plan=_plan(), bundle=invalid, repairs=[invalid, invalid]
        ),
    )

    final = _invoke(runtime)

    assert final["status"] == "BLOCKED"
    assert any("connect.arg0='invented-host'" in item for item in final["diagnostics"])
    assert any("connect.arg1=65000" in item for item in final["diagnostics"])


def test_kg_verification_failure_rolls_back_and_stops_the_graph(tmp_path: Path) -> None:
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle()),
        verify_failure=True,
    )

    with pytest.raises(RuntimeError, match="KG verification"):
        _invoke(runtime)

    assert not (runtime.request.framework_path / PYTHON_PATH).exists()
    assert not (runtime.request.framework_path / ROBOT_PATH).exists()
    assert runtime.codegen_store.get_stage4_test_case(100001).status != "ACCEPTED"  # type: ignore[union-attr]


def test_ready_kg_coordination_failure_keeps_files_and_journal_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle()),
    )
    real_save = runtime.codegen_store.save_kg_publication_attempt

    def fail_ready_attempt(**kwargs: Any) -> None:
        if kwargs["status"] == "READY":
            raise RuntimeError("injected READY coordination persistence failure")
        real_save(**kwargs)

    monkeypatch.setattr(runtime.codegen_store, "save_kg_publication_attempt", fail_ready_attempt)

    with pytest.raises(RuntimeError, match="READY coordination"):
        _invoke(runtime)

    assert (runtime.request.framework_path / PYTHON_PATH).is_file()
    assert (runtime.request.framework_path / ROBOT_PATH).is_file()
    transaction = _case_transaction(runtime)
    transaction.begin()
    assert transaction.journal.status is JournalStatus.OPEN


def test_resume_after_kg_publish_finalizes_same_tc_without_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _runtime(
        tmp_path,
        StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle()),
    )
    runtime.checkpointer = InMemorySaver()
    graph = build_scenario_codegen_graph(runtime)
    original_publish = runtime.code_graph_store.publish_test_case
    failures_remaining = 1

    def interrupt_once(record: Any) -> None:
        nonlocal failures_remaining
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("interrupted after KG publish")
        original_publish(record)

    monkeypatch.setattr(runtime.code_graph_store, "publish_test_case", interrupt_once)
    initial = {"scenario_id": "TS-1", "variant_id": "default", "status": "STARTED"}
    config = {"configurable": {"thread_id": "alpha:RUN-001:stage-4:TS-1:default"}}

    with pytest.raises(RuntimeError, match="interrupted after KG publish"):
        graph.invoke(initial, config=config)

    final = graph.invoke(None, config=config)

    assert final["status"] == "ACCEPTED"
    assert final["tc_id"] == 100001
    accepted = runtime.codegen_store.get_stage4_test_case(100001)
    assert accepted is not None and accepted.status == "ACCEPTED"
    transaction = _case_transaction(runtime)
    transaction.begin()
    assert transaction.journal.status is JournalStatus.SEALED
    assert transaction.journal.attempt == 1


def test_changed_accepted_input_returns_revision_required_without_overwrite(
    tmp_path: Path,
) -> None:
    producer = StaticPatchProducer(implementation_plan=_plan(), bundle=_bundle())
    first = _runtime(tmp_path, producer)
    assert _invoke(first)["status"] == "ACCEPTED"
    original_hash = hashlib.sha256(
        (first.request.framework_path / PYTHON_PATH).read_bytes()
    ).hexdigest()

    resumed = _runtime(tmp_path, producer)
    resumed.scenario = resumed.scenario.model_copy(
        update={"description": "Changed frozen scenario input"}
    )
    final = _invoke(resumed)
    replayed = _invoke(resumed)

    assert final["status"] == "REVISION_REQUIRED"
    assert replayed["blocker"]["blocker_id"] == final["blocker"]["blocker_id"]
    assert (
        hashlib.sha256((resumed.request.framework_path / PYTHON_PATH).read_bytes()).hexdigest()
        == original_hash
    )
    record = resumed.codegen_store.get_stage4_test_case(100001)
    assert record is not None and record.status == "ACCEPTED"
