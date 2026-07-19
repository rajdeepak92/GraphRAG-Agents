"""Optimized Stage-4 domain, settings, and persistence regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings, Stage4Settings
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CodegenBlocker,
    FrozenRunManifest,
    FrozenScenarioEntry,
    LogicalTestCaseKey,
    ProviderFingerprint,
    Stage4Request,
    Stage4TestCaseRecord,
    TcStep,
    TestDataSnapshotRef,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.errors import (
    InputManifestChanged,
    RevisionRequired,
    Stage4PersistenceConflict,
)
from multi_agentic_graph_rag.domain.identifiers import (
    make_logical_key_hash,
    make_provider_fingerprint_hash,
)


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.stage4.codegen_local_path = tmp_path / "codegen.jsonl"
    settings.paths.generated_dir = tmp_path / "generated"
    return settings


def _key() -> LogicalTestCaseKey:
    fields = {
        "project_name": "Demo Project",
        "scenario_id": "TS-1",
        "execution_profile_id": "EP-DEFAULT",
        "variant_id": "default",
    }
    return LogicalTestCaseKey(
        **fields,
        logical_key_hash=make_logical_key_hash(**fields),
    )


def _provider(model: str = "azure-deployment") -> ProviderFingerprint:
    params = {"temperature": 0.0, "seed": 7}
    return ProviderFingerprint(
        provider="azure_openai",
        model=model,
        generation_params=params,
        prompt_revision="stage4-v1",
        fingerprint_hash=make_provider_fingerprint_hash(
            provider="azure_openai",
            model=model,
            model_revision=None,
            generation_params_checksum=canonical_checksum({"params": params}),
            prompt_revision="stage4-v1",
        ),
    )


def _manifest(*, baseline: str = "sha256:framework") -> FrozenRunManifest:
    draft = FrozenRunManifest.model_construct(
        project_name="Demo Project",
        run_id="RUN-001",
        scenarios=[
            FrozenScenarioEntry(
                scenario_id="TS-1",
                scenario_checksum="sha256:scenario",
                story_ids=["US-1", "US-2"],
                requirement_ids=["REQ-1", "REQ-2"],
            )
        ],
        execution_profile_id="EP-DEFAULT",
        test_data_snapshot_checksum="sha256:test-data",
        baseline_framework_checksum=baseline,
        provider_fingerprint=_provider(),
        generation_policy_version="policy-v1",
        checksum="",
    )
    return FrozenRunManifest.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


def _accepted_record() -> Stage4TestCaseRecord:
    key = _key()
    digest = "sha256:" + "a" * 64
    return Stage4TestCaseRecord(
        tc_id=100001,
        **key.model_dump(),
        status="ACCEPTED",
        story_ids=["US-1", "US-2"],
        requirement_ids=["REQ-1", "REQ-2"],
        tc_steps=[TcStep(step=1, description="Validate threshold")],
        module="sensor",
        test_file="tests/sensor/Tc100001ValidateThreshold.py",
        robot_file="tests_robot/sensor/Tc100001ValidateThreshold.robot",
        helper_files=["test_lib/sensor/sensor_wrappers.py"],
        generated_file_hashes={
            "tests/sensor/Tc100001ValidateThreshold.py": digest,
            "tests_robot/sensor/Tc100001ValidateThreshold.robot": digest,
            "test_lib/sensor/sensor_wrappers.py": digest,
        },
        framework_snapshot_id="FWS-1",
        test_data_snapshot_id="TDS-1",
        provider_fingerprint=_provider(),
        prompt_revision="stage4-v1",
        input_fingerprint="sha256:input",
    )


def test_stage4_request_and_settings_are_no_git_by_construction(tmp_path: Path) -> None:
    request = Stage4Request(
        project_name="demo",
        run_id="RUN-001",
        framework_path=tmp_path / "framework",
        test_data_document=tmp_path / "data.xlsx",
        execution_profile_id="EP-DEFAULT",
    )
    assert request.reasoning_provider == "azure_openai"
    assert request.variant_selection == {}

    settings = Stage4Settings()
    assert settings.rollback_failed_case is True
    assert settings.robot_dryrun is True
    assert settings.model_transport_retries == 1
    assert settings.reindex_after_each_case is True
    assert settings.retain_referenced_snapshots is True
    assert settings.graphify_command == "graphify"
    assert settings.graphify_no_cluster is True
    assert settings.reasoning_provider == "azure_openai"
    assert "worktrees_dir" not in Stage4Settings.model_fields
    for locked_field in (
        "rollback_failed_case",
        "robot_dryrun",
        "reindex_after_each_case",
        "retain_referenced_snapshots",
    ):
        with pytest.raises(ValidationError):
            Stage4Settings.model_validate({locked_field: False})

    with pytest.raises(ValidationError):
        settings.robot_dryrun = False  # type: ignore[assignment]

    with pytest.raises(ValidationError):
        request.model_copy(update={"reasoning_provider": "unknown"}).model_validate(
            {**request.model_dump(), "reasoning_provider": "unknown"}
        )


def test_config_loader_populates_optimized_stage4_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "stage4": {
                    "reasoning_provider": "huggingface",
                    "graphify_command": "graphify-custom",
                    "symbol_search_limit": 12,
                    "max_context_symbols": 44,
                    "write_allowlist": ["tests/<module>/Tc<id><PascalTitle>.py"],
                },
                "huggingface": {"model_revision": "revision-123"},
                "azure_openai": {"log_llm_responses": True},
            }
        ),
        encoding="utf-8",
    )
    settings = load_config(config_path)
    assert settings.stage4.reasoning_provider == "huggingface"
    assert settings.stage4.graphify_command == "graphify-custom"
    assert settings.stage4.symbol_search_limit == 12
    assert settings.stage4.max_context_symbols == 44
    assert settings.huggingface.model_revision == "revision-123"
    assert settings.azure_openai.log_llm_responses is True


def test_config_loading_does_not_create_runtime_directories(tmp_path: Path) -> None:
    project_root = tmp_path / "empty-project"
    project_root.mkdir()

    load_config(project_root / "missing-config.json")

    assert list(project_root.iterdir()) == []


def test_accepted_record_requires_complete_consistent_identity() -> None:
    key = _key()
    with pytest.raises(ValidationError, match="ACCEPTED test case is incomplete"):
        Stage4TestCaseRecord(
            tc_id=100001,
            **key.model_dump(),
            status="ACCEPTED",
        )
    with pytest.raises(ValidationError, match="logical_key_hash"):
        Stage4TestCaseRecord(
            tc_id=100001,
            project_name=key.project_name,
            scenario_id=key.scenario_id,
            execution_profile_id=key.execution_profile_id,
            variant_id=key.variant_id,
            logical_key_hash="LKH-wrong",
        )
    assert _accepted_record().tc_steps[0].step == 1


def test_frozen_manifest_is_saved_once_and_compared_on_resume(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    original = _manifest()
    assert store.save_frozen_manifest(original) is True
    assert store.save_or_validate_frozen_manifest(original) is False
    assert store.load_frozen_manifest("Demo Project", "RUN-001") == original
    with pytest.raises(InputManifestChanged):
        store.save_frozen_manifest(_manifest(baseline="sha256:changed"))


def test_accepted_case_is_immutable_and_artifact_is_run_scoped(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    record = _accepted_record()
    store.mirror_tc_reservation(record.tc_id, _key())
    assert store.save_stage4_test_case(record, run_id="RUN-001") is True
    assert store.save_stage4_test_case(record, run_id="RUN-001") is False
    changed = record.model_copy(update={"input_fingerprint": "sha256:changed"})
    with pytest.raises(RevisionRequired):
        store.save_stage4_test_case(changed, run_id="RUN-001")

    store.save_blocker(
        "Demo Project",
        CodegenBlocker(
            blocker_id="BLK-1",
            scenario_id="TS-2",
            blocker_type="BLOCKED_MISSING_DATA",
            evidence="missing exact values",
        ),
        run_id="RUN-001",
    )
    artifact_path = store.write_test_cases_artifact("Demo Project", "RUN-001")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact_path == (
        tmp_path / "generated" / "demo-project" / "RUN-001" / "test-cases" / "test_cases.json"
    )
    assert [item["tc_id"] for item in payload["test_cases"]] == [100001]
    assert [item["blocker_id"] for item in payload["blockers"]] == ["BLK-1"]


def test_test_data_and_coordination_records_round_trip(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    snapshot = TestDataSnapshotRef(
        snapshot_id="TDS-1",
        project="Demo Project",
        schema_version="1.0",
        workbook_checksum="sha256:workbook",
        normalized_checksum="sha256:normalized",
        decision_revision="r1",
        status="ready",
    )
    normalized = {"records": {"ENDPOINT-1": {"host": "192.0.2.1", "port": 443}}}
    assert store.save_test_data_snapshot(snapshot, normalized) is True
    assert store.save_test_data_snapshot(snapshot, normalized) is False
    assert store.load_test_data_snapshot("TDS-1") == (snapshot, normalized)
    with pytest.raises(Stage4PersistenceConflict):
        store.save_test_data_snapshot(snapshot, {"records": {"invented": True}})

    store.save_file_journal_metadata(
        project_name="Demo Project",
        run_id="RUN-001",
        tc_id=100001,
        status="OPEN",
        journal_path="generated/demo/RUN-001/stage-4/journals/100001",
        payload={"entries": []},
    )
    journal = store.load_file_journal_metadata("Demo Project", "RUN-001", 100001)
    assert journal is not None and journal["status"] == "OPEN"

    store.save_kg_publication_attempt(
        attempt_id="KG-1",
        project_name="Demo Project",
        run_id="RUN-001",
        tc_id=100001,
        status="BUILDING",
        previous_snapshot_id="FWS-0",
        proposed_snapshot_id="FWS-1",
        payload={"changed_files": ["tests/sensor/Tc100001ValidateThreshold.py"]},
    )
    attempt = store.load_kg_publication_attempt("KG-1")
    assert attempt is not None and attempt["proposed_snapshot_id"] == "FWS-1"

    key, created = store.save_idempotency_record(
        project_name="Demo Project",
        run_id="RUN-001",
        tc_id=100001,
        node_name="persist_case",
        input_checksum="sha256:input",
        result_payload={"status": "ACCEPTED"},
    )
    assert created is True
    assert store.save_idempotency_record(
        project_name="Demo Project",
        run_id="RUN-001",
        tc_id=100001,
        node_name="persist_case",
        input_checksum="sha256:input",
        result_payload={"status": "ACCEPTED"},
    ) == (key, False)


class _SequenceCursor:
    def __init__(self, row: tuple[str, ...]) -> None:
        self.row = row

    def execute(self, _statement: str) -> None:
        return None

    def fetchone(self) -> tuple[str, ...]:
        return self.row


def test_sequence_contract_includes_increment_and_cache(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    valid = _SequenceCursor(("integer", "100001", "100001", "999999", "1", "NO", "1"))
    store._verify_sequence_contract(valid)
    invalid = _SequenceCursor(("integer", "100001", "100001", "999999", "2", "NO", "10"))
    with pytest.raises(RuntimeError, match="contract violation"):
        store._verify_sequence_contract(invalid)
