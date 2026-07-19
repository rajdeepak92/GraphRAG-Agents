"""P0 tests: six-digit TC identity, logical-key uniqueness, provider pinning.

These exercise the deterministic Stage-4 foundation with the ``local_json``
fallback so no live PostgreSQL server is required. They verify allocation
boundaries, idempotent reuse, exhaustion, and the provider fingerprint pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.codegen_schemas import (
    FrozenRunManifest,
    FrozenScenarioEntry,
    LogicalTestCaseKey,
    ProviderFingerprint,
    Stage4TestCaseRecord,
    TcStep,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.domain.identifiers import (
    TC_ID_MAX,
    TC_ID_MIN,
    make_logical_key_hash,
    make_provider_fingerprint_hash,
    pascal_title,
    tc_stem,
)


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.neo4j.mode = "local_json"
    settings.stage4.codegen_local_path = tmp_path / "codegen.jsonl"
    return settings


def _logical_key(scenario_id: str, *, variant: str = "default") -> LogicalTestCaseKey:
    return LogicalTestCaseKey(
        project_name="Demo Project",
        scenario_id=scenario_id,
        execution_profile_id="EP-DEFAULT",
        variant_id=variant,
        logical_key_hash=make_logical_key_hash(
            project_name="Demo Project",
            scenario_id=scenario_id,
            execution_profile_id="EP-DEFAULT",
            variant_id=variant,
        ),
    )


# --- naming ------------------------------------------------------------------


def test_pascal_title_and_tc_stem() -> None:
    assert pascal_title("Validate Temperature Sensor Threshold") == (
        "ValidateTemperatureSensorThreshold"
    )
    assert tc_stem(100001, "Validate Temperature Sensor Threshold") == (
        "Tc100001ValidateTemperatureSensorThreshold"
    )
    # Leading-digit titles stay legal Python identifiers.
    assert pascal_title("3 phase power")[0].isalpha()


def test_tc_stem_rejects_out_of_range_ids() -> None:
    with pytest.raises(ValueError):
        tc_stem(TC_ID_MIN - 1, "Too Small")
    with pytest.raises(ValueError):
        tc_stem(TC_ID_MAX + 1, "Too Big")


def test_logical_key_hash_validates_fields() -> None:
    key = _logical_key("TS-1")
    assert key.logical_key_hash.startswith("LKH-")
    with pytest.raises(ValueError):
        LogicalTestCaseKey(
            project_name="Demo Project",
            scenario_id="TS-1",
            execution_profile_id="EP-DEFAULT",
            variant_id="default",
            logical_key_hash="LKH-wrong",
        )


# --- allocator ---------------------------------------------------------------


def test_allocator_is_idempotent_per_logical_key(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    store.mirror_tc_reservation(TC_ID_MIN, _logical_key("TS-1"))

    first, created_first = store.allocate_or_get_tc_id(_logical_key("TS-1"))
    assert first == TC_ID_MIN
    assert created_first is False

    again, created_again = store.allocate_or_get_tc_id(_logical_key("TS-1"))
    assert again == first
    assert created_again is False


def test_local_json_never_invents_an_id(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    with pytest.raises(ConfigurationError, match="PostgreSQL is authoritative"):
        store.allocate_or_get_tc_id(_logical_key("TS-NEW"))


def test_mirrored_ids_preserve_variants_without_renumbering(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    default_key = _logical_key("TS-1")
    hot_key = _logical_key("TS-1", variant="hot")
    store.mirror_tc_reservation(TC_ID_MIN, default_key)
    store.mirror_tc_reservation(TC_ID_MAX, hot_key)
    assert store.allocate_or_get_tc_id(default_key) == (TC_ID_MIN, False)
    assert store.allocate_or_get_tc_id(hot_key) == (TC_ID_MAX, False)


# --- provider pinning --------------------------------------------------------


def _fingerprint(model: str = "gpt-4o-deploy") -> ProviderFingerprint:
    params = {"temperature": 0.0}
    return ProviderFingerprint(
        provider="azure_openai",
        model=model,
        model_revision=None,
        generation_params=params,
        prompt_revision="stage4-plan-v1",
        fingerprint_hash=make_provider_fingerprint_hash(
            provider="azure_openai",
            model=model,
            model_revision=None,
            generation_params_checksum=canonical_checksum({"params": params}),
            prompt_revision="stage4-plan-v1",
        ),
    )


def test_provider_fingerprint_validates_and_matches() -> None:
    pin = _fingerprint()
    assert pin.matches(_fingerprint())
    assert not pin.matches(_fingerprint(model="other-deploy"))
    with pytest.raises(ValueError):
        ProviderFingerprint(
            provider="azure_openai",
            model="gpt-4o-deploy",
            prompt_revision="stage4-plan-v1",
            fingerprint_hash="PFP-wrong",
        )


def test_frozen_run_manifest_checksum_roundtrip() -> None:
    scenarios = [
        FrozenScenarioEntry(
            scenario_id="TS-1",
            scenario_checksum="sha256:a",
            story_ids=["US-1", "US-2"],
            requirement_ids=["REQ-1", "REQ-2"],
        )
    ]
    draft = FrozenRunManifest.model_construct(
        project_name="Demo Project",
        run_id="RUN-1",
        scenarios=scenarios,
        execution_profile_id="EP-DEFAULT",
        test_data_snapshot_checksum="sha256:td",
        baseline_framework_checksum="sha256:fw",
        provider_fingerprint=_fingerprint(),
        generation_policy_version="policy-1",
        checksum="",
    )
    manifest = FrozenRunManifest.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )
    # All story and requirement IDs are preserved (not just the first).
    assert manifest.scenarios[0].story_ids == ["US-1", "US-2"]
    assert manifest.scenarios[0].requirement_ids == ["REQ-1", "REQ-2"]


def test_stage4_test_case_persist_roundtrip(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    key = _logical_key("TS-1")
    store.mirror_tc_reservation(TC_ID_MIN, key)
    tc_id, _ = store.allocate_or_get_tc_id(key)
    digest = "sha256:" + "a" * 64
    record = Stage4TestCaseRecord(
        tc_id=tc_id,
        project_name=key.project_name,
        scenario_id=key.scenario_id,
        execution_profile_id=key.execution_profile_id,
        variant_id=key.variant_id,
        logical_key_hash=key.logical_key_hash,
        status="ACCEPTED",
        story_ids=["US-1"],
        requirement_ids=["REQ-1"],
        tc_steps=[TcStep(step=1, description="Validate sensor")],
        module="sensor",
        test_file="tests/sensor/Tc100001ValidateSensor.py",
        robot_file="tests_robot/sensor/Tc100001ValidateSensor.robot",
        generated_file_hashes={
            "tests/sensor/Tc100001ValidateSensor.py": digest,
            "tests_robot/sensor/Tc100001ValidateSensor.robot": digest,
        },
        framework_snapshot_id="FWS-1",
        test_data_snapshot_id="TDS-1",
        provider_fingerprint=_fingerprint(),
        prompt_revision="stage4-plan-v1",
        input_fingerprint="sha256:input",
    )
    assert store.save_stage4_test_case(record, run_id="RUN-1") is True
    assert store.save_stage4_test_case(record, run_id="RUN-1") is False
    loaded = store.get_stage4_test_case(tc_id)
    assert loaded is not None
    assert loaded.status == "ACCEPTED"
    assert loaded.story_ids == ["US-1"]
