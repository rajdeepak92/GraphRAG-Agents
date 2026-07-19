"""Contract and identity regression tests for Stage 4 (test-code generation)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from multi_agentic_graph_rag.domain.codegen_schemas import (
    CapabilityBinding,
    CodegenBlocker,
    ContextManifest,
    ContextManifestItem,
    OracleRef,
    RecordRef,
    ResolvedScenarioTestDataBundle,
    TestCaseRecord,
    TestCasesArtifact,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.identifiers import (
    make_framework_snapshot_id,
    make_resolved_bundle_id,
    make_scenario_data_binding_id,
    make_test_data_snapshot_id,
    new_codegen_run_id,
    new_test_case_id,
)


def _checksummed(model_cls: type[Any], **fields: Any) -> Any:
    """Build a checksum-validated model the way the Stage builders do."""
    payload = model_cls.model_construct(checksum="", **fields)
    return model_cls.model_validate(
        {**payload.model_dump(mode="json"), "checksum": canonical_checksum(payload)}
    )


def _bundle(**overrides: Any) -> ResolvedScenarioTestDataBundle:
    fields: dict[str, Any] = {
        "bundle_id": "TDB-1",
        "scenario_id": "TS-1",
        "binding_id": "BND-1",
        "snapshot_id": "TDS-1",
        "execution_profile": RecordRef(record_id="PROFILE-1"),
        "fixture": RecordRef(record_id="FIXTURE-1"),
        "oracles": [OracleRef(record_id="ORACLE-1", predicate={"equals": 200})],
        "cleanup": RecordRef(record_id="CLEANUP-1"),
    }
    fields.update(overrides)
    return _checksummed(ResolvedScenarioTestDataBundle, **fields)


# --- Identities --------------------------------------------------------------


def test_snapshot_ids_are_deterministic_and_shareable() -> None:
    kwargs = {
        "repository_id": "svc",
        "tree_hash": "abc",
        "dirty_hash": "clean",
        "extractor_version": "graphify-8",
        "extractor_config_hash": "cfg",
    }
    first = make_framework_snapshot_id(**kwargs)
    assert first == make_framework_snapshot_id(**kwargs)
    assert first.startswith("FWS-")
    assert first != make_framework_snapshot_id(**{**kwargs, "tree_hash": "def"})


def test_test_data_snapshot_id_is_project_scoped() -> None:
    kwargs = {
        "workbook_checksum": "wb",
        "normalized_checksum": "nm",
        "schema_version": "1.0",
        "decision_revision": "r1",
    }
    assert make_test_data_snapshot_id(project="alpha", **kwargs) != make_test_data_snapshot_id(
        project="beta", **kwargs
    )


def test_binding_and_bundle_ids_are_deterministic() -> None:
    binding = make_scenario_data_binding_id(
        scenario_id="TS-1", execution_profile_id="PROFILE-1", test_data_snapshot_id="TDS-1"
    )
    assert binding.startswith("BND-")
    bundle = make_resolved_bundle_id(binding_id=binding, test_data_snapshot_id="TDS-1")
    assert bundle == make_resolved_bundle_id(binding_id=binding, test_data_snapshot_id="TDS-1")
    assert bundle.startswith("TDB-")


def test_operational_ids_are_unique_and_prefixed() -> None:
    assert new_codegen_run_id("alpha") != new_codegen_run_id("alpha")
    assert new_codegen_run_id("alpha").startswith("CGR-")
    assert new_test_case_id() != new_test_case_id()
    assert new_test_case_id().startswith("TC-")


# --- Bundle contract ---------------------------------------------------------


def test_bundle_checksum_is_validated() -> None:
    bundle = _bundle()
    assert bundle.checksum == canonical_checksum(bundle)
    with pytest.raises(ValidationError):
        bundle.checksum = "sha256:tampered"


def test_bundle_rejects_plaintext_secrets() -> None:
    with pytest.raises(ValidationError):
        _bundle(secret_refs=["hunter2"])
    # A proper reference is accepted.
    assert _bundle(secret_refs=["secret://vault/db"]).secret_refs == ["secret://vault/db"]


def test_bundle_requires_at_least_one_oracle() -> None:
    with pytest.raises(ValidationError):
        _bundle(oracles=[])


# --- Context manifest --------------------------------------------------------


def test_context_manifest_checksum_and_positive_budget() -> None:
    item = ContextManifestItem(
        source_type="code_symbol",
        source_id="SYM-1",
        revision="FWS-1",
        content_hash="sha256:aa",
        retrieval_reason="implements start",
        authoritative_source="repository",
    )
    manifest = _checksummed(
        ContextManifest,
        manifest_id="CTX-1",
        scenario_id="TS-1",
        framework_snapshot_id="FWS-1",
        test_data_snapshot_id="TDS-1",
        worktree_tree_hash="tree",
        items=[item],
        token_budget=24000,
    )
    assert manifest.checksum == canonical_checksum(manifest)
    with pytest.raises(ValidationError):
        manifest.token_budget = 0


# --- Capability binding ------------------------------------------------------


def test_reuse_decision_requires_selected_symbol() -> None:
    with pytest.raises(ValidationError):
        CapabilityBinding(
            binding_id="CAP-1",
            scenario_id="TS-1",
            scenario_action="Start a simulated slave",
            capability_id="simulator.lifecycle.start",
            decision="wrap",
            reason="async vs sync",
        )
    ok = CapabilityBinding(
        binding_id="CAP-1",
        scenario_id="TS-1",
        scenario_action="Start a simulated slave",
        capability_id="simulator.lifecycle.start",
        decision="wrap",
        selected_symbol="framework.simulator.SlaveServer.start",
        adapter_to_create="tests/support/ssb_simulator.py::start",
        reason="async vs sync",
    )
    assert ok.selected_symbol is not None


def test_implement_decision_needs_no_symbol() -> None:
    binding = CapabilityBinding(
        binding_id="CAP-2",
        scenario_id="TS-1",
        scenario_action="Assert register value",
        capability_id="assert.register",
        decision="implement",
        reason="no existing helper",
    )
    assert binding.selected_symbol is None


# --- Blockers ----------------------------------------------------------------


def test_blocker_must_carry_a_blocking_status() -> None:
    with pytest.raises(ValidationError):
        CodegenBlocker(
            blocker_id="BLK-1",
            scenario_id="TS-1",
            blocker_type="READY",
            evidence="n/a",
        )
    blocker = CodegenBlocker(
        blocker_id="BLK-1",
        scenario_id="TS-1",
        blocker_type="BLOCKED_MISSING_DATA",
        evidence="no approved binding for scenario",
    )
    assert blocker.blocker_type == "BLOCKED_MISSING_DATA"


# --- Test-case master --------------------------------------------------------


def _test_case(**overrides: Any) -> TestCaseRecord:
    fields: dict[str, Any] = {
        "test_case_id": "TC-1",
        "test_case_revision": 1,
        "scenario_id": "TS-1",
        "requirement_ids": ["REQ-1"],
        "test_name": "test_start_slave",
        "test_file": "tests/test_slave.py",
        "test_function": "test_start_slave",
        "execution_profile_id": "PROFILE-1",
        "scenario_data_binding_id": "BND-1",
        "resolved_test_data_bundle_id": "TDB-1",
        "resolved_test_data_bundle_checksum": "sha256:aa",
        "priority": "High",
        "validation_status": "EXECUTABLE",
        "context_manifest_id": "CTX-1",
        "generation_model": "claude-opus-4-8",
        "prompt_revision": "stage4-v1",
    }
    fields.update(overrides)
    return TestCaseRecord(**fields)


def test_test_case_rejects_unknown_fields_and_bad_revision() -> None:
    with pytest.raises(ValidationError):
        _test_case(unexpected="x")
    with pytest.raises(ValidationError):
        _test_case(test_case_revision=0)


def test_test_cases_artifact_checksum_roundtrip() -> None:
    artifact = _checksummed(
        TestCasesArtifact,
        project="alpha",
        run_id="RUN-1",
        test_cases=[_test_case()],
    )
    assert artifact.artifact_schema_version == "1.0-test-cases"
    assert artifact.checksum == canonical_checksum(artifact)
    with pytest.raises(ValidationError):
        artifact.checksum = "sha256:tampered"
