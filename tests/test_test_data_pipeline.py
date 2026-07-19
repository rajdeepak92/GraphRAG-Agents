"""End-to-end deterministic test-data pipeline: read -> validate -> bind -> bundle."""

from __future__ import annotations

from typing import Any

import pytest

from multi_agentic_graph_rag.domain.test_data_schemas import ScenarioDataBinding
from multi_agentic_graph_rag.services.scenario_data_binder import resolve_binding
from multi_agentic_graph_rag.services.test_data_bundle_builder import (
    BundleBuildError,
    build_bundle,
)
from multi_agentic_graph_rag.services.test_data_document_reader import read_document
from multi_agentic_graph_rag.services.test_data_ingestion import ingest_document


def _record(
    record_id: str,
    record_type: str,
    payload: dict[str, Any] | None = None,
    **over: Any,
) -> dict[str, Any]:
    base = {
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
    base.update(over)
    return base


def _binding(**over: Any) -> dict[str, Any]:
    base = {
        "binding_id": "BND-1",
        "scenario_id": "TS-1",
        "execution_profile_id": "PROFILE-1",
        "fixture_id": "FIX-1",
        "cleanup_id": "CLEAN-1",
        "oracle_ids": ["ORA-1"],
        "test_vector_ids": ["VEC-1"],
        "approval_status": "APPROVED",
    }
    base.update(over)
    return base


def _document(records: list[dict[str, Any]], bindings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "manifest": {
            "project": "alpha",
            "schema_version": "1.0",
            "workbook_checksum": "sha256:wb",
            "decision_revision": "r1",
        },
        "records": records,
        "bindings": bindings,
    }


def _golden_records() -> list[dict[str, Any]]:
    return [
        _record("PROFILE-1", "ExecutionProfile", {"environment": "sil"}),
        _record("FIX-1", "Fixture", {"initial_state": "stopped"}),
        _record("CLEAN-1", "Cleanup", {"idempotent": True}),
        _record("ORA-1", "Oracle", {"predicate": {"equals": 200}}),
        _record("VEC-1", "TestVector", {"parameters": {"timeout": 5}}),
    ]


# --- Golden path -------------------------------------------------------------


def test_golden_path_produces_ready_snapshot_and_bundle() -> None:
    document = read_document(_document(_golden_records(), [_binding()]))
    result = ingest_document(document, scenario_ids={"TS-1"})
    assert result.is_ready, result.report.issues
    normalized = result.normalized
    assert normalized is not None
    assert normalized.snapshot_id.startswith("TDS-")

    resolution = resolve_binding(
        scenario_id="TS-1", execution_profile_id="PROFILE-1", bindings=normalized.bindings
    )
    assert resolution.is_ready
    assert resolution.binding is not None

    bundle = build_bundle(binding=resolution.binding, normalized=normalized)
    assert bundle.scenario_id == "TS-1"
    assert bundle.execution_profile.record_id == "PROFILE-1"
    assert [o.record_id for o in bundle.oracles] == ["ORA-1"]
    assert bundle.oracles[0].predicate == {"equals": 200}
    assert [v.record_id for v in bundle.vectors] == ["VEC-1"]
    assert bundle.checksum.startswith("sha256:")


def test_snapshot_id_is_deterministic_for_identical_documents() -> None:
    doc_a = read_document(_document(_golden_records(), [_binding()]))
    doc_b = read_document(_document(_golden_records(), [_binding()]))
    first = ingest_document(doc_a, scenario_ids={"TS-1"})
    second = ingest_document(doc_b, scenario_ids={"TS-1"})
    assert first.normalized is not None and second.normalized is not None
    assert first.normalized.snapshot_id == second.normalized.snapshot_id


# --- Validation gates --------------------------------------------------------


def test_placeholder_value_blocks_snapshot() -> None:
    records = _golden_records()
    records.append(_record("EP-1", "Endpoint", {"address": "TBD"}))
    result = ingest_document(read_document(_document(records, [_binding()])), scenario_ids={"TS-1"})
    assert not result.is_ready
    codes = {issue.issue_code for issue in result.report.issues}
    assert "SEMANTIC_PLACEHOLDER_VALUE" in codes


def test_missing_foreign_key_blocks_snapshot() -> None:
    document = read_document(_document(_golden_records(), [_binding(fixture_id="FIX-GONE")]))
    result = ingest_document(document, scenario_ids={"TS-1"})
    assert not result.is_ready
    assert "REFERENCE_MISSING_FK" in {i.issue_code for i in result.report.issues}


def test_wrong_reference_type_blocks_snapshot() -> None:
    # Point the fixture reference at the Oracle record.
    document = read_document(_document(_golden_records(), [_binding(fixture_id="ORA-1")]))
    result = ingest_document(document, scenario_ids={"TS-1"})
    assert "REFERENCE_WRONG_TYPE" in {i.issue_code for i in result.report.issues}


def test_scenario_without_binding_is_flagged() -> None:
    document = read_document(_document(_golden_records(), [_binding()]))
    result = ingest_document(document, scenario_ids={"TS-1", "TS-2"})
    assert not result.is_ready
    assert "COVERAGE_NO_BINDING" in {i.issue_code for i in result.report.issues}


def test_unapproved_referenced_record_blocks() -> None:
    records = _golden_records()
    records[1] = _record("FIX-1", "Fixture", {"initial_state": "stopped"}, record_status="DRAFT")
    document = read_document(_document(records, [_binding()]))
    result = ingest_document(document, scenario_ids={"TS-1"})
    assert "SEMANTIC_UNAPPROVED_RECORD" in {i.issue_code for i in result.report.issues}


# --- Binder cardinality ------------------------------------------------------


def _model_binding(**over: Any) -> ScenarioDataBinding:
    return ScenarioDataBinding.model_validate(_binding(**over))


def test_binder_blocks_when_no_binding() -> None:
    resolution = resolve_binding(scenario_id="TS-9", execution_profile_id="PROFILE-1", bindings=[])
    assert resolution.status == "BLOCKED_MISSING_DATA"


def test_binder_blocks_on_equal_priority_ambiguity() -> None:
    bindings = [
        _model_binding(binding_id="BND-A", selection_priority=1),
        _model_binding(binding_id="BND-B", selection_priority=1),
    ]
    resolution = resolve_binding(
        scenario_id="TS-1", execution_profile_id="PROFILE-1", bindings=bindings
    )
    assert resolution.status == "BLOCKED_AMBIGUOUS_BINDING"
    assert resolution.candidates == ("BND-A", "BND-B")


def test_binder_uses_priority_tiebreak() -> None:
    bindings = [
        _model_binding(binding_id="BND-A", selection_priority=1),
        _model_binding(binding_id="BND-B", selection_priority=5),
    ]
    resolution = resolve_binding(
        scenario_id="TS-1", execution_profile_id="PROFILE-1", bindings=bindings
    )
    assert resolution.is_ready
    assert resolution.binding is not None and resolution.binding.binding_id == "BND-B"


# --- Bundle builder guards ---------------------------------------------------


def test_bundle_builder_rejects_unapproved_binding() -> None:
    document = read_document(_document(_golden_records(), [_binding()]))
    result = ingest_document(document, scenario_ids={"TS-1"})
    assert result.normalized is not None
    draft_binding = _model_binding(approval_status="DRAFT")
    with pytest.raises(BundleBuildError):
        build_bundle(binding=draft_binding, normalized=result.normalized)
