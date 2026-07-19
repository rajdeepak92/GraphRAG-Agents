"""Deterministic Stage-4 workflow spine: framework indexing + scenario planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData
from multi_agentic_graph_rag.services.capability_mapper import CapabilityCandidate
from multi_agentic_graph_rag.services.framework_indexer import index_framework
from multi_agentic_graph_rag.services.test_data_document_reader import read_document
from multi_agentic_graph_rag.services.test_data_ingestion import ingest_document
from multi_agentic_graph_rag.workflows.test_code_generation_graph import (
    ScenarioAction,
    ScenarioPlanRequest,
    plan_scenario,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.neo4j.mode = "local_json"
    settings.postgres.mode = "local_json"
    settings.stage4.code_graph_local_path = tmp_path / "code_graph.jsonl"
    settings.stage4.framework_allowed_roots = [_REPO_ROOT]
    return settings


# --- Framework indexing (real repository) ------------------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / "graphify-out" / ".graphify_extract.json").exists(),
    reason="graphify extraction not present",
)
def test_index_framework_publishes_real_code_graph(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = index_framework(
        settings=settings,
        framework_path=_REPO_ROOT,
        graphify_out_dir=_REPO_ROOT / "graphify-out",
        repository_id="self",
    )
    assert result.snapshot.status == "ready"
    assert result.symbol_count > 0
    assert result.file_count > 0

    store = CodeGraphStore(settings)
    hits = store.search_symbols(result.snapshot.snapshot_id, "canonical_checksum")
    assert hits, "expected the indexed graph to be queryable by symbol name"


# --- Scenario planning (dry-run) ---------------------------------------------


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


def _snapshot() -> FrameworkSnapshot:
    return FrameworkSnapshot(
        snapshot_id="FWS-1",
        repository_id="repo",
        canonical_path="/repo",
        branch="main",
        commit="c" * 40,
        tree_hash="t" * 40,
        dirty=False,
        dirty_hash="clean",
        extractor_version="graphify-test",
        extractor_config_hash="cfg",
        status="ready",
    )


def test_plan_scenario_ready_path() -> None:
    normalized = _normalized()
    request = ScenarioPlanRequest(
        scenario_id="TS-1",
        execution_profile_id="PROFILE-1",
        codegen_run_id="CGR-1",
        framework_snapshot=_snapshot(),
        normalized=normalized,
        actions=[
            ScenarioAction(
                action_text="Assert response is 200",
                capability_id="assert.status",
                candidates=[CapabilityCandidate("SYM-A", "fw.assert_status", "exact")],
            )
        ],
    )
    plan = plan_scenario(request)
    assert plan.is_ready, plan.readiness.evidence
    assert plan.resolved_bundle_id is not None
    assert plan.capability_bindings[0].decision == "reuse"
    assert plan.context_manifest_id is not None


def test_plan_scenario_blocks_on_missing_binding() -> None:
    normalized = _normalized()
    request = ScenarioPlanRequest(
        scenario_id="TS-UNBOUND",
        execution_profile_id="PROFILE-1",
        codegen_run_id="CGR-1",
        framework_snapshot=_snapshot(),
        normalized=normalized,
        actions=[],
    )
    plan = plan_scenario(request)
    assert not plan.is_ready
    assert plan.readiness.status == "BLOCKED_MISSING_DATA"
    assert plan.binding_status == "BLOCKED_MISSING_DATA"


def test_plan_scenario_blocks_on_capability_gap() -> None:
    normalized = _normalized()
    request = ScenarioPlanRequest(
        scenario_id="TS-1",
        execution_profile_id="PROFILE-1",
        codegen_run_id="CGR-1",
        framework_snapshot=_snapshot(),
        normalized=normalized,
        actions=[
            ScenarioAction(action_text="Do novel thing", capability_id="cap.novel", candidates=[])
        ],
        modification_authorized=False,
    )
    plan = plan_scenario(request)
    assert plan.readiness.status == "BLOCKED_MISSING_CAPABILITY"
    assert plan.unresolved_capabilities == ["cap.novel"]
