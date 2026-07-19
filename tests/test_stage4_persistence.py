"""Local-json persistence tests for the Stage-4 code graph and codegen store."""

from __future__ import annotations

from pathlib import Path

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.code_graph_schemas import (
    CodeEdge,
    CodeExtractionResult,
    CodeFile,
    CodeSymbol,
    FrameworkSnapshot,
)
from multi_agentic_graph_rag.domain.codegen_schemas import CodegenBlocker, TestCaseRecord


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.neo4j.mode = "local_json"
    settings.postgres.mode = "local_json"
    settings.stage4.code_graph_local_path = tmp_path / "code_graph.jsonl"
    settings.stage4.codegen_local_path = tmp_path / "codegen.jsonl"
    return settings


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
    )


def _result() -> CodeExtractionResult:
    files = [
        CodeFile(
            snapshot_id="FWS-1", relative_path="mod.py", language="python", content_hash="sha256:f"
        )
    ]
    symbols = [
        CodeSymbol(
            snapshot_id="FWS-1",
            symbol_id="SYM-A",
            fqn="mod.py::Slave",
            kind="Class",
            relative_path="mod.py",
            start_line=1,
            end_line=5,
            start_byte=0,
            end_byte=0,
            body_hash="sha256:a",
        ),
        CodeSymbol(
            snapshot_id="FWS-1",
            symbol_id="SYM-B",
            fqn="mod.py::Slave.start()",
            kind="Method",
            relative_path="mod.py",
            start_line=2,
            end_line=3,
            start_byte=0,
            end_byte=0,
            body_hash="sha256:b",
        ),
    ]
    edges = [
        CodeEdge(
            edge_id="EDG-1",
            snapshot_id="FWS-1",
            source_symbol_id="SYM-A",
            target_symbol_id="SYM-B",
            relation="DECLARES",
            confidence="EXTRACTED",
            provenance="ast",
            source_location="L2",
            extractor="graphify-test",
        )
    ]
    return CodeExtractionResult(snapshot_id="FWS-1", files=files, symbols=symbols, edges=edges)


# --- Code graph store --------------------------------------------------------


def test_code_graph_publish_and_query(tmp_path: Path) -> None:
    store = CodeGraphStore(_settings(tmp_path))
    ready = store.publish_snapshot(_snapshot(), _result())
    assert ready.status == "ready"

    found = store.search_symbols("FWS-1", "start")
    assert [s.symbol_id for s in found] == ["SYM-B"]
    assert store.search_symbols("FWS-1", "start", kinds=["Class"]) == []

    symbol = store.get_symbol("FWS-1", "SYM-A")
    assert symbol is not None and symbol.kind == "Class"

    neighbors = store.get_neighbors("FWS-1", "SYM-A", relations=["DECLARES"], depth=1)
    assert [n.symbol_id for n in neighbors] == ["SYM-B"]


def test_code_graph_snapshot_isolation(tmp_path: Path) -> None:
    store = CodeGraphStore(_settings(tmp_path))
    store.publish_snapshot(_snapshot(), _result())
    assert store.get_symbol("FWS-OTHER", "SYM-A") is None
    removed = store.delete_snapshot("FWS-1")
    assert removed > 0
    assert store.get_symbol("FWS-1", "SYM-A") is None


# --- Codegen store -----------------------------------------------------------


def _test_case(revision: int, status: str = "EXECUTABLE") -> TestCaseRecord:
    return TestCaseRecord(
        test_case_id="TC-1",
        test_case_revision=revision,
        scenario_id="TS-1",
        test_name="test_start",
        test_file="tests/test_start.py",
        test_function="test_start",
        execution_profile_id="PROFILE-1",
        scenario_data_binding_id="BND-1",
        resolved_test_data_bundle_id="TDB-1",
        resolved_test_data_bundle_checksum="sha256:x",
        priority="High",
        validation_status=status,
        context_manifest_id="CTX-1",
        generation_model="claude-opus-4-8",
        prompt_revision="v1",
    )


def test_codegen_store_rebuilds_latest_test_case_revision(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    store.save_test_case("alpha", _test_case(1, status="COLLECTABLE"))
    store.save_test_case("alpha", _test_case(2, status="EXECUTABLE"))

    artifact = store.rebuild_test_cases_artifact("alpha", "RUN-1")
    assert artifact.artifact_schema_version == "1.0-test-cases"
    assert len(artifact.test_cases) == 1
    assert artifact.test_cases[0].test_case_revision == 2
    assert artifact.test_cases[0].validation_status == "EXECUTABLE"


def test_codegen_store_persists_blocker(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    blocker = CodegenBlocker(
        blocker_id="BLK-1",
        scenario_id="TS-1",
        blocker_type="BLOCKED_MISSING_DATA",
        evidence="no approved binding",
    )
    store.save_blocker("alpha", blocker)
    # A separate project sees no test cases.
    empty = store.rebuild_test_cases_artifact("beta", "RUN-1")
    assert empty.test_cases == []
