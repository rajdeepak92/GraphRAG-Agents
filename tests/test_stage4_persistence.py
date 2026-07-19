"""Local-json persistence tests for the Stage-4 code graph and codegen store."""

from __future__ import annotations

from pathlib import Path

import pytest

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
from multi_agentic_graph_rag.domain.codegen_schemas import (
    CodegenBlocker,
    LogicalTestCaseKey,
    ProviderFingerprint,
    Stage4TestCaseRecord,
    TcStep,
    canonical_checksum,
)
from multi_agentic_graph_rag.domain.identifiers import (
    make_logical_key_hash,
    make_provider_fingerprint_hash,
)


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
        filesystem_checksum="sha256:" + "c" * 64,
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
    with pytest.raises(ValueError, match="active READY snapshot"):
        store.delete_snapshot("FWS-1")
    assert store.get_symbol("FWS-1", "SYM-A") is not None


# --- Codegen store -----------------------------------------------------------


def _logical_key() -> LogicalTestCaseKey:
    logical_key_hash = make_logical_key_hash(
        project_name="alpha",
        scenario_id="TS-1",
        execution_profile_id="PROFILE-1",
        variant_id="default",
    )
    return LogicalTestCaseKey(
        project_name="alpha",
        scenario_id="TS-1",
        execution_profile_id="PROFILE-1",
        variant_id="default",
        logical_key_hash=logical_key_hash,
    )


def _test_case() -> Stage4TestCaseRecord:
    key = _logical_key()
    parameters = {"temperature": 0.0}
    fingerprint = ProviderFingerprint(
        provider="azure_openai",
        model="deployment",
        generation_params=parameters,
        prompt_revision="stage4-v1",
        fingerprint_hash=make_provider_fingerprint_hash(
            provider="azure_openai",
            model="deployment",
            model_revision=None,
            generation_params_checksum=canonical_checksum({"params": parameters}),
            prompt_revision="stage4-v1",
        ),
    )
    digest = "sha256:" + "a" * 64
    return Stage4TestCaseRecord(
        tc_id=100001,
        **key.model_dump(),
        status="ACCEPTED",
        tc_steps=[TcStep(step=1, description="Start")],
        module="network",
        test_file="tests/network/Tc100001Start.py",
        robot_file="tests_robot/network/Tc100001Start.robot",
        generated_file_hashes={
            "tests/network/Tc100001Start.py": digest,
            "tests_robot/network/Tc100001Start.robot": digest,
        },
        framework_snapshot_id="FWS-1",
        test_data_snapshot_id="TDS-1",
        provider_fingerprint=fingerprint,
        prompt_revision="stage4-v1",
        input_fingerprint="sha256:input",
    )


def test_codegen_store_rebuilds_run_scoped_accepted_cases(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    store.mirror_tc_reservation(100001, _logical_key())
    store.save_stage4_test_case(_test_case(), run_id="RUN-1")

    artifact = store.rebuild_test_cases_artifact("alpha", "RUN-1")
    assert artifact.artifact_schema_version == "1.0-test-cases"
    assert len(artifact.test_cases) == 1
    assert artifact.test_cases[0].tc_id == 100001
    assert artifact.test_cases[0].status == "ACCEPTED"


def test_codegen_store_persists_blocker(tmp_path: Path) -> None:
    store = CodegenPostgresStore(_settings(tmp_path))
    store.ensure_schema()
    blocker = CodegenBlocker(
        blocker_id="BLK-1",
        scenario_id="TS-1",
        blocker_type="BLOCKED_MISSING_DATA",
        evidence="no approved binding",
    )
    store.save_blocker("alpha", blocker, run_id="RUN-1")
    artifact = store.rebuild_test_cases_artifact("alpha", "RUN-1")
    assert [item.blocker_id for item in artifact.blockers] == ["BLK-1"]
    # A separate project sees no test cases.
    empty = store.rebuild_test_cases_artifact("beta", "RUN-1")
    assert empty.test_cases == []
