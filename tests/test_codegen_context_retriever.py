"""Complete lineage/data/source retrieval and budget-tier tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.domain.code_graph_schemas import CodeSymbol, FrameworkSnapshot
from multi_agentic_graph_rag.domain.schemas import canonical_checksum
from multi_agentic_graph_rag.domain.test_data_schemas import (
    NormalizedTestData,
    ScenarioDataBinding,
    TestDataRecord,
)
from multi_agentic_graph_rag.services.codegen_context_retriever import CodegenContextRetriever


class _Lineage:
    def load_test_scenarios(self, project: str, run_id: str) -> dict[str, Any]:
        del project, run_id
        return {
            "scenarios": [
                {
                    "scenario_id": "TS-1",
                    "story_ids": ["US-1", "US-2"],
                    "requirement_ids": ["REQ-1", "REQ-2"],
                    "title": "Validate sensor threshold",
                    "action": "Read the temperature sensor",
                    "expected_result": "Threshold is enforced",
                    "traceability": {"evidence_chunk_ids": ["CHK-S"]},
                }
            ]
        }

    def load_user_stories(self, project: str, run_id: str) -> dict[str, Any]:
        del project, run_id
        return {
            "stories": [
                {
                    "story_id": "US-1",
                    "requirement_ids": ["REQ-1"],
                    "traceability": {"evidence_chunk_ids": ["CHK-1"]},
                },
                {
                    "story_id": "US-2",
                    "requirement_ids": ["REQ-2"],
                    "traceability": {"evidence_chunk_ids": ["CHK-2"]},
                },
            ]
        }

    def load_requirements(self, project: str, run_id: str) -> dict[str, Any]:
        del project, run_id
        return {
            "requirements": [
                {
                    "requirement_id": "REQ-1",
                    "requirement_text": "Read exact temperature.",
                    "evidence": [{"chunk_id": "CHK-1", "quote": "exact temperature"}],
                },
                {
                    "requirement_id": "REQ-2",
                    "requirement_text": "Enforce threshold.",
                    "evidence": [{"chunk_id": "CHK-2", "quote": "threshold"}],
                },
            ]
        }


class _Evidence:
    def fetch_chunks(self, project: str, chunk_ids: set[str]) -> list[tuple[str, str]]:
        del project
        return [(chunk_id, f"source text for {chunk_id}") for chunk_id in sorted(chunk_ids)]


class _CodeGraph:
    def __init__(self, matched: CodeSymbol, neighbor: CodeSymbol) -> None:
        self.matched = matched
        self.neighbor = neighbor
        self.neighbor_requests: list[tuple[str, ...]] = []

    def get_symbol(self, snapshot_id: str, symbol_id: str) -> CodeSymbol | None:
        del snapshot_id
        return self.matched if symbol_id == self.matched.symbol_id else None

    def search_symbols(
        self,
        snapshot_id: str,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[CodeSymbol]:
        del snapshot_id, query, kinds, limit
        return [self.matched]

    def get_neighbors(
        self,
        snapshot_id: str,
        symbol_id: str,
        *,
        relations: list[str] | None = None,
        depth: int = 1,
    ) -> list[CodeSymbol]:
        del snapshot_id, symbol_id, depth
        current = tuple(relations or [])
        self.neighbor_requests.append(current)
        return [self.neighbor] if current == ("CALLS",) else []


def _hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode()).hexdigest()}"


def _record(record_id: str, record_type: str, payload: dict[str, Any]) -> TestDataRecord:
    return TestDataRecord(
        record_id=record_id,
        record_status="APPROVED",
        name=record_id,
        owner="qa",
        valid_from_revision="r1",
        record_type=record_type,
        natural_key=record_id.casefold(),
        payload=payload,
        payload_checksum=canonical_checksum({"payload": payload}),
        source_sheet=record_type,
        source_row=2,
    )


def _normalized() -> NormalizedTestData:
    records = [
        _record(
            "EP-1",
            "ExecutionProfile",
            {"ip": "10.0.0.10", "credential": "secret://vault/sensor"},
        ),
        _record("FIX-DEFAULT", "Fixture", {"profile_ref": "EP-1", "temperature": 20}),
        _record("FIX-HOT", "Fixture", {"profile_ref": "EP-1", "temperature": 90}),
        _record("CLEAN", "Cleanup", {"mode": "safe"}),
        _record("ORACLE", "Oracle", {"predicate": {"equals": 80}}),
        _record(
            "STEP-1",
            "ActionStep",
            {"sequence_id": "SEQ-1", "step": 1, "action_id": "read", "port": 502},
        ),
    ]
    bindings = [
        ScenarioDataBinding(
            binding_id="BND-DEFAULT",
            scenario_id="TS-1",
            execution_profile_id="EP-1",
            variant_id="default",
            fixture_id="FIX-DEFAULT",
            action_sequence_id="SEQ-1",
            oracle_ids=["ORACLE"],
            cleanup_id="CLEAN",
            approval_status="APPROVED",
        ),
        ScenarioDataBinding(
            binding_id="BND-HOT",
            scenario_id="TS-1",
            execution_profile_id="EP-1",
            variant_id="hot",
            fixture_id="FIX-HOT",
            oracle_ids=["ORACLE"],
            cleanup_id="CLEAN",
            approval_status="APPROVED",
        ),
    ]
    draft = NormalizedTestData.model_construct(
        snapshot_id="TDS-1",
        project="demo",
        schema_version="1.0",
        workbook_checksum="sha256:workbook",
        decision_revision="r1",
        records=records,
        bindings=bindings,
        checksum="",
    )
    return NormalizedTestData.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )


def _snapshot(root: Path) -> FrameworkSnapshot:
    return FrameworkSnapshot(
        snapshot_id="FWS-1",
        repository_id="framework",
        canonical_path=str(root),
        filesystem_checksum="sha256:framework",
        extractor_version="graphify",
        extractor_config_hash="sha256:config",
        status="ready",
        active=True,
    )


def _retriever(tmp_path: Path) -> tuple[CodegenContextRetriever, FrameworkSnapshot]:
    source = "def read_sensor():\n    return True\n\ndef normalize_sensor():\n    return 80\n"
    source_path = tmp_path / "framework" / "sensor.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(source, encoding="utf-8")
    wrapper = tmp_path / "framework" / "test_lib" / "sensor" / "sensor_wrappers.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(
        "def read_sensor_for_test():\n"
        "    from sensor import read_sensor\n"
        "    return read_sensor()\n",
        encoding="utf-8",
    )
    matched_body = "def read_sensor():\n    return True"
    neighbor_body = "def normalize_sensor():\n    return 80"
    matched = CodeSymbol(
        snapshot_id="FWS-1",
        symbol_id="SYM-READ",
        fqn="sensor.py::read_sensor",
        kind="Function",
        signature="read_sensor()",
        relative_path="sensor.py",
        start_line=1,
        end_line=2,
        start_byte=0,
        end_byte=0,
        body_hash=_hash(matched_body),
    )
    neighbor = CodeSymbol(
        snapshot_id="FWS-1",
        symbol_id="SYM-NORMALIZE",
        fqn="sensor.py::normalize_sensor",
        kind="Function",
        signature="normalize_sensor()",
        relative_path="sensor.py",
        start_line=4,
        end_line=5,
        start_byte=0,
        end_byte=0,
        body_hash=_hash(neighbor_body),
    )
    root = tmp_path / "framework"
    return (
        CodegenContextRetriever(
            lineage_store=_Lineage(),
            evidence_store=_Evidence(),
            code_graph=_CodeGraph(matched, neighbor),
            framework_root=root,
        ),
        _snapshot(root),
    )


def test_retrieves_complete_lineage_exact_data_source_and_neighbors(tmp_path: Path) -> None:
    retriever, snapshot = _retriever(tmp_path)
    context = retriever.scenario_codegen_context(
        project="demo",
        run_id="RUN-1",
        scenario_id="TS-1",
        execution_profile_id="EP-1",
        normalized_test_data=_normalized(),
        framework_snapshot=snapshot,
        matched_symbol_ids=("SYM-READ",),
        module_hints=("sensor",),
        token_budget=50_000,
    )

    assert [story["story_id"] for story in context.stories] == ["US-1", "US-2"]
    assert [req["requirement_id"] for req in context.requirements] == ["REQ-1", "REQ-2"]
    assert context.evidence_chunk_ids == ("CHK-S", "CHK-1", "CHK-2")
    assert context.framework_snapshot_id == "FWS-1"
    records = {record["record_id"]: record for record in context.test_data_records}
    assert records["EP-1"]["payload"]["ip"] == "10.0.0.10"
    assert records["STEP-1"]["payload"]["port"] == 502
    assert context.secret_refs == ("secret://vault/sensor",)
    assert context.matched_symbols[0].source_body == "def read_sensor():\n    return True"
    assert context.neighbor_symbols[0].relations == ("CALLS",)
    assert context.test_lib_symbols[0].fqn.endswith("::read_sensor_for_test")


def test_variant_selects_only_matching_binding_and_records(tmp_path: Path) -> None:
    retriever, snapshot = _retriever(tmp_path)
    context = retriever.scenario_codegen_context(
        project="demo",
        run_id="RUN-1",
        scenario_id="TS-1",
        execution_profile_id="EP-1",
        variant_id="hot",
        normalized_test_data=_normalized(),
        framework_snapshot=snapshot,
        token_budget=50_000,
    )
    assert context.resolved_binding["binding_id"] == "BND-HOT"
    record_ids = {record["record_id"] for record in context.test_data_records}
    assert "FIX-HOT" in record_ids
    assert "FIX-DEFAULT" not in record_ids


def test_mandatory_lineage_and_exact_data_are_never_dropped(tmp_path: Path) -> None:
    retriever, snapshot = _retriever(tmp_path)
    context = retriever.scenario_codegen_context(
        project="demo",
        run_id="RUN-1",
        scenario_id="TS-1",
        execution_profile_id="EP-1",
        normalized_test_data=_normalized(),
        framework_snapshot=snapshot,
        token_budget=1,
    )
    assert context.mandatory_over_budget is True
    assert context.scenario["scenario_id"] == "TS-1"
    assert len(context.stories) == 2
    assert context.test_data_records
    assert context.matched_symbols == ()
    assert context.evidence_chunks == ()
