"""LangGraph apply-loop tests: happy path, approval interrupt/resume, blockers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.codegen_postgres import CodegenPostgresStore
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData
from multi_agentic_graph_rag.services.capability_mapper import CapabilityCandidate
from multi_agentic_graph_rag.services.patch_executor import PatchExecutor
from multi_agentic_graph_rag.services.patch_producer import FileOp, StaticPatchProducer
from multi_agentic_graph_rag.services.test_data_document_reader import read_document
from multi_agentic_graph_rag.services.test_data_ingestion import ingest_document
from multi_agentic_graph_rag.services.validation_runner import ValidationRunner
from multi_agentic_graph_rag.workflows.codegen_apply_graph import (
    CodegenRuntime,
    build_scenario_codegen_graph,
)
from multi_agentic_graph_rag.workflows.test_code_generation_graph import (
    ScenarioAction,
    ScenarioPlanRequest,
)

_PASSING_TEST = "def test_generated():\n    assert 1 + 1 == 2\n"
_BROKEN_TEST = "def test_broken(:\n    assert True\n"


def _settings(tmp_path: Path) -> AppSettings:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.stage4.codegen_local_path = tmp_path / "codegen.jsonl"
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


def _normalized(scenario_id: str = "TS-1") -> NormalizedTestData:
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
                    "scenario_id": scenario_id,
                    "execution_profile_id": "PROFILE-1",
                    "fixture_id": "FIX-1",
                    "cleanup_id": "CLEAN-1",
                    "oracle_ids": ["ORA-1"],
                    "approval_status": "APPROVED",
                }
            ],
        }
    )
    result = ingest_document(document, scenario_ids={scenario_id})
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


def _plan_request(
    scenario_id: str, normalized: NormalizedTestData, *, mapped: bool = True
) -> ScenarioPlanRequest:
    candidates = [CapabilityCandidate("SYM-A", "fw.assert_status", "exact")] if mapped else []
    actions = [
        ScenarioAction(
            action_text="Assert status 200",
            capability_id="assert.status",
            candidates=candidates,
        )
    ]
    return ScenarioPlanRequest(
        scenario_id=scenario_id,
        execution_profile_id="PROFILE-1",
        codegen_run_id="CGR-1",
        framework_snapshot=_snapshot(),
        normalized=normalized,
        actions=actions,
    )


def _runtime(
    tmp_path: Path, request: ScenarioPlanRequest, ops: list[FileOp], **over: Any
) -> CodegenRuntime:
    settings = _settings(tmp_path)
    store = CodegenPostgresStore(settings)
    store.ensure_schema()
    worktree = tmp_path / "wt"
    worktree.mkdir(exist_ok=True)
    defaults: dict[str, Any] = {
        "settings": settings,
        "plan_request": request,
        "patch_producer": StaticPatchProducer(ops),
        "patch_executor": PatchExecutor(worktree),
        "validation_runner": ValidationRunner(worktree, python_executable=sys.executable),
        "codegen_store": store,
        "project": "alpha",
        "test_name": "test_generated",
        "test_file": "tests/test_generated.py",
        "test_function": "test_generated",
        "checkpointer": InMemorySaver(),
        "max_repair_attempts": 1,
    }
    defaults.update(over)
    return CodegenRuntime(**defaults)


def _config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def test_apply_happy_path_generates_and_persists(tmp_path: Path) -> None:
    normalized = _normalized()
    request = _plan_request("TS-1", normalized)
    ops = [FileOp(kind="create", relative_path="tests/test_generated.py", content=_PASSING_TEST)]
    runtime = _runtime(tmp_path, request, ops)
    graph = build_scenario_codegen_graph(runtime)

    final = graph.invoke(
        {"scenario_id": "TS-1", "codegen_run_id": "CGR-1"}, config=_config("t-happy")
    )
    assert final["status"] == "COMPLETED"
    assert final["test_case_id"].startswith("TC-")
    assert (tmp_path / "wt" / "tests" / "test_generated.py").exists()

    artifact = runtime.codegen_store.rebuild_test_cases_artifact("alpha", "RUN-1")
    assert len(artifact.test_cases) == 1
    assert artifact.test_cases[0].scenario_id == "TS-1"


def test_apply_requires_approval_then_resumes(tmp_path: Path) -> None:
    normalized = _normalized()
    request = _plan_request("TS-1", normalized)
    ops = [FileOp(kind="create", relative_path="tests/test_generated.py", content=_PASSING_TEST)]
    runtime = _runtime(tmp_path, request, ops, require_approval=True)
    graph = build_scenario_codegen_graph(runtime)

    first = graph.invoke(
        {"scenario_id": "TS-1", "codegen_run_id": "CGR-1"}, config=_config("t-approve")
    )
    assert "__interrupt__" in first  # paused at the approval gate

    resumed = graph.invoke(Command(resume={"approved": True}), config=_config("t-approve"))
    assert resumed["status"] == "COMPLETED"


def test_apply_rejected_approval_blocks(tmp_path: Path) -> None:
    normalized = _normalized()
    request = _plan_request("TS-1", normalized)
    ops = [FileOp(kind="create", relative_path="tests/test_generated.py", content=_PASSING_TEST)]
    runtime = _runtime(tmp_path, request, ops, require_approval=True)
    graph = build_scenario_codegen_graph(runtime)

    graph.invoke({"scenario_id": "TS-1", "codegen_run_id": "CGR-1"}, config=_config("t-reject"))
    final = graph.invoke(Command(resume={"approved": False}), config=_config("t-reject"))
    assert final["status"] == "BLOCKED"
    assert not (tmp_path / "wt" / "tests" / "test_generated.py").exists()


def test_apply_blocks_on_unmapped_capability(tmp_path: Path) -> None:
    normalized = _normalized()
    request = _plan_request("TS-1", normalized, mapped=False)
    ops = [FileOp(kind="create", relative_path="tests/test_generated.py", content=_PASSING_TEST)]
    runtime = _runtime(tmp_path, request, ops)
    graph = build_scenario_codegen_graph(runtime)

    final = graph.invoke(
        {"scenario_id": "TS-1", "codegen_run_id": "CGR-1"}, config=_config("t-cap")
    )
    assert final["status"] == "BLOCKED"
    # Nothing was written because the readiness gate blocked before apply.
    assert not (tmp_path / "wt" / "tests" / "test_generated.py").exists()


def test_apply_repairs_then_blocks_on_persistent_syntax_error(tmp_path: Path) -> None:
    normalized = _normalized()
    request = _plan_request("TS-1", normalized)
    ops = [FileOp(kind="create", relative_path="tests/test_broken.py", content=_BROKEN_TEST)]
    runtime = _runtime(tmp_path, request, ops, max_repair_attempts=1)
    graph = build_scenario_codegen_graph(runtime)

    final = graph.invoke(
        {"scenario_id": "TS-1", "codegen_run_id": "CGR-1"}, config=_config("t-repair")
    )
    assert final["status"] == "BLOCKED"
    assert final["validation_status"] == "FAILED_VALIDATION"
    assert final["repair_attempt"] == 1  # one repair attempt was consumed
