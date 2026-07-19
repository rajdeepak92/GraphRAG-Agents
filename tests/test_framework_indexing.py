"""Tests for framework snapshot identity and the Graphify code-graph adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from multi_agentic_graph_rag.domain.code_graph_schemas import CodeExtractionResult
from multi_agentic_graph_rag.services.framework_snapshot import (
    FrameworkPathError,
    compute_framework_snapshot,
    validate_framework_path,
)
from multi_agentic_graph_rag.services.graphify_adapter import (
    GraphifyAdapter,
    GraphifyExtractionError,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# --- Path safety -------------------------------------------------------------


def test_validate_framework_path_rejects_outside_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(FrameworkPathError):
        validate_framework_path(outside, [allowed])
    assert validate_framework_path(allowed, [allowed]) == allowed.resolve()


def test_validate_framework_path_requires_allowed_roots(tmp_path: Path) -> None:
    with pytest.raises(FrameworkPathError):
        validate_framework_path(tmp_path, [])


# --- Snapshot identity -------------------------------------------------------


def test_framework_snapshot_is_deterministic_for_a_revision() -> None:
    kwargs = {
        "allowed_roots": [_REPO_ROOT],
        "extractor_version": "graphify-test",
        "extractor_config": {"mode": "ast"},
        "repository_id": "self",
    }
    first = compute_framework_snapshot(_REPO_ROOT, **kwargs)
    second = compute_framework_snapshot(_REPO_ROOT, **kwargs)
    assert first.snapshot_id == second.snapshot_id
    assert first.snapshot_id.startswith("FWS-")
    assert len(first.commit) == 40
    assert len(first.tree_hash) == 40
    # A different extractor config must yield a different snapshot identity.
    other = compute_framework_snapshot(
        _REPO_ROOT, **{**kwargs, "extractor_config": {"mode": "deep"}}
    )
    assert other.snapshot_id != first.snapshot_id


# --- Adapter transform (synthetic) -------------------------------------------


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_adapter_transforms_and_enforces_integrity(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "mod.py", "class Slave:\n    def start(self):\n        return 1\n")
    extraction = {
        "nodes": [
            {"id": "f1", "label": "mod.py", "file_type": "code", "source_file": "pkg/mod.py"},
            {
                "id": "c1",
                "label": "Slave",
                "file_type": "code",
                "source_file": "pkg/mod.py",
                "source_location": "L1",
                "confidence": "EXTRACTED",
            },
            {
                "id": "m1",
                "label": "Slave.start()",
                "file_type": "code",
                "source_file": "pkg/mod.py",
                "source_location": "L2-L3",
            },
            {
                "id": "r1",
                "label": "a rationale note",
                "file_type": "rationale",
                "source_file": "pkg/mod.py",
            },
        ],
        "edges": [
            {"source": "c1", "target": "m1", "relation": "method", "source_location": "L2"},
            {"source": "c1", "target": "m1", "relation": "rationale_for"},
        ],
    }
    adapter = GraphifyAdapter(extractor_version="graphify-test")
    result = adapter.transform(extraction, snapshot_id="FWS-1", repository_root=tmp_path)
    assert isinstance(result, CodeExtractionResult)
    assert {f.relative_path for f in result.files} == {"pkg/mod.py"}
    kinds = {s.kind for s in result.symbols}
    assert kinds == {"Class", "Method"}
    # Only the structural 'method' edge survives; rationale_for is dropped.
    assert len(result.edges) == 1
    assert result.edges[0].relation == "DECLARES"


def test_adapter_rejects_missing_worktree_file(tmp_path: Path) -> None:
    extraction = {
        "nodes": [
            {"id": "f1", "label": "gone.py", "file_type": "code", "source_file": "gone.py"},
        ],
        "edges": [],
    }
    adapter = GraphifyAdapter(extractor_version="graphify-test")
    with pytest.raises(GraphifyExtractionError):
        adapter.transform(extraction, snapshot_id="FWS-1", repository_root=tmp_path)


# --- Adapter transform (real repository extraction) --------------------------


@pytest.mark.skipif(
    not (_REPO_ROOT / "graphify-out" / ".graphify_extract.json").exists(),
    reason="graphify extraction not present",
)
def test_adapter_transforms_real_repository_extraction() -> None:
    adapter = GraphifyAdapter(
        extractor_version="graphify-real",
        graphify_out_dir=_REPO_ROOT / "graphify-out",
    )
    extraction = adapter.load_extraction()
    # Constructing the result validates referential integrity (no dangling edges).
    result = adapter.transform(extraction, snapshot_id="FWS-real", repository_root=_REPO_ROOT)
    assert result.files, "expected at least one indexed file"
    assert result.symbols, "expected at least one extracted symbol"
    symbol_ids = {s.symbol_id for s in result.symbols}
    for edge in result.edges:
        assert edge.source_symbol_id in symbol_ids
        assert edge.target_symbol_id in symbol_ids
