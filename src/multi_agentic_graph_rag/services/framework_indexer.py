"""Compose snapshot identity + Graphify extraction + code-graph publication (§7).

This is the deterministic framework-indexing pipeline: validate the path, pin an
immutable snapshot identity, transform the Graphify extraction into the canonical
code graph against the live worktree, and publish it READY to the code-graph
store only after integrity validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.services.framework_snapshot import compute_framework_snapshot
from multi_agentic_graph_rag.services.graphify_adapter import GraphifyAdapter


@dataclass(frozen=True)
class FrameworkIndexResult:
    """Summary of a completed framework indexing run."""

    snapshot: FrameworkSnapshot
    file_count: int
    symbol_count: int
    edge_count: int
    dependency_count: int


def index_framework(
    *,
    settings: AppSettings,
    framework_path: Path,
    graphify_out_dir: Path,
    repository_id: str | None = None,
) -> FrameworkIndexResult:
    """Index a framework revision into the code graph and return a summary."""
    extractor_version = settings.stage4.extractor_version
    snapshot = compute_framework_snapshot(
        framework_path,
        allowed_roots=settings.stage4.framework_allowed_roots,
        extractor_version=extractor_version,
        extractor_config={"source": "graphify", "out_dir": graphify_out_dir.name},
        repository_id=repository_id,
    )
    adapter = GraphifyAdapter(
        extractor_version=extractor_version, graphify_out_dir=graphify_out_dir
    )
    extraction = adapter.load_extraction()
    result = adapter.transform(
        extraction,
        snapshot_id=snapshot.snapshot_id,
        repository_root=Path(snapshot.canonical_path),
    )
    store = CodeGraphStore(settings)
    store.ensure_schema()
    published = store.publish_snapshot(snapshot, result)
    return FrameworkIndexResult(
        snapshot=published,
        file_count=len(result.files),
        symbol_count=len(result.symbols),
        edge_count=len(result.edges),
        dependency_count=len(result.dependencies),
    )


__all__ = ["FrameworkIndexResult", "index_framework"]
