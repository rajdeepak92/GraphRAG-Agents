"""Filesystem snapshot + installed Graphify update + verified KG publication."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.db.code_graph_store import CodeGraphStore
from multi_agentic_graph_rag.domain.code_graph_schemas import (
    CodeExtractionResult,
    CodeFile,
    FrameworkSnapshot,
)
from multi_agentic_graph_rag.services.framework_snapshot import compute_framework_snapshot
from multi_agentic_graph_rag.services.graphify_adapter import GraphifyAdapter


class GraphifyCommandError(RuntimeError):
    """Installed Graphify is missing, unsupported, or failed its update."""


@dataclass(frozen=True)
class GraphifyCapabilities:
    command: str
    version: str
    supports_no_cluster: bool


@dataclass(frozen=True)
class FrameworkIndexResult:
    snapshot: FrameworkSnapshot
    file_count: int
    symbol_count: int
    edge_count: int
    dependency_count: int
    graphify_version: str


def detect_graphify_capabilities(
    command: str,
    *,
    timeout: int = 30,
) -> GraphifyCapabilities:
    """Inspect only installed, documented Graphify capabilities."""
    version = _run_graphify([command, "--version"], timeout=timeout)
    help_text = _run_graphify([command, "update", "--help"], timeout=timeout)
    return GraphifyCapabilities(
        command=command,
        version=version.strip().splitlines()[0] if version.strip() else "unknown",
        supports_no_cluster="--no-cluster" in help_text,
    )


def run_graphify_update(
    framework_path: Path,
    capabilities: GraphifyCapabilities,
    *,
    no_cluster: bool,
    timeout: int = 300,
) -> None:
    """Run the documented ``graphify update <path> [--no-cluster]`` command."""
    argv = [capabilities.command, "update", str(framework_path)]
    if no_cluster and capabilities.supports_no_cluster:
        argv.append("--no-cluster")
    _run_graphify(argv, cwd=framework_path, timeout=timeout)


def index_framework(
    *,
    settings: AppSettings,
    framework_path: Path,
    graphify_out_dir: Path | None = None,
    repository_id: str | None = None,
    test_data_snapshot_id: str | None = None,
    changed_files: list[str] | None = None,
    run_update: bool = True,
) -> FrameworkIndexResult:
    """Update Graphify, publish BUILDING content, verify, then activate READY."""
    building = publish_framework_building(
        settings=settings,
        framework_path=framework_path,
        graphify_out_dir=graphify_out_dir,
        repository_id=repository_id,
        test_data_snapshot_id=test_data_snapshot_id,
        changed_files=changed_files,
        run_update=run_update,
    )
    root = framework_path.resolve(strict=True)
    expected = {path: _content_hash(root / path) for path in changed_files or []}
    if not expected:
        store = CodeGraphStore(settings)
        expected = store.snapshot_file_hashes(building.snapshot.snapshot_id)
    try:
        published = verify_and_activate_framework_snapshot(
            settings=settings,
            snapshot_id=building.snapshot.snapshot_id,
            expected_file_hashes=expected,
        )
    except Exception:
        CodeGraphStore(settings).mark_snapshot_failed(building.snapshot.snapshot_id)
        raise
    return FrameworkIndexResult(
        snapshot=published,
        file_count=building.file_count,
        symbol_count=building.symbol_count,
        edge_count=building.edge_count,
        dependency_count=building.dependency_count,
        graphify_version=building.graphify_version,
    )


def publish_framework_building(
    *,
    settings: AppSettings,
    framework_path: Path,
    graphify_out_dir: Path | None = None,
    repository_id: str | None = None,
    test_data_snapshot_id: str | None = None,
    changed_files: list[str] | None = None,
    run_update: bool = True,
) -> FrameworkIndexResult:
    """Publish a BUILDING snapshot; activation is a separate replayable node."""
    capabilities = detect_graphify_capabilities(settings.stage4.graphify_command)
    validated_root = framework_path.resolve(strict=True)
    if run_update:
        run_graphify_update(
            validated_root,
            capabilities,
            no_cluster=settings.stage4.graphify_no_cluster,
            timeout=max(settings.runtime.timeout_seconds, 300),
        )
    out_dir = graphify_out_dir or validated_root / "graphify-out"
    extractor_version = settings.stage4.extractor_version
    snapshot = compute_framework_snapshot(
        validated_root,
        allowed_roots=settings.stage4.framework_allowed_roots,
        extractor_version=extractor_version,
        extractor_config={
            "source": "graphify",
            "graphify_version": capabilities.version,
            "no_cluster": settings.stage4.graphify_no_cluster and capabilities.supports_no_cluster,
        },
        repository_id=repository_id,
        test_data_snapshot_id=test_data_snapshot_id,
    )
    adapter = GraphifyAdapter(extractor_version=extractor_version, graphify_out_dir=out_dir)
    extraction = adapter.load_extraction()
    result = adapter.transform(
        extraction,
        snapshot_id=snapshot.snapshot_id,
        repository_root=validated_root,
    )
    result = _include_changed_files(
        result, validated_root, changed_files or [], snapshot.snapshot_id
    )

    store = CodeGraphStore(settings)
    store.ensure_schema()
    store.begin_snapshot(
        snapshot,
        result,
        test_data_snapshot_id=test_data_snapshot_id,
    )
    return FrameworkIndexResult(
        snapshot=snapshot,
        file_count=len(result.files),
        symbol_count=len(result.symbols),
        edge_count=len(result.edges),
        dependency_count=len(result.dependencies),
        graphify_version=capabilities.version,
    )


def verify_and_activate_framework_snapshot(
    *,
    settings: AppSettings,
    snapshot_id: str,
    expected_file_hashes: dict[str, str],
) -> FrameworkSnapshot:
    """Verify changed artifacts then atomically select the new READY snapshot."""
    store = CodeGraphStore(settings)
    store.verify_snapshot(snapshot_id, expected_file_hashes)
    return store.activate_snapshot(snapshot_id)


def _include_changed_files(
    result: CodeExtractionResult,
    framework_root: Path,
    changed_files: list[str],
    snapshot_id: str,
) -> CodeExtractionResult:
    files = {file.relative_path: file for file in result.files}
    root = framework_root.resolve()
    for relative in changed_files:
        normalized = Path(relative).as_posix()
        if not normalized.endswith((".py", ".robot")):
            raise GraphifyCommandError(
                f"changed generated artifact is not Python/Robot: {normalized}"
            )
        path = (root / normalized).resolve(strict=True)
        if path != root and root not in path.parents:
            raise GraphifyCommandError(f"changed artifact escapes framework: {normalized}")
        files[normalized] = CodeFile(
            snapshot_id=snapshot_id,
            relative_path=normalized,
            language="robot" if normalized.endswith(".robot") else "python",
            content_hash=_content_hash(path),
        )
    return result.model_copy(update={"files": list(files.values())})


def _content_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run_graphify(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
) -> str:
    executable_name = Path(argv[0]).name.casefold()
    if executable_name in {"git", "git.exe"}:
        raise GraphifyCommandError(
            "Stage 4 forbids launching an executable named git; configure Graphify directly"
        )
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise GraphifyCommandError(f"Graphify executable not found: {Path(argv[0]).name}") from exc
    except subprocess.TimeoutExpired as exc:
        raise GraphifyCommandError(
            f"Graphify command timed out: {Path(argv[0]).name} {argv[1]}"
        ) from exc
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()
        if len(diagnostic) > 1000:
            diagnostic = diagnostic[:1000] + "...[truncated]"
        raise GraphifyCommandError(
            f"Graphify command failed ({completed.returncode}): {diagnostic}"
        )
    return completed.stdout


__all__ = [
    "FrameworkIndexResult",
    "GraphifyCapabilities",
    "GraphifyCommandError",
    "detect_graphify_capabilities",
    "index_framework",
    "publish_framework_building",
    "run_graphify_update",
    "verify_and_activate_framework_snapshot",
]
