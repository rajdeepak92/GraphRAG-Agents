"""Deterministic framework snapshot identity from Git provenance (plan §7.1-7.2).

The snapshot identity is derived from repository identity, the Git tree/dirty
hashes, and the extractor version/config so that an identical revision produces
an identical ``snapshot_id`` shareable across projects (plan §6.1, §23.1).
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.identifiers import make_framework_snapshot_id, stable_token


class FrameworkPathError(ValueError):
    """Raised when a framework path is outside the allowed roots or unsafe."""


class GitError(RuntimeError):
    """Raised when Git provenance cannot be established."""


def _run_git(repository_root: Path, *args: str) -> str:
    """Run a read-only Git command inside the repository and return stdout."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise GitError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raise GitError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing dependent
        raise GitError(f"git {' '.join(args)} timed out") from exc
    return completed.stdout.strip()


def validate_framework_path(framework_path: Path, allowed_roots: list[Path]) -> Path:
    """Resolve the path and reject traversal / symlink escapes (plan §7.2 steps 1-2)."""
    if not allowed_roots:
        raise FrameworkPathError("no allowed roots configured for framework indexing")
    resolved = framework_path.resolve()
    if not resolved.exists():
        raise FrameworkPathError(f"framework path does not exist: {resolved}")
    for root in allowed_roots:
        root_resolved = root.resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved
    raise FrameworkPathError(f"framework path {resolved} is outside the allowed roots")


def _repository_root(framework_path: Path) -> Path:
    top = _run_git(framework_path, "rev-parse", "--show-toplevel")
    return Path(top).resolve()


def compute_dirty_hash(repository_root: Path) -> tuple[bool, str]:
    """Return dirty status and a stable hash of the working-tree overlay (plan §6.1)."""
    porcelain = _run_git(repository_root, "status", "--porcelain")
    diff = _run_git(repository_root, "diff", "--no-color")
    dirty = bool(porcelain.strip())
    payload = f"{porcelain}\n---\n{diff}"
    dirty_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return dirty, dirty_hash


def compute_framework_snapshot(
    framework_path: Path,
    *,
    allowed_roots: list[Path],
    extractor_version: str,
    extractor_config: dict[str, object] | None = None,
    repository_id: str | None = None,
) -> FrameworkSnapshot:
    """Build an immutable framework snapshot identity for a pinned revision."""
    validated = validate_framework_path(framework_path, allowed_roots)
    repository_root = _repository_root(validated)
    branch = _run_git(repository_root, "rev-parse", "--abbrev-ref", "HEAD")
    commit = _run_git(repository_root, "rev-parse", "HEAD")
    tree_hash = _run_git(repository_root, "rev-parse", "HEAD^{tree}")
    dirty, dirty_hash = compute_dirty_hash(repository_root)
    config = extractor_config or {}
    extractor_config_hash = stable_token(
        *(f"{key}={config[key]}" for key in sorted(config)),
        length=24,
    )
    resolved_repository_id = repository_id or stable_token(str(repository_root), length=16)
    snapshot_id = make_framework_snapshot_id(
        repository_id=resolved_repository_id,
        tree_hash=tree_hash,
        dirty_hash=dirty_hash if dirty else "clean",
        extractor_version=extractor_version,
        extractor_config_hash=extractor_config_hash,
    )
    return FrameworkSnapshot(
        snapshot_id=snapshot_id,
        repository_id=resolved_repository_id,
        canonical_path=str(repository_root),
        branch=branch,
        commit=commit,
        tree_hash=tree_hash,
        dirty=dirty,
        dirty_hash=dirty_hash if dirty else "clean",
        extractor_version=extractor_version,
        extractor_config_hash=extractor_config_hash,
        status="building",
    )


__all__ = [
    "FrameworkPathError",
    "GitError",
    "compute_dirty_hash",
    "compute_framework_snapshot",
    "validate_framework_path",
]
