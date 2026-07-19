"""Isolated Git worktree lifecycle for each codegen run (plan §14).

Every codegen run gets a dedicated branch and worktree so the user's primary
checkout is never touched. Before/after tree hashes are recorded, and failed
worktrees can be retained for diagnosis per policy.
"""

from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from multi_agentic_graph_rag.domain.identifiers import normalize_project


class WorktreeError(RuntimeError):
    """Raised when a worktree operation fails or would be unsafe."""


@dataclass(frozen=True)
class WorktreeHandle:
    """A created, isolated worktree bound to one codegen run."""

    codegen_run_id: str
    path: Path
    branch: str
    base_commit: str
    base_tree_hash: str
    primary_was_dirty: bool


def _git(cwd: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except FileNotFoundError as exc:  # pragma: no cover - environment dependent
        raise WorktreeError("git executable not found") from exc
    except subprocess.CalledProcessError as exc:
        raise WorktreeError(f"git {' '.join(args)} failed: {exc.stderr.strip()}") from exc
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - timing dependent
        raise WorktreeError(f"git {' '.join(args)} timed out") from exc
    return completed.stdout.strip()


class WorktreeManager:
    """Create and reclaim per-run worktrees under a managed root."""

    def __init__(self, worktrees_root: Path) -> None:
        self.worktrees_root = worktrees_root

    def create_worktree(
        self,
        *,
        repository_root: Path,
        codegen_run_id: str,
        base_commit: str | None = None,
    ) -> WorktreeHandle:
        """Add a detached-branch worktree for a run; refuse to reuse an existing one."""
        repository_root = repository_root.resolve()
        commit = base_commit or _git(repository_root, "rev-parse", "HEAD")
        base_tree_hash = _git(repository_root, "rev-parse", f"{commit}^{{tree}}")
        primary_dirty = bool(_git(repository_root, "status", "--porcelain").strip())

        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        slug = normalize_project(codegen_run_id)
        path = (self.worktrees_root / slug).resolve()
        if path.exists():
            raise WorktreeError(f"worktree already exists for run {codegen_run_id}: {path}")
        branch = f"codegen/{codegen_run_id}"
        _git(repository_root, "worktree", "add", "-b", branch, str(path), commit)
        return WorktreeHandle(
            codegen_run_id=codegen_run_id,
            path=path,
            branch=branch,
            base_commit=commit,
            base_tree_hash=base_tree_hash,
            primary_was_dirty=primary_dirty,
        )

    def current_tree_hash(self, handle: WorktreeHandle) -> str:
        """Stage all changes in the isolated worktree index and return its tree hash."""
        _git(handle.path, "add", "-A")
        return _git(handle.path, "write-tree")

    def remove_worktree(self, handle: WorktreeHandle, *, force: bool = True) -> None:
        """Remove the worktree and delete its codegen branch (call after persistence)."""
        repository_root = self._repository_root(handle.path)
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(handle.path))
        _git(repository_root, *args)
        # The codegen branch may already be gone; removal is best-effort.
        with contextlib.suppress(WorktreeError):
            _git(repository_root, "branch", "-D", handle.branch)

    def _repository_root(self, worktree_path: Path) -> Path:
        common = _git(worktree_path, "rev-parse", "--path-format=absolute", "--git-common-dir")
        return Path(common).parent.resolve()


__all__ = ["WorktreeError", "WorktreeHandle", "WorktreeManager"]
