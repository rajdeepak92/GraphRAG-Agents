"""Tests for the worktree manager and hash-guarded patch executor."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from multi_agentic_graph_rag.services.patch_executor import (
    PatchError,
    PatchExecutor,
    PatchLimits,
    StalePatchError,
    file_sha256,
)
from multi_agentic_graph_rag.services.worktree_manager import WorktreeError, WorktreeManager


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def temp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "mod.py").write_text("def start():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


# --- Worktree manager --------------------------------------------------------


def test_worktree_lifecycle_isolates_from_primary(tmp_path: Path, temp_repo: Path) -> None:
    manager = WorktreeManager(tmp_path / "worktrees")
    handle = manager.create_worktree(repository_root=temp_repo, codegen_run_id="CGR-1")
    assert handle.path.exists()
    assert (handle.path / "mod.py").exists()
    assert len(handle.base_tree_hash) == 40
    assert handle.primary_was_dirty is False

    # Editing inside the worktree must not touch the primary checkout.
    (handle.path / "new_test.py").write_text("x = 1\n", encoding="utf-8")
    assert not (temp_repo / "new_test.py").exists()
    tree_hash = manager.current_tree_hash(handle)
    assert tree_hash != handle.base_tree_hash

    manager.remove_worktree(handle)
    assert not handle.path.exists()


def test_worktree_refuses_duplicate_run(tmp_path: Path, temp_repo: Path) -> None:
    manager = WorktreeManager(tmp_path / "worktrees")
    manager.create_worktree(repository_root=temp_repo, codegen_run_id="CGR-1")
    with pytest.raises(WorktreeError):
        manager.create_worktree(repository_root=temp_repo, codegen_run_id="CGR-1")


# --- Patch executor ----------------------------------------------------------


def test_create_apply_delete_with_hash_guards(tmp_path: Path) -> None:
    root = tmp_path / "wt"
    root.mkdir()
    executor = PatchExecutor(root)

    created_hash = executor.create_file("tests/test_new.py", "a = 1\n")
    assert created_hash == file_sha256("a = 1\n")
    assert executor.read_hash("tests/test_new.py") == created_hash
    assert "tests/test_new.py" in executor.touched_files

    # Creating over an existing file is refused.
    with pytest.raises(PatchError):
        executor.create_file("tests/test_new.py", "b = 2\n")

    # A stale expected hash blocks the patch.
    with pytest.raises(StalePatchError):
        executor.apply_patch("tests/test_new.py", "c = 3\n", expected_hash="sha256:stale")

    new_hash = executor.apply_patch("tests/test_new.py", "c = 3\n", expected_hash=created_hash)
    assert (root / "tests" / "test_new.py").read_text(encoding="utf-8") == "c = 3\n"

    with pytest.raises(StalePatchError):
        executor.delete_generated_file("tests/test_new.py", expected_hash="sha256:stale")
    executor.delete_generated_file("tests/test_new.py", expected_hash=new_hash)
    assert not (root / "tests" / "test_new.py").exists()


def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "wt"
    root.mkdir()
    executor = PatchExecutor(root)
    with pytest.raises(PatchError):
        executor.create_file("../escape.py", "x = 1\n")
    with pytest.raises(PatchError):
        executor.create_file("/etc/passwd", "x = 1\n")


def test_file_count_and_size_limits(tmp_path: Path) -> None:
    root = tmp_path / "wt"
    root.mkdir()
    executor = PatchExecutor(root, limits=PatchLimits(max_files=1, max_bytes_per_file=10))
    executor.create_file("a.py", "x=1\n")
    with pytest.raises(PatchError):
        executor.create_file("b.py", "y=2\n")  # exceeds max_files
    big = PatchExecutor(root, limits=PatchLimits(max_files=5, max_bytes_per_file=3))
    with pytest.raises(PatchError):
        big.create_file("c.py", "way too long\n")  # exceeds size


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "wt"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (root / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")
    executor = PatchExecutor(root)
    with pytest.raises(PatchError):
        executor.create_file("link/evil.py", "x = 1\n")
