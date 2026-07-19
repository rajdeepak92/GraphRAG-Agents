"""Controlled, hash-guarded write tools confined to the isolated worktree (§13.1).

The reasoning model never gets raw filesystem access. Every write resolves and
validates its path inside the worktree, rejects symlink escapes and traversal,
enforces file-count/diff-size limits, and is guarded by the expected content
hash so a patch is never applied over stale or unexpected source (plan §10.6).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


class PatchError(ValueError):
    """Raised when a write is unsafe, out of policy, or violates a hash guard."""


class StalePatchError(PatchError):
    """Raised when the on-disk hash does not match the expected hash (§10.6)."""


@dataclass(frozen=True)
class PatchLimits:
    """Bounded editing policy (plan §13.3)."""

    max_files: int = 20
    max_bytes_per_file: int = 256_000


def file_sha256(text: str) -> str:
    """Content hash in the same ``sha256:`` form used across the code graph."""
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


class PatchExecutor:
    """Apply create/patch/delete operations strictly inside one worktree."""

    def __init__(self, worktree_root: Path, *, limits: PatchLimits | None = None) -> None:
        self.worktree_root = worktree_root.resolve()
        self.limits = limits or PatchLimits()
        self._touched: set[str] = set()

    @property
    def touched_files(self) -> set[str]:
        """Relative paths created, patched, or deleted so far this run."""
        return set(self._touched)

    def _resolve(self, relative_path: str) -> Path:
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise PatchError(f"unsafe path (absolute or traversal): {relative_path}")
        target = self.worktree_root / rel
        real_root = os.path.realpath(self.worktree_root)
        real_parent = os.path.realpath(target.parent)
        if real_parent != real_root and not real_parent.startswith(real_root + os.sep):
            raise PatchError(f"path escapes worktree via symlink: {relative_path}")
        if target.exists() and target.is_symlink():
            raise PatchError(f"refusing to write through a symlink: {relative_path}")
        return target

    def _register(self, relative_path: str) -> None:
        if relative_path not in self._touched and len(self._touched) >= self.limits.max_files:
            raise PatchError(f"file-count limit ({self.limits.max_files}) exceeded")
        self._touched.add(relative_path)

    def _check_size(self, relative_path: str, content: str) -> None:
        size = len(content.encode("utf-8"))
        if size > self.limits.max_bytes_per_file:
            raise PatchError(
                f"{relative_path} is {size} bytes, exceeding the "
                f"{self.limits.max_bytes_per_file}-byte limit"
            )

    def read_hash(self, relative_path: str) -> str | None:
        """Return the current content hash, or ``None`` when the file is absent."""
        target = self._resolve(relative_path)
        if not target.exists():
            return None
        return file_sha256(target.read_text(encoding="utf-8"))

    def create_file(self, relative_path: str, content: str, *, expected_absent: bool = True) -> str:
        """Create a new file, refusing to clobber an existing one by default."""
        self._check_size(relative_path, content)
        target = self._resolve(relative_path)
        if expected_absent and target.exists():
            raise PatchError(f"file already exists: {relative_path}")
        self._register(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return file_sha256(content)

    def apply_patch(self, relative_path: str, new_content: str, *, expected_hash: str) -> str:
        """Replace a file's content only when its current hash matches the guard."""
        self._check_size(relative_path, new_content)
        current = self.read_hash(relative_path)
        if current is None:
            raise StalePatchError(f"cannot patch missing file: {relative_path}")
        if current != expected_hash:
            raise StalePatchError(
                f"hash guard failed for {relative_path}: expected {expected_hash}, found {current}"
            )
        self._register(relative_path)
        self._resolve(relative_path).write_text(new_content, encoding="utf-8")
        return file_sha256(new_content)

    def delete_generated_file(self, relative_path: str, *, expected_hash: str) -> None:
        """Delete a file only when its current hash matches the guard."""
        current = self.read_hash(relative_path)
        if current is None:
            raise StalePatchError(f"cannot delete missing file: {relative_path}")
        if current != expected_hash:
            raise StalePatchError(f"hash guard failed for delete of {relative_path}")
        self._register(relative_path)
        self._resolve(relative_path).unlink()


__all__ = ["PatchError", "PatchExecutor", "PatchLimits", "StalePatchError", "file_sha256"]
