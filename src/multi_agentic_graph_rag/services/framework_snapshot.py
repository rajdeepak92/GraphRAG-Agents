"""Deterministic Stage-4 framework identity from filesystem content only.

The coding workflow is intentionally independent of repository metadata.  A
snapshot is identified by the sorted set of normalized relative paths and the
SHA-256 of each file.  Tool-owned indexes, orchestration artifacts, caches, and
logs are excluded so publishing a KG does not change the framework identity it
is publishing.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path

from multi_agentic_graph_rag.domain.code_graph_schemas import FrameworkSnapshot
from multi_agentic_graph_rag.domain.identifiers import make_framework_snapshot_id, stable_token

_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".global_cache",
        ".graphify_python",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        ".vscode",
        "__pycache__",
        "generated",
        "graphify-out",
        "logs",
        "node_modules",
        "runtime",
        "venv",
    }
)
_EXCLUDED_SUFFIXES = frozenset({".log", ".pyc", ".pyo", ".tmp"})


class FrameworkPathError(ValueError):
    """The framework path is unsafe or outside the configured roots."""


def validate_framework_path(framework_path: Path, allowed_roots: list[Path]) -> Path:
    """Resolve the path and reject traversal or symlink escapes."""
    if not allowed_roots:
        raise FrameworkPathError("no allowed roots configured for framework indexing")
    resolved = framework_path.resolve(strict=True)
    if not resolved.is_dir():
        raise FrameworkPathError(f"framework path is not a directory: {resolved}")
    for root in allowed_roots:
        root_resolved = root.resolve(strict=True)
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved
    raise FrameworkPathError(f"framework path {resolved} is outside the allowed roots")


def filesystem_tree_entries(
    framework_root: Path,
    *,
    extra_excluded_dirs: Iterable[str] = (),
) -> list[tuple[str, str]]:
    """Return sorted ``(relative_path, sha256)`` entries for snapshot content."""
    root = framework_root.resolve(strict=True)
    excluded_dirs = _EXCLUDED_DIRS | frozenset(extra_excluded_dirs)
    entries: list[tuple[str, str]] = []
    casefolded: dict[str, str] = {}

    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            candidate = current_path / directory
            if candidate.is_symlink():
                resolved = candidate.resolve(strict=True)
                if resolved != root and root not in resolved.parents:
                    raise FrameworkPathError(
                        f"framework directory symlink escapes root: {candidate}"
                    )
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in excluded_dirs and not (current_path / directory).is_symlink()
        )
        for filename in sorted(filenames):
            path = current_path / filename
            if path.suffix.lower() in _EXCLUDED_SUFFIXES:
                continue
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if resolved != root and root not in resolved.parents:
                    raise FrameworkPathError(f"framework symlink escapes root: {path}")
            relative = path.relative_to(root).as_posix()
            folded = relative.casefold()
            prior = casefolded.get(folded)
            if prior is not None and prior != relative:
                raise FrameworkPathError(
                    f"case-normalization collision between {prior!r} and {relative!r}"
                )
            casefolded[folded] = relative
            digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((relative, digest))
    return sorted(entries, key=lambda row: row[0])


def compute_filesystem_checksum(
    framework_root: Path,
    *,
    extra_excluded_dirs: Iterable[str] = (),
) -> str:
    """Hash normalized paths and per-file hashes into one stable tree identity."""
    hasher = hashlib.sha256()
    for relative, digest in filesystem_tree_entries(
        framework_root, extra_excluded_dirs=extra_excluded_dirs
    ):
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(digest.encode("ascii"))
        hasher.update(b"\n")
    return "sha256:" + hasher.hexdigest()


def compute_framework_snapshot(
    framework_path: Path,
    *,
    allowed_roots: list[Path],
    extractor_version: str,
    extractor_config: dict[str, object] | None = None,
    repository_id: str | None = None,
    test_data_snapshot_id: str | None = None,
) -> FrameworkSnapshot:
    """Build a BUILDING snapshot without reading or invoking repository tools."""
    validated = validate_framework_path(framework_path, allowed_roots)
    filesystem_checksum = compute_filesystem_checksum(validated)
    config = extractor_config or {}
    extractor_config_hash = stable_token(
        *(f"{key}={config[key]}" for key in sorted(config)), length=24
    )
    resolved_repository_id = repository_id or stable_token(str(validated), length=16)
    snapshot_id = make_framework_snapshot_id(
        repository_id=resolved_repository_id,
        filesystem_checksum=filesystem_checksum,
        extractor_version=extractor_version,
        extractor_config_hash=extractor_config_hash,
    )
    return FrameworkSnapshot(
        snapshot_id=snapshot_id,
        repository_id=resolved_repository_id,
        canonical_path=str(validated),
        filesystem_checksum=filesystem_checksum,
        extractor_version=extractor_version,
        extractor_config_hash=extractor_config_hash,
        test_data_snapshot_id=test_data_snapshot_id,
        status="building",
        active=False,
    )


__all__ = [
    "FrameworkPathError",
    "compute_filesystem_checksum",
    "compute_framework_snapshot",
    "filesystem_tree_entries",
    "validate_framework_path",
]
