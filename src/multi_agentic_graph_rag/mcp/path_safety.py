"""Project-relative path validation for MCP inputs."""

from __future__ import annotations

from pathlib import Path

DEFAULT_ALLOWED_ROOTS = ("documents", "generated", "runtime", ".generated")


class UnsafePathError(ValueError):
    """Raised when a user path escapes the project boundary."""


def resolve_project_path(
    project_root: Path,
    user_path: str,
    *,
    allowed_roots: tuple[str, ...] | None = DEFAULT_ALLOWED_ROOTS,
) -> Path:
    root = project_root.resolve()
    raw_path = Path(user_path)
    if raw_path.is_absolute():
        raise UnsafePathError(f"absolute paths are not allowed: {user_path}")

    candidate = (root / raw_path).resolve()
    if root != candidate and root not in candidate.parents:
        raise UnsafePathError(f"path escapes project root: {user_path}")

    if allowed_roots is not None and candidate != root:
        relative = candidate.relative_to(root)
        if not relative.parts or relative.parts[0] not in allowed_roots:
            allowed = ", ".join(allowed_roots)
            raise UnsafePathError(f"path must be under one of [{allowed}]: {user_path}")

    return candidate
