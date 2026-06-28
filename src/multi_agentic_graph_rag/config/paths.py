"""Runtime path resolution and bootstrap helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the nearest parent directory containing pyproject.toml."""

    current = (start or Path.cwd()).resolve()

    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate

    msg = "Unable to locate project root because pyproject.toml was not found."
    raise RuntimeError(msg)


def resolve_under_root(project_root: Path, raw_path: str | Path) -> Path:
    """Resolve a path relative to the project root unless already absolute."""

    path = Path(raw_path).expanduser()

    if not path.is_absolute():
        path = project_root / path

    return path.resolve()


def require_under(parent: Path, child: Path, *, label: str) -> None:
    """Reject paths that escape their approved parent directory."""

    parent = parent.resolve()
    child = child.resolve()

    try:
        child.relative_to(parent)
    except ValueError as exc:
        msg = f"{label} must resolve under {parent}. Resolved value: {child}"
        raise ValueError(msg) from exc


def create_directories(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Create approved directories and return the created/resolved paths."""

    created: list[Path] = []

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        created.append(path.resolve())

    return tuple(created)


def set_cache_environment(cache_env: Mapping[str, Path]) -> None:
    """Redirect process cache environment variables."""

    for env_name, path in cache_env.items():
        os.environ[env_name] = str(path)
