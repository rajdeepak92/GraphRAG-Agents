"""Durable per-project display aliases for generated artifacts."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DISPLAY_WIDTH = 3
_LOCAL_REGISTRY_PATH = Path("runtime") / "staging" / "artifact_display_ids.jsonl"
_LOCAL_LOCK_PATH = Path("runtime") / "locks" / "artifact_display_ids.lock"
_DISPLAY_RE = re.compile(r"^[A-Z]+-(\d+)$")
_KIND_PREFIX = {
    "requirement": "REQ",
    "requirements": "REQ",
    "req": "REQ",
    "user_story": "US",
    "user_stories": "US",
    "story": "US",
    "test_scenario": "TS",
    "test_scenarios": "TS",
    "scenario": "TS",
}


def resolve_many(
    project: str,
    kind: str,
    internal_ids: Sequence[str],
    cursor: Any | None = None,
) -> dict[str, str]:
    """Resolve internal IDs to stable per-project display IDs."""
    normalized_project = project.strip()
    normalized_kind = _normalize_kind(kind)
    ordered_ids = _unique_ids(internal_ids)
    if not ordered_ids:
        return {}
    if not normalized_project:
        raise ValueError("project must not be empty")
    if cursor is not None:
        return _resolve_postgres(
            normalized_project,
            normalized_kind,
            ordered_ids,
            cursor,
        )
    return _resolve_local(normalized_project, normalized_kind, ordered_ids)


def _resolve_postgres(
    project: str,
    kind: str,
    internal_ids: list[str],
    cursor: Any,
) -> dict[str, str]:
    cursor.execute(
        """
        select internal_id, display_id
        from artifact_display_ids
        where internal_id = any(%s)
        """,
        (internal_ids,),
    )
    resolved = {str(row[0]): str(row[1]) for row in cursor.fetchall()}
    missing = [internal_id for internal_id in internal_ids if internal_id not in resolved]
    if not missing:
        return {internal_id: resolved[internal_id] for internal_id in internal_ids}

    cursor.execute(
        """
        insert into display_id_counters (project, kind, next_value)
        values (%s, %s, 1)
        on conflict (project, kind) do nothing
        """,
        (project, kind),
    )
    cursor.execute(
        """
        select next_value
        from display_id_counters
        where project = %s and kind = %s
        for update
        """,
        (project, kind),
    )
    row = cursor.fetchone()
    next_value = int(row[0]) if row else 1
    prefix = _KIND_PREFIX[kind]
    for internal_id in missing:
        display_id = _format_display_id(prefix, next_value)
        next_value += 1
        cursor.execute(
            """
            insert into artifact_display_ids
              (project, kind, internal_id, display_id)
            values (%s, %s, %s, %s)
            on conflict (internal_id) do update
            set display_id = artifact_display_ids.display_id
            returning display_id
            """,
            (project, kind, internal_id, display_id),
        )
        returned = cursor.fetchone()
        resolved[internal_id] = str(returned[0]) if returned else display_id
    cursor.execute(
        """
        update display_id_counters
        set next_value = %s, updated_at = now()
        where project = %s and kind = %s
        """,
        (next_value, project, kind),
    )
    return {internal_id: resolved[internal_id] for internal_id in internal_ids}


def _resolve_local(project: str, kind: str, internal_ids: list[str]) -> dict[str, str]:
    with _local_lock():
        rows = _read_local_rows()
        resolved = {
            str(row["internal_id"]): str(row["display_id"])
            for row in rows
            if row.get("project") == project
            and row.get("kind") == kind
            and row.get("internal_id") in internal_ids
        }
        next_value = _next_local_value(rows, project, kind)
        prefix = _KIND_PREFIX[kind]
        changed = False
        for internal_id in internal_ids:
            if internal_id in resolved:
                continue
            display_id = _format_display_id(prefix, next_value)
            next_value += 1
            resolved[internal_id] = display_id
            rows.append(
                {
                    "project": project,
                    "kind": kind,
                    "internal_id": internal_id,
                    "display_id": display_id,
                    "written_at": datetime.now(UTC).isoformat(),
                }
            )
            changed = True
        if changed:
            _write_local_rows(rows)
    return {internal_id: resolved[internal_id] for internal_id in internal_ids}


class _local_lock:
    def __init__(self) -> None:
        self._fd: int | None = None

    def __enter__(self) -> None:
        _LOCAL_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30.0
        while True:
            try:
                self._fd = os.open(
                    _LOCAL_LOCK_PATH,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return None
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for {_LOCAL_LOCK_PATH}") from None
                time.sleep(0.05)

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
        with suppress(FileNotFoundError):
            _LOCAL_LOCK_PATH.unlink()


def _read_local_rows() -> list[dict[str, Any]]:
    if not _LOCAL_REGISTRY_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in _LOCAL_REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_local_rows(rows: list[dict[str, Any]]) -> None:
    _LOCAL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOCAL_REGISTRY_PATH.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _next_local_value(rows: list[dict[str, Any]], project: str, kind: str) -> int:
    maximum = 0
    for row in rows:
        if row.get("project") != project or row.get("kind") != kind:
            continue
        match = _DISPLAY_RE.match(str(row.get("display_id", "")))
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def _format_display_id(prefix: str, value: int) -> str:
    return f"{prefix}-{value:0{_DISPLAY_WIDTH}d}"


def _normalize_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace("-", "_")
    if normalized not in _KIND_PREFIX:
        raise ValueError(f"unsupported display-id kind: {kind!r}")
    if normalized in {"requirements", "req"}:
        return "requirement"
    if normalized in {"user_stories", "story"}:
        return "user_story"
    if normalized in {"test_scenarios", "scenario"}:
        return "test_scenario"
    return normalized


def _unique_ids(internal_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in internal_ids:
        internal_id = str(value).strip()
        if internal_id and internal_id not in seen:
            seen.add(internal_id)
            ordered.append(internal_id)
    return ordered
