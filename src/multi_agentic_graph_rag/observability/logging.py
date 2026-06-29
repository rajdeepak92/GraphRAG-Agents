"""Structured runtime logging."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

T = TypeVar("T")


class RunLogger:
    def __init__(self, log_dir: Path, run_id: str) -> None:
        self.path = log_dir / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, **payload: Any) -> None:
        payload.setdefault("timestamp", datetime.now(UTC).isoformat())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str) + "\n")

    def node(self, *, project: str, version: str, step: str, fn: Callable[[], T]) -> T:
        started = perf_counter()
        self.event(project=project, version=version, step=step, status="started")
        try:
            result = fn()
        except Exception as exc:
            self.event(
                project=project,
                version=version,
                step=step,
                status="failed",
                duration_seconds=round(perf_counter() - started, 6),
                error_category=exc.__class__.__name__,
                error=str(exc),
            )
            raise
        self.event(
            project=project,
            version=version,
            step=step,
            status="completed",
            duration_seconds=round(perf_counter() - started, 6),
        )
        return result
