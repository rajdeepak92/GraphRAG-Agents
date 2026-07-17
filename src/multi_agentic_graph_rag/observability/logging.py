"""Runtime logging: console tracing, raw LLM capture, and safe fingerprints."""

from __future__ import annotations

import hashlib
import logging
import sys
from typing import Any

_ROOT_NAME = "multi_agentic_graph_rag"
_configured = False


def sanitized_exception_summary(exc: BaseException) -> str:
    """Return a content-free exception fingerprint suitable for logs."""
    text = str(exc)
    fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{exc.__class__.__name__}(length={len(text)}, fingerprint={fingerprint})"


def configure_logging(level: str = "INFO") -> None:
    """Configure a single readable stderr handler for the package logger tree."""
    global _configured
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(resolved)
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a package-scoped logger; children inherit the configured handler."""
    return logging.getLogger(name)


class RunLogger:
    """Adapter exposing the structured/raw-block interface model adapters expect."""

    def __init__(self, logger: logging.Logger, *, capture_raw: bool = True) -> None:
        self._logger = logger
        self._capture_raw = capture_raw

    @staticmethod
    def _render(event: str, fields: dict[str, Any]) -> str:
        parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        return f"{event} {parts}".rstrip()

    def info(self, event: str, **fields: Any) -> None:
        self._logger.info(self._render(event, _without_raw(fields)))

    def warning(self, event: str, **fields: Any) -> None:
        self._logger.warning(self._render(event, _without_raw(fields)))
        raw = fields.get("raw_response")
        if raw and self._capture_raw:
            self.raw_block(title=event, body=str(raw))

    def raw_block(self, *, title: str, body: str, also_stderr: bool = False) -> None:
        del also_stderr
        if self._capture_raw:
            self._logger.info("RAW %s:\n%s", title, body)


def _without_raw(fields: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in fields.items() if key != "raw_response"}


__all__ = [
    "RunLogger",
    "configure_logging",
    "get_logger",
    "sanitized_exception_summary",
]
