"""Run-scoped structured logging."""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
_SENSITIVE_KEY = re.compile(r"(password|secret|token|api[_-]?key|dsn)", re.I)
_BODY_KEY = re.compile(r"(document|content|body|text|prompt|payload)", re.I)
_URI_WITH_AUTH = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]*://)([^/@:]+)(?::([^/@]*))?@", re.ASCII)
_DEFAULT_LOG_LIMIT = 512
_DEBUG_LOG_LIMIT = 160
_LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "EXCEPTION": 50,
}


def session_slug(value: str) -> str:
    slug = _SAFE_SLUG.sub("-", value.strip()).strip("-")
    return slug or "ITEM"


def _normalize_level(level: str) -> str:
    normalized = level.strip().upper()
    return normalized if normalized in _LEVEL_ORDER else "INFO"


def _format_timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_flush(handle: Any) -> None:
    handle.flush()
    with suppress(OSError):
        os.fsync(handle.fileno())


def _safe_render(template: str, values: Mapping[str, Any]) -> str:
    class _DefaultDict(dict[str, Any]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_DefaultDict(values))


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    head = max(24, limit - 18)
    return f"{value[:head]}...<redacted len={len(value)}>"


def _redact_scalar(key: str | None, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, str):
        if key and key.lower() == "traceback":
            return value
        if key and _SENSITIVE_KEY.search(key):
            return "<redacted>"
        if key and _BODY_KEY.search(key):
            return _shorten(value, _DEBUG_LOG_LIMIT)
        if _URI_WITH_AUTH.match(value):
            return _URI_WITH_AUTH.sub(r"\1\2:<redacted>@", value)
        return _shorten(value, _DEFAULT_LOG_LIMIT)
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_scalar(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_scalar(key, item) for item in value]
    if isinstance(value, tuple):
        return [_redact_scalar(key, item) for item in value]
    if isinstance(value, set):
        return [_redact_scalar(key, item) for item in value]
    return _shorten(str(value), _DEFAULT_LOG_LIMIT)


def _sanitize_context(context: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        sanitized[str(key)] = _redact_scalar(str(key), value)
    return sanitized


def _caller_frame(skip: int = 2) -> tuple[str, str]:
    frame = sys._getframe(skip)
    module = frame.f_globals.get("__name__", "__main__")
    function = frame.f_code.co_name
    return str(module), function


@dataclass
class _LogSink:
    run_id: str
    project: str
    version: str
    jsonl_path: Path
    log_path: Path
    level: str
    jsonl_handle: Any
    log_handle: Any
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        _atomic_flush(self.jsonl_handle)
        _atomic_flush(self.log_handle)
        self.jsonl_handle.close()
        self.log_handle.close()
        self.closed = True


class RunLogger:
    """Structured logger bound to a single run session."""

    def __init__(
        self,
        jsonl_path: Path,
        log_path: Path,
        *,
        run_id: str,
        project: str,
        version: str,
        level: str = "INFO",
        _sink: _LogSink | None = None,
        _bound: Mapping[str, Any] | None = None,
    ) -> None:
        if _sink is None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _sink = _LogSink(
                run_id=run_id,
                project=project,
                version=version,
                jsonl_path=jsonl_path,
                log_path=log_path,
                level=_normalize_level(level),
                jsonl_handle=jsonl_path.open("a", encoding="utf-8"),
                log_handle=log_path.open("a", encoding="utf-8"),
            )
        self._sink = _sink
        self._bound = dict(_bound or {})

    @property
    def level(self) -> str:
        return self._sink.level

    def set_level(self, level: str) -> None:
        self._sink.level = _normalize_level(level)

    def bind(self, **context: Any) -> RunLogger:
        merged = {**self._bound, **context}
        return RunLogger(
            self._sink.jsonl_path,
            self._sink.log_path,
            run_id=self._sink.run_id,
            project=self._sink.project,
            version=self._sink.version,
            level=self._sink.level,
            _sink=self._sink,
            _bound=merged,
        )

    def debug(self, message: str, /, **context: Any) -> None:
        self._emit("DEBUG", message, **context)

    def info(self, message: str, /, **context: Any) -> None:
        self._emit("INFO", message, **context)

    def warning(self, message: str, /, **context: Any) -> None:
        self._emit("WARNING", message, **context)

    def warn(self, message: str, /, **context: Any) -> None:
        self.warning(message, **context)

    def error(self, message: str, /, **context: Any) -> None:
        self._emit("ERROR", message, **context)

    def exception(
        self,
        message: str,
        /,
        *,
        exc: BaseException | None = None,
        **context: Any,
    ) -> None:
        error = exc or sys.exc_info()[1]
        if error is None:
            self._emit("EXCEPTION", message, **context)
            return
        trace = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ).rstrip()
        context = {**context, "traceback": trace, "exception_type": error.__class__.__name__}
        self._emit("EXCEPTION", message, **context)

    def close(self) -> None:
        self._sink.close()

    def _emit(self, level: str, message: str, /, **context: Any) -> None:
        level = _normalize_level(level)
        if _LEVEL_ORDER[level] < _LEVEL_ORDER[self._sink.level]:
            return

        format_context = {**self._bound, **context}
        record_context = dict(format_context)
        module = str(record_context.pop("module", "") or "")
        function = str(record_context.pop("function", "") or "")
        step = str(record_context.pop("step", "") or "")
        if not module or not function:
            caller_module, caller_function = _caller_frame(skip=3)
            module = module or caller_module
            function = function or caller_function
        rendered_message = _safe_render(message, format_context)
        sanitized_context = _sanitize_context(record_context)

        record = {
            "timestamp": _format_timestamp(),
            "level": level,
            "run_id": self._sink.run_id,
            "project": self._sink.project,
            "version": self._sink.version,
            "step": step,
            "module": module,
            "function": function,
            "message": rendered_message,
            "context": sanitized_context,
        }

        self._sink.jsonl_handle.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._sink.log_handle.write(_format_text_line(record) + "\n")
        if level == "EXCEPTION" and "traceback" in sanitized_context:
            self._sink.log_handle.write(str(sanitized_context["traceback"]) + "\n")
        _atomic_flush(self._sink.jsonl_handle)
        _atomic_flush(self._sink.log_handle)


def _format_text_line(record: Mapping[str, Any]) -> str:
    base = (
        f"{record['timestamp']} {record['level']} "
        f"run_id={record['run_id']} project={record['project']} version={record['version']}"
    )
    step = record.get("step")
    if step:
        base += f" step={step}"
    module = record.get("module")
    function = record.get("function")
    if module or function:
        base += f" {module}:{function}"
    base += f" {record['message']}"
    context = record.get("context")
    if isinstance(context, Mapping) and context:
        base += f" context={json.dumps(context, ensure_ascii=False, default=str)}"
    return base
