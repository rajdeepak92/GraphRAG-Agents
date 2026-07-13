"""Provide run-scoped, zero-secret structured logging and debug flow spans."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Any, Literal

_SAFE_SLUG = re.compile(r"[^A-Za-z0-9._-]+")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:password|passwd|pwd|secret|client_secret|token|access_token|"
    r"refresh_token|api_key|apikey|authorization|bearer|credential|cookie|"
    r"set_cookie|dsn|database_url|connection_string|private_key)(?:$|[_-])",
    re.I,
)
_BODY_KEY = re.compile(
    r"^(?:prompt|system_message|source_text|document_text|document_content|content|body|"
    r"payload|request_payload|response|raw_response|completion|embedding|embeddings)$",
    re.I,
)
_URI_WITH_AUTH = re.compile(
    r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^\s/@:]+)(?::([^\s/@]*))?@",
    re.ASCII,
)
_SENSITIVE_PAIR = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|client[_-]?secret|access[_-]?token|"
    r"refresh[_-]?token|api[_-]?key|apikey|authorization|credential|cookie|"
    r"set[_-]?cookie|dsn|database[_-]?url|connection[_-]?string|private[_-]?key)"
    r"\s*([=:])\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;\]}]+)"
)
_SENSITIVE_QUERY = re.compile(
    r"(?i)([?&](?:password|passwd|pwd|secret|client_secret|token|access_token|"
    r"refresh_token|api_key|apikey|authorization|credential|signature|sig)=)[^&#\s]+"
)
_BEARER = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_PROVIDER_SECRET = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|eyJ[A-Za-z0-9_-]{8,}"
    r"\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
)
_DEFAULT_LOG_LIMIT = 512
_LEVEL_ORDER = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "EXCEPTION": 40,
}
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"]


def session_slug(value: str) -> str:
    """Return a filesystem-safe session label without retaining unsafe characters."""
    slug = _SAFE_SLUG.sub("-", value.strip()).strip("-")
    return slug or "ITEM"


def normalize_log_level(level: str) -> str:
    """Normalize a configured threshold, falling back safely to ``INFO``."""
    normalized = level.strip().upper()
    return normalized if normalized in _LEVEL_ORDER else "INFO"


def _format_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(UTC).isoformat()


def _atomic_flush(handle: Any) -> None:
    """Flush a log handle and best-effort synchronize it to durable storage."""
    handle.flush()
    with suppress(OSError):
        os.fsync(handle.fileno())


def _shorten(value: str, limit: int) -> str:
    """Bound diagnostic text while reporting the original character count."""
    if len(value) <= limit:
        return value
    head = max(24, limit - 24)
    return f"{value[:head]}...<truncated len={len(value)}>"


def sanitize_text(value: str, *, limit: int = _DEFAULT_LOG_LIMIT) -> str:
    """Redact credential patterns and bound arbitrary diagnostic text.

    Args:
        value (str): Untrusted message, exception, or traceback text.
        limit (int): Maximum retained diagnostic length after redaction.

    Returns:
        str: Text safe for console and run-log sinks.

    Security:
        Redacts authenticated URIs, sensitive key/value pairs, query credentials,
        authorization bearer values, and common provider-token formats.
    """
    sanitized = _URI_WITH_AUTH.sub(r"\1\2:<redacted>@", value)
    sanitized = _SENSITIVE_QUERY.sub(r"\1<redacted>", sanitized)
    sanitized = _BEARER.sub("Bearer <redacted>", sanitized)
    sanitized = _PROVIDER_SECRET.sub("<redacted>", sanitized)
    sanitized = _SENSITIVE_PAIR.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", sanitized
    )
    return _shorten(sanitized, limit)


def sanitized_exception_summary(error: BaseException) -> str:
    """Return content-free exception diagnostics suitable for external logs.

    Exception messages are untrusted because drivers and providers may echo DSNs,
    headers, prompts, source excerpts, or response payloads. The summary therefore
    exposes only the exception type, message size, and a correlation fingerprint.
    """
    message = str(error)
    fingerprint = hashlib.sha256(message.encode("utf-8")).hexdigest()[:16]
    return f"{error.__class__.__name__}: <message redacted len={len(message)} sha256={fingerprint}>"


def _sanitized_traceback(error: BaseException) -> str:
    """Render traceback frames without source lines, local values, or exception text."""
    frames: list[str] = ["Traceback (sanitized):"]
    current = error.__traceback__
    while current is not None:
        code = current.tb_frame.f_code
        frames.append(
            f'  File "{Path(code.co_filename).name}", line {current.tb_lineno}, in {code.co_name}'
        )
        current = current.tb_next
    frames.append(sanitized_exception_summary(error))
    return "\n".join(frames)


def _redact_scalar(key: str | None, value: Any) -> Any:
    """Convert one context value into an allowlist-oriented safe representation."""
    if value is None:
        return None
    if key and _SENSITIVE_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Path):
        return sanitize_text(str(value))
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if key and _BODY_KEY.search(key):
        length = len(value) if isinstance(value, (str, bytes, list, tuple, set, Mapping)) else None
        return "<redacted>" if length is None else f"<redacted len={length}>"
    if isinstance(value, str):
        limit = 4096 if key and key.lower() == "traceback" else _DEFAULT_LOG_LIMIT
        return sanitize_text(value, limit=limit)
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_scalar(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_scalar(key, item) for item in value]
    # Arbitrary __str__/Pydantic repr output is not an approved logging contract.
    return f"<{value.__class__.__name__}>"


def sanitize_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively sanitize structured context before it reaches any sink."""
    sanitized: dict[str, Any] = {}
    for key, value in context.items():
        if value is None:
            continue
        sanitized[str(key)] = _redact_scalar(str(key), value)
    return sanitized


def _safe_render(template: str, values: Mapping[str, Any]) -> str:
    """Render a message only with already-sanitized formatting values."""

    class _DefaultDict(dict[str, Any]):
        """Preserve unresolved format fields instead of raising during logging."""

        def __missing__(self, key: str) -> str:
            """Return the original placeholder for an absent formatting key."""
            return "{" + key + "}"

    try:
        rendered = template.format_map(_DefaultDict(values))
    except (KeyError, ValueError, AttributeError, IndexError):
        rendered = template
    return sanitize_text(rendered)


def _caller_frame(skip: int = 2) -> tuple[str, str]:
    """Return the application module and function at a known stack depth."""
    frame = sys._getframe(skip)
    module = frame.f_globals.get("__name__", "__main__")
    return str(module), frame.f_code.co_name


@dataclass
class _LogSink:
    """Own the shared file handles and immutable run identity for bound loggers."""

    run_id: str
    project: str
    version: str
    jsonl_path: Path
    log_path: Path
    level: str
    jsonl_handle: Any
    log_handle: Any
    closed: bool = False
    owned_exceptions: list[BaseException] | None = None

    def close(self) -> None:
        """Flush and close both run-log handles exactly once."""
        if self.closed:
            return
        _atomic_flush(self.jsonl_handle)
        _atomic_flush(self.log_handle)
        self.jsonl_handle.close()
        self.log_handle.close()
        self.closed = True


class _DebugSpan:
    """Emit safe start, completion, and propagation events for one function boundary."""

    def __init__(
        self,
        logger: RunLogger,
        *,
        module: str,
        function: str,
        step: str,
        operation: str,
        context: Mapping[str, Any],
    ) -> None:
        """Initialize a span without inspecting function arguments or return values."""
        self._logger = logger
        self._module = module
        self._function = function
        self._step = step
        self._operation = operation
        self._context = dict(context)
        self._started_at = 0.0
        self._enabled = logger.is_enabled("DEBUG")

    def __enter__(self) -> _DebugSpan:
        """Start timing and emit ``function.started`` when DEBUG is enabled."""
        if self._enabled:
            self._started_at = perf_counter()
            self._logger.debug(
                "function.started",
                module=self._module,
                function=self._function,
                step=self._step,
                operation=self._operation,
                status="started",
                **self._context,
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> Literal[False]:
        """Emit the span outcome and preserve exception propagation semantics."""
        if not self._enabled:
            return False
        duration_ms = round((perf_counter() - self._started_at) * 1000, 3)
        if exc is None:
            self._logger.debug(
                "function.completed",
                module=self._module,
                function=self._function,
                step=self._step,
                operation=self._operation,
                duration_ms=duration_ms,
                status="completed",
                **self._context,
            )
        else:
            self._logger.debug(
                "function.propagated_failure",
                module=self._module,
                function=self._function,
                step=self._step,
                operation=self._operation,
                duration_ms=duration_ms,
                status="propagated_failure",
                exception_type=exc.__class__.__name__,
                error_summary=sanitized_exception_summary(exc),
                **self._context,
            )
        return False


class RunLogger:
    """Write sanitized structured records for one command run.

    Bound logger instances share file handles and threshold state with their parent.
    The logger owns no business resources and never inspects arguments, return values,
    prompts, source text, or model payloads.
    """

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
        """Open or share the per-run text and JSONL sinks."""
        if _sink is None:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _sink = _LogSink(
                run_id=run_id,
                project=project,
                version=version,
                jsonl_path=jsonl_path,
                log_path=log_path,
                level=normalize_log_level(level),
                jsonl_handle=jsonl_path.open("a", encoding="utf-8"),
                log_handle=log_path.open("a", encoding="utf-8"),
                owned_exceptions=[],
            )
        self._sink = _sink
        self._bound = dict(_bound or {})

    @property
    def level(self) -> str:
        """Return the active threshold shared by all bound loggers."""
        return self._sink.level

    def is_enabled(self, level: str) -> bool:
        """Return whether a record at ``level`` passes the current threshold."""
        normalized = normalize_log_level(level)
        return _LEVEL_ORDER[normalized] >= _LEVEL_ORDER[self._sink.level]

    def set_level(self, level: str) -> None:
        """Set the shared threshold, normalizing unsupported values to ``INFO``."""
        self._sink.level = normalize_log_level(level)

    def bind(self, **context: Any) -> RunLogger:
        """Return a lightweight logger view with additional structured context."""
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

    def span(
        self,
        *,
        step: str,
        operation: str,
        module: str | None = None,
        function: str | None = None,
        **context: Any,
    ) -> _DebugSpan:
        """Create a low-overhead DEBUG span for a meaningful function boundary."""
        if not module or not function:
            caller_module, caller_function = _caller_frame(skip=2)
            module = module or caller_module
            function = function or caller_function
        return _DebugSpan(
            self,
            module=module,
            function=function,
            step=step,
            operation=operation,
            context=context,
        )

    def debug(self, message: str, /, **context: Any) -> None:
        """Emit a DEBUG diagnostic when the configured threshold permits it."""
        self._emit("DEBUG", message, **context)

    def info(self, message: str, /, **context: Any) -> None:
        """Emit an INFO workflow milestone when the threshold permits it."""
        self._emit("INFO", message, **context)

    def warning(self, message: str, /, **context: Any) -> None:
        """Emit a WARNING for a recoverable or degraded operation."""
        self._emit("WARNING", message, **context)

    def warn(self, message: str, /, **context: Any) -> None:
        """Emit a WARNING through the deprecated backward-compatible alias."""
        self.warning(message, **context)

    def error(self, message: str, /, **context: Any) -> None:
        """Emit an ERROR for a terminal operation without an active traceback."""
        self._emit("ERROR", message, **context)

    def exception(
        self,
        message: str,
        /,
        *,
        exc: BaseException | None = None,
        **context: Any,
    ) -> None:
        """Emit one sanitized terminal failure with its traceback at the owning boundary."""
        error = exc or sys.exc_info()[1]
        if error is None:
            self._emit("EXCEPTION", message, **context)
            return
        owned = self._sink.owned_exceptions
        if owned is not None and any(item is error for item in owned):
            self.debug(
                "function.propagated_failure",
                **{
                    **context,
                    "exception_type": error.__class__.__name__,
                    "error_summary": sanitized_exception_summary(error),
                    "status": "propagated_failure",
                },
            )
            return
        if owned is not None:
            owned.append(error)
        trace = _sanitized_traceback(error)
        context = {
            **context,
            "traceback": sanitize_text(trace, limit=4096),
            "exception_type": error.__class__.__name__,
            "error_summary": sanitized_exception_summary(error),
        }
        self._emit("EXCEPTION", message, **context)

    def raw_block(self, *, title: str, body: str, also_stderr: bool = False) -> None:
        """Record only a fingerprint and size for a legacy raw diagnostic block.

        The raw body is intentionally never emitted. ``also_stderr`` is accepted for
        backward compatibility but cannot enable payload output.
        """
        del also_stderr
        self.debug(
            "raw_block.suppressed",
            step="diagnostic_payload",
            title=sanitize_text(title, limit=160),
            body_length=len(body),
            body_fingerprint=hashlib.sha256(body.encode("utf-8")).hexdigest()[:16],
            status="suppressed",
        )

    def close(self) -> None:
        """Close the shared per-run sinks."""
        self._sink.close()

    def _emit(self, level: LogLevel, message: str, /, **context: Any) -> None:
        """Sanitize and write one record consistently to stderr, text, and JSONL."""
        if not self.is_enabled(level):
            return

        format_context = {**self._bound, **context}
        sanitized_format_context = sanitize_context(format_context)
        record_context = dict(sanitized_format_context)
        module = str(record_context.pop("module", "") or "")
        function = str(record_context.pop("function", "") or "")
        step = str(record_context.pop("step", "") or "")
        if not module or not function:
            caller_module, caller_function = _caller_frame(skip=3)
            module = module or caller_module
            function = function or caller_function
        rendered_message = _safe_render(message, sanitized_format_context)

        record = {
            "timestamp": _format_timestamp(),
            "level": level,
            "run_id": sanitize_text(self._sink.run_id),
            "project": sanitize_text(self._sink.project),
            "version": sanitize_text(self._sink.version),
            "step": sanitize_text(step),
            "module": sanitize_text(module),
            "function": sanitize_text(function),
            "message": rendered_message,
            "context": record_context,
        }
        text_line = _format_text_line(record)
        self._sink.jsonl_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._sink.log_handle.write(text_line + "\n")
        if level == "EXCEPTION" and "traceback" in record_context:
            self._sink.log_handle.write(str(record_context["traceback"]) + "\n")
        sys.stderr.write(_format_console_line(record) + "\n")
        sys.stderr.flush()
        _atomic_flush(self._sink.jsonl_handle)
        _atomic_flush(self._sink.log_handle)


def _format_text_line(record: Mapping[str, Any]) -> str:
    """Render one ANSI-free text record from sanitized structured fields."""
    base = (
        f"{record['timestamp']} {record['level']} "
        f"run_id={record['run_id']} project={record['project']} version={record['version']}"
    )
    if record.get("step"):
        base += f" step={record['step']}"
    if record.get("module") or record.get("function"):
        base += f" module={record['module']} function={record['function']}"
    base += f" {record['message']}"
    context = record.get("context")
    if isinstance(context, Mapping) and context:
        base += f" context={json.dumps(context, ensure_ascii=False)}"
    return base


def _format_console_line(record: Mapping[str, Any]) -> str:
    """Render concise INFO output while retaining full DEBUG/failure context."""
    if record.get("level") != "INFO":
        return _format_text_line(record)
    base = (
        f"{record['timestamp']} INFO run_id={record['run_id']} "
        f"project={record['project']} version={record['version']}"
    )
    if record.get("step"):
        base += f" step={record['step']}"
    return f"{base} {record['message']}"
