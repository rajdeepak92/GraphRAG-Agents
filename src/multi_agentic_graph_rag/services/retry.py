"""Bounded retry classification for provider and database operations."""

from __future__ import annotations

from collections.abc import Callable

_TRANSIENT_NAMES = {
    "APITimeoutError",
    "RateLimitError",
    "ServiceUnavailable",
    "SessionExpired",
    "TransientError",
    "OperationalError",
    "ConnectionError",
    "TimeoutError",
}


def is_transient(exc: Exception) -> bool:
    """Classify only known provider/connectivity failures as retryable."""
    return exc.__class__.__name__ in _TRANSIENT_NAMES


def retry_transient_once[T](operation: Callable[[], T]) -> T:
    """Run an operation with at most one retry for classified transient failure."""
    try:
        return operation()
    except Exception as exc:
        if not is_transient(exc):
            raise
    return operation()


__all__ = ["is_transient", "retry_transient_once"]
