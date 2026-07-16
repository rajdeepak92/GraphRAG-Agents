"""Secret-safe diagnostic helpers used by model adapters."""

from __future__ import annotations

import hashlib


def sanitized_exception_summary(exc: BaseException) -> str:
    """Return a content-free exception fingerprint suitable for logs."""
    text = str(exc)
    fingerprint = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"{exc.__class__.__name__}(length={len(text)}, fingerprint={fingerprint})"


__all__ = ["sanitized_exception_summary"]
