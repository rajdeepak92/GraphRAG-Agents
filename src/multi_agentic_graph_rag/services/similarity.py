"""Neutral vector-similarity helpers shared across services.

This module holds provider-neutral numeric helpers (currently cosine
similarity) that several services reuse for bounded embedding recall. It has no
dependency on any generation stage, schema, or human-review path.
"""

from __future__ import annotations

import math


def cosine(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity between two equal-length vectors.

    Args:
        left (list[float]): Left vector required by the operation's typed contract.
        right (list[float]): Right vector required by the operation's typed contract.

    Returns:
        float: Cosine similarity in ``[-1.0, 1.0]``; ``0.0`` when either vector has zero norm.
    """
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
