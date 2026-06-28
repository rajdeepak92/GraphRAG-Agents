"""PostgreSQL-backed public identifier formatting."""

from __future__ import annotations


def format_fact_id(sequence_value: int) -> str:
    """Format a permanent public fact ID."""
    if sequence_value < 1:
        raise ValueError("fact sequence value must be >= 1.")

    return f"FACT-{sequence_value:06d}"


def format_requirement_id(sequence_value: int) -> str:
    """Format a permanent public requirement ID."""
    if sequence_value < 1:
        raise ValueError("requirement sequence value must be >= 1.")

    return f"REQ-{sequence_value:06d}"
