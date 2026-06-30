"""JSON cleanup helpers for model outputs."""

from __future__ import annotations

import json
import re
from typing import Any, cast


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract the final valid balanced JSON object from a provider response."""
    cleaned = _cleanup_model_output(raw)
    candidates: list[dict[str, Any]] = []
    start = cleaned.find("{")
    saw_json_start = False
    saw_incomplete_json = False
    while start >= 0:
        saw_json_start = True
        candidate, end, incomplete = _parse_balanced_object(cleaned, start)
        if candidate is not None:
            candidates.append(candidate)
        if incomplete:
            saw_incomplete_json = True
            break
        start = cleaned.find("{", max(start + 1, end))
    if candidates:
        return candidates[-1]
    if saw_json_start and saw_incomplete_json:
        raise ValueError(
            f"incomplete_json: model output started a JSON object but did not close it "
            f"preview={sanitized_output_preview(raw)}"
        )
    raise ValueError(
        f"model output did not contain a valid JSON object preview={sanitized_output_preview(raw)}"
    )


def sanitized_output_preview(raw: str, *, limit: int = 400) -> str:
    preview = re.sub(r"\s+", " ", raw).strip()
    if len(preview) > limit:
        preview = preview[:limit] + "..."
    return repr(preview)


def _cleanup_model_output(raw: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", raw)
    text = re.sub(r"(?is)<think>.*?(?=\{)", "", text)
    fenced = re.findall(r"(?is)```(?:json)?\s*(.*?)```", text)
    if fenced:
        return "\n".join(fenced)
    return text


def _parse_balanced_object(raw: str, start: int) -> tuple[dict[str, Any] | None, int, bool]:
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(raw[start:], start=start):
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(raw[start : index + 1])
                except json.JSONDecodeError:
                    return None, index + 1, False
                if isinstance(parsed, dict):
                    return cast(dict[str, Any], parsed), index + 1, False
                return None, index + 1, False

    return None, len(raw), depth > 0 or in_string
