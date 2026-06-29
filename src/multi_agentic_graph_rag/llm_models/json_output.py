"""JSON cleanup helpers for model outputs."""

from __future__ import annotations

import json
from typing import Any, cast


def parse_json_object(raw: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from a provider response."""
    start = raw.find("{")
    if start < 0:
        raise ValueError("model output did not contain a JSON object")

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
                return cast(dict[str, Any], json.loads(raw[start : index + 1]))

    raise ValueError("model output contained an unterminated JSON object")
