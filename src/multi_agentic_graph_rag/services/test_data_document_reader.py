"""Deterministic reader for the approved test-data document (plan §8.7 steps 1-6).

The canonical, portable representation is ``normalized-test-data.json`` (§8.1):
a manifest plus typed records and scenario bindings. Reading an XLSX workbook is
a thin optional front-end — when ``openpyxl`` is unavailable it raises a clear
error rather than guessing. Executable cells must reject formulas and the reader
must open workbooks with ``keep_vba=False``/``keep_links=False`` (§8.7).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from multi_agentic_graph_rag.domain.schemas import canonical_checksum
from multi_agentic_graph_rag.domain.test_data_schemas import (
    ScenarioDataBinding,
    TestDataRecord,
    ValidationIssue,
)


class TestDataDocumentError(ValueError):
    """Raised when the test-data document cannot be structurally parsed."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class RawTestDataDocument:
    """Structurally-parsed document prior to semantic validation."""

    project: str
    schema_version: str
    workbook_checksum: str
    decision_revision: str
    records: list[TestDataRecord]
    bindings: list[ScenarioDataBinding]
    structural_issues: list[ValidationIssue]


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise TestDataDocumentError(f"{context} is missing required key '{key}'")
    return mapping[key]


def read_document(source: Path | dict[str, Any]) -> RawTestDataDocument:
    """Parse a normalized JSON test-data document into typed models (§8.7 s.5-6)."""
    if isinstance(source, Path):
        if source.suffix.lower() in {".xlsx", ".xlsm"}:
            raise TestDataDocumentError(
                "XLSX ingestion requires 'openpyxl'; export the approved workbook to "
                "normalized-test-data.json or install the optional dependency"
            )
        data = json.loads(source.read_text(encoding="utf-8"))
    else:
        data = source
    if not isinstance(data, dict):
        raise TestDataDocumentError("test-data document must be a JSON object")

    manifest = _require(data, "manifest", "document")
    if not isinstance(manifest, dict):
        raise TestDataDocumentError("document 'manifest' must be an object")

    structural_issues: list[ValidationIssue] = []
    records = _parse_records(data.get("records", []), structural_issues)
    bindings = _parse_bindings(data.get("bindings", []), structural_issues)

    return RawTestDataDocument(
        project=str(_require(manifest, "project", "manifest")),
        schema_version=str(_require(manifest, "schema_version", "manifest")),
        workbook_checksum=str(_require(manifest, "workbook_checksum", "manifest")),
        decision_revision=str(manifest.get("decision_revision", "r0")),
        records=records,
        bindings=bindings,
        structural_issues=structural_issues,
    )


def _parse_records(rows: list[Any], issues: list[ValidationIssue]) -> list[TestDataRecord]:
    records: list[TestDataRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(
                ValidationIssue(
                    issue_code="STRUCT_RECORD_NOT_OBJECT",
                    severity="ERROR",
                    message=f"record at position {index} is not an object",
                    row=index + 1,
                )
            )
            continue
        payload = row.get("payload", {})
        prepared = {**row, "payload_checksum": canonical_checksum({"payload": payload})}
        try:
            records.append(TestDataRecord.model_validate(prepared))
        except ValidationError as exc:
            issues.append(
                ValidationIssue(
                    issue_code="STRUCT_RECORD_INVALID",
                    severity="ERROR",
                    message=_first_error(exc),
                    record_id=str(row.get("record_id")) if row.get("record_id") else None,
                    sheet=str(row.get("source_sheet")) if row.get("source_sheet") else None,
                    row=_int_or_none(row.get("source_row")),
                )
            )
    return records


def _int_or_none(value: Any) -> int | None:
    return int(value) if str(value if value is not None else "").isdigit() else None


def _parse_bindings(rows: list[Any], issues: list[ValidationIssue]) -> list[ScenarioDataBinding]:
    bindings: list[ScenarioDataBinding] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(
                ValidationIssue(
                    issue_code="STRUCT_BINDING_NOT_OBJECT",
                    severity="ERROR",
                    message=f"binding at position {index} is not an object",
                    row=index + 1,
                )
            )
            continue
        try:
            bindings.append(ScenarioDataBinding.model_validate(row))
        except ValidationError as exc:
            issues.append(
                ValidationIssue(
                    issue_code="STRUCT_BINDING_INVALID",
                    severity="ERROR",
                    message=_first_error(exc),
                    record_id=str(row.get("binding_id")) if row.get("binding_id") else None,
                )
            )
    return bindings


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "validation error"
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ()))
    return f"{location}: {first.get('msg', 'invalid')}"


__all__ = ["RawTestDataDocument", "TestDataDocumentError", "read_document"]
