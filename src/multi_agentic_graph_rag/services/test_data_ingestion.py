"""Deterministic ingestion: document -> validation -> immutable snapshot (§8.7).

Ties the reader, validator, and snapshot identity together. A snapshot is only
published READY when mandatory gates pass; otherwise a precise blocked report is
returned. The snapshot ID is derived from project + workbook checksum +
normalized-record checksum + schema version + decision revision (§8.9).
"""

from __future__ import annotations

from dataclasses import dataclass

from multi_agentic_graph_rag.domain.identifiers import make_test_data_snapshot_id
from multi_agentic_graph_rag.domain.schemas import canonical_checksum
from multi_agentic_graph_rag.domain.test_data_schemas import (
    NormalizedTestData,
    TestDataValidationReport,
)
from multi_agentic_graph_rag.services.test_data_document_reader import RawTestDataDocument
from multi_agentic_graph_rag.services.test_data_validator import validate_test_data


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of ingesting one document for a set of canonical scenarios."""

    report: TestDataValidationReport
    normalized: NormalizedTestData | None

    @property
    def is_ready(self) -> bool:
        return self.report.status == "READY" and self.normalized is not None


def _records_checksum(document: RawTestDataDocument) -> str:
    payload = {
        "records": [record.model_dump(mode="json") for record in document.records],
        "bindings": [binding.model_dump(mode="json") for binding in document.bindings],
    }
    return canonical_checksum(payload)


def ingest_document(
    document: RawTestDataDocument,
    *,
    scenario_ids: set[str],
) -> IngestionResult:
    """Validate a parsed document and, if clean, publish an immutable snapshot."""
    normalized_checksum = _records_checksum(document)
    snapshot_id = make_test_data_snapshot_id(
        project=document.project,
        workbook_checksum=document.workbook_checksum,
        normalized_checksum=normalized_checksum,
        schema_version=document.schema_version,
        decision_revision=document.decision_revision,
    )
    report = validate_test_data(
        project=document.project,
        snapshot_id=snapshot_id,
        schema_version=document.schema_version,
        records=document.records,
        bindings=document.bindings,
        scenario_ids=scenario_ids,
        structural_issues=document.structural_issues,
    )
    if report.status != "READY":
        return IngestionResult(report=report, normalized=None)

    draft = NormalizedTestData.model_construct(
        snapshot_id=snapshot_id,
        project=document.project,
        schema_version=document.schema_version,
        workbook_checksum=document.workbook_checksum,
        decision_revision=document.decision_revision,
        records=document.records,
        bindings=document.bindings,
        checksum="",
    )
    normalized = NormalizedTestData.model_validate(
        {**draft.model_dump(mode="json"), "checksum": canonical_checksum(draft)}
    )
    return IngestionResult(report=report, normalized=normalized)


__all__ = ["IngestionResult", "ingest_document"]
