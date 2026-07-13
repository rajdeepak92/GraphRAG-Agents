"""Deterministic cumulative per-project master projection (Phase C).

Each pipeline stage keeps one cumulative master per project: a materialized
projection of the normalized PostgreSQL rows (which remain the source of truth),
mirrored to a stable per-project JSON file. The materializer is a *pure,
deterministic* function of the normalized rows — reads are ordered by permanent
id and the checksum covers only reproducible content — so re-materializing
unchanged rows yields a byte-identical payload and drift detection never reports
false positives.

The checksum deliberately excludes run/time/revision metadata
(``run_id``/``updated_at``/``payload_revision``): those are recorded as table
columns and envelope metadata, not as reproducible content.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from multi_agentic_graph_rag.domain.schemas import StageMasterArtifact

MASTER_STAGES: tuple[str, ...] = ("requirements", "user_stories", "test_scenarios")

STAGE_SCHEMA_VERSIONS: dict[str, str] = {
    "requirements": "requirements-master-1.0",
    "user_stories": "user-stories-master-1.0",
    "test_scenarios": "test-scenarios-master-1.0",
}

MASTER_FILENAMES: dict[str, str] = {
    "requirements": "requirements.json",
    "user_stories": "user_stories.json",
    "test_scenarios": "test_scenarios.json",
}


def master_content_checksum(
    *,
    schema_version: str,
    stage: str,
    project: str,
    document_id: str,
    records: list[dict[str, Any]],
) -> str:
    """SHA-256 over the reproducible content only (never run/time metadata)."""
    content = {
        "artifact_schema_version": schema_version,
        "stage": stage,
        "project": project,
        "document_id": document_id,
        "records": records,
    }
    raw = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _derive_document_id(records: list[dict[str, Any]], fallback: str) -> str:
    """Derive document id deterministically within the active scope.

    Args:
        records (list[dict[str, Any]]): Ordered records processed without changing their identities.
        fallback (str): Fallback required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    for record in records:
        document_id = record.get("document_id")
        if document_id:
            return str(document_id)
    return fallback


def materialize_master(
    store: Any,
    *,
    project: str,
    stage: str,
    document_id: str = "",
    current_document_version_id: str = "",
    run_id: str = "",
    cursor: Any | None = None,
) -> StageMasterArtifact:
    """Build the cumulative master for ``project``/``stage`` from normalized rows.

    ``document_id`` is derived from the records when present (keeping the checksum
    a pure function of the rows) and only falls back to the passed value for an
    empty master. When ``cursor`` is supplied the read runs inside the caller's
    transaction so same-transaction materialization sees just-written rows.
    """
    records = store.load_master_records(project=project, stage=stage, cursor=cursor)
    resolved_document_id = _derive_document_id(records, document_id)
    schema_version = STAGE_SCHEMA_VERSIONS[stage]
    checksum = master_content_checksum(
        schema_version=schema_version,
        stage=stage,
        project=project,
        document_id=resolved_document_id,
        records=records,
    )
    return StageMasterArtifact(
        artifact_schema_version=schema_version,
        stage=stage,  # type: ignore[arg-type]
        project=project,
        document_id=resolved_document_id,
        current_document_version_id=current_document_version_id,
        run_id=run_id,
        checksum=checksum,
        record_count=len(records),
        records=records,
    )


def recompute_checksum(master: StageMasterArtifact) -> str:
    """Recompute the content checksum of a loaded master (for drift detection)."""
    return master_content_checksum(
        schema_version=master.artifact_schema_version,
        stage=master.stage,
        project=master.project,
        document_id=master.document_id,
        records=master.records,
    )
