"""Polluted-lineage detection and repair (Increment 4/9)."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.identifiers import is_requirement_uuid7
from multi_agentic_graph_rag.domain.schemas import (
    RequirementArtifact,
    SourceTrace,
    VerifiedRequirement,
)
from multi_agentic_graph_rag.services.requirement_repair import (
    analyze_requirement_artifact,
    apply_repair,
    migrate_legacy_catalog_payload,
)


def _requirement(requirement_id: str, revision_id: str, statement: str, status: str) -> dict:
    return VerifiedRequirement(
        requirement_id=requirement_id,
        revision_id=revision_id,
        statement=statement,
        normalized_statement=statement.lower(),
        requirement_type="Functional Requirement",
        priority="Medium",
        status=status,  # type: ignore[arg-type]
        fact_ids=["FACT-1"],
        source_trace=SourceTrace(
            chunk_id="CHUNK-1", quote=statement, start_char=0, end_char=len(statement)
        ),
    ).model_dump(mode="json")


def _artifact(requirements: list[dict]) -> RequirementArtifact:
    return RequirementArtifact.model_validate(
        {
            "artifact_schema_version": "2.1",
            "project": "SIIMCS",
            "document_id": "DOC",
            "document_version_id": "DV-1",
            "version": "1.0",
            "source_checksum": "chk",
            "facts": [],
            "requirements": requirements,
        }
    )


class RepairTests(unittest.TestCase):
    def test_local_project_repair_is_idempotent_and_remaps_downstream_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = _local_store(root)
            rows = [
                {
                    "kind": "requirement_artifact",
                    "_local_key": "DV-1",
                    "artifact_path": str(root / "requirements.json"),
                    "artifact": _canonical_payload(),
                },
                {
                    "kind": "user_story",
                    "_local_key": "US-1",
                    "project": "PROJECT",
                    "requirement_revision_id": "REV-1",
                    "requirement_id": "REQ-LEGACY",
                    "payload": {
                        "project": "PROJECT",
                        "story_id": "US-1",
                        "revision_id": "REV-1",
                        "requirement_id": "REQ-LEGACY",
                    },
                },
            ]
            store.settings.postgres.local_path.parent.mkdir(parents=True, exist_ok=True)
            store.settings.postgres.local_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            store.repair_project_identities(project="PROJECT", apply=True)
            repaired = store._read_local_rows()
            artifact = next(
                row["artifact"] for row in repaired if row["kind"] == "requirement_artifact"
            )
            requirement = artifact["requirements"][0]
            story = next(row for row in repaired if row["kind"] == "user_story")
            self.assertTrue(is_requirement_uuid7(requirement["requirement_id"]))
            self.assertTrue(is_requirement_uuid7(requirement["revision_id"]))
            self.assertTrue(is_requirement_uuid7(requirement["evidence"][0]["evidence_id"]))
            self.assertEqual(story["payload"]["revision_id"], requirement["revision_id"])
            self.assertEqual(story["payload"]["requirement_id"], requirement["requirement_id"])
            second = store.repair_project_identities(project="PROJECT", apply=True)
            self.assertFalse(second["revision_id_remap"])
            self.assertFalse(second["evidence_id_remap"])

    def test_document_identity_lock_serializes_equivalent_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            identity_file = root / "identity.txt"
            entered: list[str] = []
            guard = threading.Lock()

            def allocate(name: str) -> str:
                with _local_store(root).document_identity_lock("PROJECT", "DOC"):
                    with guard:
                        entered.append(name)
                    if identity_file.exists():
                        return identity_file.read_text(encoding="utf-8")
                    time.sleep(0.05)
                    value = "REQ-01900000-0000-7000-8000-000000000001"
                    identity_file.write_text(value, encoding="utf-8")
                    return value

            with ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(allocate, ("first", "second")))
            self.assertEqual(values[0], values[1])
            self.assertEqual(len(entered), 2)

    def test_legacy_catalog_migrates_to_one_canonical_row_with_all_evidence(self) -> None:
        base = {
            "requirement_uid": "REQ-OLD",
            "revision_id": "REV-OLD",
            "confidence": 0.9,
            "requirement_text": "Sensor-1 shall poll every 5 seconds.",
            "requirement_type": "Functional Requirement",
            "priority": "Medium",
            "status": "Active",
        }
        migrated = migrate_legacy_catalog_payload(
            {
                "artifact_schema_version": "4.0-catalog",
                "project": "SIIMCS",
                "document_id": "DOC",
                "document_version_id": "DV-1",
                "doc_version": "1.0",
                "requirements": [
                    {**base, "chunk_id": "CHUNK-1", "fact_id": "FACT-1"},
                    {**base, "chunk_id": "CHUNK-2", "fact_id": "FACT-2"},
                ],
            }
        )
        self.assertEqual(migrated.artifact_schema_version, "5.0-requirements")
        self.assertEqual(len(migrated.requirements), 1)
        self.assertEqual(len(migrated.requirements[0].evidence), 2)

    def test_polluted_lineage_is_detected_and_split(self) -> None:
        # Sensor-1 and Sensor-2 wrongly merged under one lineage (distinct signatures).
        artifact = _artifact(
            [
                _requirement(
                    "REQ-MERGED", "REV-1", "Sensor-1 shall poll every 5 seconds.", "active"
                ),
                _requirement(
                    "REQ-MERGED", "REV-2", "Sensor-2 shall poll every 5 seconds.", "superseded"
                ),
            ]
        )
        report = analyze_requirement_artifact(artifact)
        self.assertEqual(report.polluted_lineages, 1)

        repaired = apply_repair(artifact, report)
        ids = {r.requirement_id for r in repaired.requirements}
        self.assertEqual(len(ids), 2)  # split into two lineages
        # Each corrected lineage has exactly one active revision.
        for requirement_id in ids:
            group = [r for r in repaired.requirements if r.requirement_id == requirement_id]
            self.assertEqual(sum(1 for r in group if r.status == "active"), 1)

    def test_clean_legacy_lineage_is_migrated_and_apply_is_idempotent(self) -> None:
        # Same obligation, threshold changed: one lineage, two revisions, NOT polluted.
        artifact = _artifact(
            [
                _requirement("REQ-1", "REV-1", "The controller shall trip at 70C.", "superseded"),
                _requirement("REQ-1", "REV-2", "The controller shall trip at 80C.", "active"),
            ]
        )
        report = analyze_requirement_artifact(artifact)
        self.assertEqual(report.polluted_lineages, 0)
        repaired = apply_repair(artifact, report)
        ids = {r.requirement_id for r in repaired.requirements}
        self.assertEqual(len(ids), 1)
        self.assertTrue(all(is_requirement_uuid7(value) for value in ids))
        self.assertTrue(all(is_requirement_uuid7(row.revision_id) for row in repaired.requirements))
        # Re-analyzing the repaired artifact is a no-op (idempotent).
        second = analyze_requirement_artifact(repaired)
        self.assertEqual(second.polluted_lineages, 0)
        self.assertEqual(second.id_remap, {})
        self.assertEqual(second.revision_id_remap, {})


def _local_store(root: Path) -> PostgresStore:
    return PostgresStore(
        AppSettings(
            paths=PathsSettings(
                project_root=root,
                global_cache_dir=root / ".cache",
                documents_inbox_dir=root / "documents",
                generated_requirements_dir=root / "generated",
                chroma_persist_dir=root / "chroma",
                runtime_staging_dir=root / "runtime" / "staging",
                runtime_logs_dir=root / "runtime" / "logs",
                runtime_locks_dir=root / "runtime" / "locks",
            ),
            postgres=PostgresSettings(
                mode="local_json", local_path=root / "runtime" / "postgres.jsonl"
            ),
            neo4j=Neo4jSettings(mode="local_json", local_path=root / "runtime" / "neo4j.jsonl"),
        )
    )


def _canonical_payload() -> dict[str, object]:
    statement = "The system shall import files."
    return {
        "artifact_schema_version": "5.0-requirements",
        "project": "PROJECT",
        "document_id": "DOC",
        "document_version_id": "DV-1",
        "doc_version": "1.0",
        "requirements": [
            {
                "requirement_id": "REQ-LEGACY",
                "revision_id": "REV-1",
                "semantic_signature": "functional requirement::the system shall import files",
                "confidence": 0.9,
                "requirement_text": statement,
                "requirement_type": "Functional Requirement",
                "priority": "Medium",
                "status": "Active",
                "evidence": [
                    {
                        "evidence_id": "EVID-1",
                        "document_version_id": "DV-1",
                        "chunk_id": "CHUNK-1",
                        "fact_ids": ["FACT-1"],
                        "quote": statement,
                        "start_char": 0,
                        "end_char": len(statement),
                    }
                ],
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
