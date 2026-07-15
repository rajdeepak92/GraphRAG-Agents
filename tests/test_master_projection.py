"""Phase C: cumulative per-project master projection + JSON sync."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.postgres import PostgresStore
from multi_agentic_graph_rag.domain.schemas import (
    UserStoryBuildResult,
    UserStoryRecord,
    UserStoryStatement,
)
from multi_agentic_graph_rag.services.artifact_mirror import ArtifactMirror
from multi_agentic_graph_rag.services.master_projection import (
    materialize_master,
    recompute_checksum,
)
from multi_agentic_graph_rag.services.user_story_builder import project_user_story_artifact


class MasterProjectionDeterminismTests(unittest.TestCase):
    # 1. THE GATE: materializing the same rows twice is byte-identical.
    def test_materialize_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PostgresStore(_settings(Path(temp_dir)))
            store.persist_user_story_artifact(_stories(("US-2", "US-1")), "path", "RUN-1")

            first = materialize_master(
                store, project="PROJECT", stage="user_stories", run_id="RUN-1"
            )
            second = materialize_master(
                store, project="PROJECT", stage="user_stories", run_id="RUN-2"
            )

        # run_id differs but content checksum is identical, and records are ordered.
        self.assertEqual(first.checksum, second.checksum)
        self.assertEqual(first.record_count, 2)
        self.assertEqual([r["story_id"] for r in first.records], ["US-1", "US-2"])
        self.assertEqual(recompute_checksum(first), first.checksum)


class MasterProjectionSyncTests(unittest.TestCase):
    def _mirror_and_store(self, root: Path) -> tuple[ArtifactMirror, PostgresStore]:
        store = PostgresStore(_settings(root))
        return ArtifactMirror(store), store

    def _persist_stories(
        self, mirror: ArtifactMirror, root: Path, story_ids: tuple[str, ...]
    ) -> Path:
        run_dir = root / "generated" / "PROJECT" / "us" / "RUN-1"
        mirror.persist_committed_artifact(
            artifact=_stories(story_ids),
            artifact_path=run_dir / "user_stories.json",
            run_id="RUN-1",
        )
        return mirror.master_file_path("PROJECT", "user_stories")

    # Persist writes both the master row (in-tx) and the stable master file.
    def test_persist_writes_master_row_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, store = self._mirror_and_store(root)
            master_path = self._persist_stories(mirror, root, ("US-1",))

            master = store.get_stage_master(project="PROJECT", stage="user_stories")
            assert master is not None
            file_payload = json.loads(master_path.read_text("utf-8"))

            self.assertEqual(master.record_count, 1)
            self.assertTrue(master_path.exists())
            self.assertEqual(file_payload["checksum"], master.checksum)

    # V1 -> V2: new records accumulate and supersede status is preserved.
    def test_v1_v2_cumulative_and_supersede(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, store = self._mirror_and_store(root)
            self._persist_stories(mirror, root, ("US-1",))

            superseded = _story("US-1")
            superseded.status = "superseded"
            superseded.doc_version = "V2"
            new = _story("US-2")
            records = {"US-1": superseded, "US-2": new}
            artifact = project_user_story_artifact(
                project="PROJECT",
                document_id="DOC",
                document_version_id="DOC-v2",
                doc_version="V2",
                records=records,
            )
            mirror.persist_committed_artifact(
                artifact=UserStoryBuildResult(
                    artifact=artifact, records=records, coverage={"REQ-1": ["US-1", "US-2"]}
                ),
                artifact_path=root / "generated" / "PROJECT" / "us" / "RUN-2" / "user_stories.json",
                run_id="RUN-2",
            )
            master = store.get_stage_master(project="PROJECT", stage="user_stories")

        assert master is not None
        by_id = {r["story_id"]: r for r in master.records}
        self.assertEqual(set(by_id), {"US-1", "US-2"})
        self.assertEqual(by_id["US-1"]["status"], "superseded")
        self.assertEqual(by_id["US-2"]["status"], "active")

    def test_missing_master_file_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, _ = self._mirror_and_store(root)
            master_path = self._persist_stories(mirror, root, ("US-1",))
            master_path.unlink()
            self.assertFalse(master_path.exists())

            mirror.reconcile_masters(project="PROJECT")
            self.assertTrue(master_path.exists())

    def test_stale_master_file_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, store = self._mirror_and_store(root)
            master_path = self._persist_stories(mirror, root, ("US-1",))
            payload = json.loads(master_path.read_text("utf-8"))
            payload["checksum"] = "stale-does-not-match"
            master_path.write_text(json.dumps(payload), encoding="utf-8")

            mirror.reconcile_masters(project="PROJECT")
            repaired = json.loads(master_path.read_text("utf-8"))
            master = store.get_stage_master(project="PROJECT", stage="user_stories")
        assert master is not None
        self.assertEqual(repaired["checksum"], master.checksum)

    def test_corrupt_master_file_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, _ = self._mirror_and_store(root)
            master_path = self._persist_stories(mirror, root, ("US-1",))
            master_path.write_text("{ not json", encoding="utf-8")

            mirror.reconcile_masters(project="PROJECT")
            recovered = json.loads(master_path.read_text("utf-8"))
        self.assertEqual(recovered["record_count"], 1)

    # Master payload disagreeing with normalized rows -> rebuilt from normalized + warning.
    def test_normalized_master_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, store = self._mirror_and_store(root)
            self._persist_stories(mirror, root, ("US-1", "US-2"))

            tampered = store.get_stage_master(project="PROJECT", stage="user_stories")
            assert tampered is not None
            bad = tampered.model_dump(mode="json")
            bad["records"] = []
            bad["record_count"] = 0
            bad["checksum"] = "tampered-checksum-does-not-match-normalized"
            store._upsert_local(
                "user_stories_master",
                "PROJECT",
                {"kind": "user_stories_master", "project": "PROJECT", "master": bad},
            )

            report = mirror.reconcile_masters(project="PROJECT")
            rebuilt = store.get_stage_master(project="PROJECT", stage="user_stories")

        assert rebuilt is not None
        self.assertEqual(rebuilt.record_count, 2)
        self.assertTrue(any("user_stories_master rebuilt" in w for w in report.warnings))

    def test_payload_revision_increments_on_repersist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, store = self._mirror_and_store(root)
            self._persist_stories(mirror, root, ("US-1",))
            self._persist_stories(mirror, root, ("US-1", "US-2"))
            master = store.get_stage_master(project="PROJECT", stage="user_stories")
        assert master is not None
        self.assertEqual(master.payload_revision, 2)

    def test_atomic_write_leaves_no_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, _ = self._mirror_and_store(root)
            master_path = self._persist_stories(mirror, root, ("US-1",))
            leftovers = list(master_path.parent.glob("*.tmp"))
            self.assertEqual(leftovers, [])
            self.assertTrue(master_path.exists())

    # First materialization backfills a project that has rows but no master row.
    def test_backfill_via_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mirror, store = self._mirror_and_store(root)
            # Simulate a pre-Phase-C project: normalized rows exist, no master row.
            store._upsert_local(
                "user_story",
                "US-9",
                {
                    "kind": "user_story",
                    "project": "PROJECT",
                    "user_story": _story("US-9").model_dump(mode="json"),
                },
            )
            self.assertIsNone(store.get_stage_master(project="PROJECT", stage="user_stories"))

            mirror.reconcile_masters(project="PROJECT")
            master = store.get_stage_master(project="PROJECT", stage="user_stories")
        assert master is not None
        self.assertEqual(master.record_count, 1)

    # Requirements master preserves multiple revisions per requirement and a fact
    # shared across requirements (many-to-many), materialized deterministically.
    def test_requirements_master_history_and_shared_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = PostgresStore(_settings(Path(temp_dir)))
            _seed_requirement(store, "REQ-1", ["REQREV-1a", "REQREV-1b"], shared_fact="FACT-1")
            _seed_requirement(store, "REQ-2", ["REQREV-2a"], shared_fact="FACT-1")

            master = materialize_master(store, project="PROJECT", stage="requirements")
            first = master.checksum
            again = materialize_master(store, project="PROJECT", stage="requirements").checksum

        by_id = {r["requirement_id"]: r for r in master.records}
        self.assertEqual(set(by_id), {"REQ-1", "REQ-2"})
        self.assertEqual(len(by_id["REQ-1"]["revisions"]), 2)  # history preserved
        fact_owners = {
            r["requirement_id"]
            for r in master.records
            for rev in r["revisions"]
            for ev in rev.get("evidence", [])
            if "FACT-1" in ev.get("fact_ids", [])
        }
        self.assertEqual(fact_owners, {"REQ-1", "REQ-2"})  # many-to-many
        self.assertEqual(first, again)  # deterministic


def _seed_requirement(
    store: PostgresStore, requirement_id: str, revision_ids: list[str], *, shared_fact: str
) -> None:
    store._upsert_local(
        "requirement",
        requirement_id,
        {
            "kind": "requirement",
            "project": "PROJECT",
            "document_id": "DOC",
            "requirement_id": requirement_id,
            "source_req_id": None,
            "id_generation_type": "generated",
            "confidence": 1.0,
            "status": "active",
            "active_revision_id": revision_ids[-1],
            "first_seen_document_version_id": "DOC-v1",
        },
    )
    for revision_id in revision_ids:
        store._upsert_local(
            "requirement_revision",
            revision_id,
            {
                "kind": "requirement_revision",
                "project": "PROJECT",
                "document_id": "DOC",
                "document_version_id": "DOC-v1",
                "status": "active",
                "requirement": {
                    "requirement_id": requirement_id,
                    "revision_id": revision_id,
                    "statement": f"Statement for {revision_id}",
                    "requirement_type": "Functional Requirement",
                    "priority": "High",
                    "status": "active",
                    "evidence": [
                        {
                            "evidence_id": f"REQEVID-{revision_id}",
                            "document_version_id": "DOC-v1",
                            "chunk_id": "CHUNK-1",
                            "quote": "quote",
                            "start_char": 0,
                            "end_char": 5,
                            "fact_ids": [shared_fact],
                        }
                    ],
                },
            },
        )


def _settings(root: Path) -> AppSettings:
    return AppSettings(
        paths=PathsSettings(
            project_root=root,
            global_cache_dir=root / ".global_cache",
            documents_inbox_dir=root / "documents" / "inbox",
            generated_requirements_dir=root / "generated",
            chroma_persist_dir=root / "runtime" / "databases" / "chroma",
            runtime_staging_dir=root / "runtime" / "staging",
            runtime_logs_dir=root / "runtime" / "logs",
            runtime_locks_dir=root / "runtime" / "locks",
        ),
        postgres=PostgresSettings(mode="local_json", local_path=root / "runtime" / "pg.jsonl"),
        neo4j=Neo4jSettings(mode="local_json", local_path=root / "runtime" / "neo4j.jsonl"),
    )


def _story(story_id: str) -> UserStoryRecord:
    return UserStoryRecord(
        story_id=story_id,
        requirement_id="REQ-1",
        requirement_revision_id="REQREV-1",
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        title=f"Story {story_id}",
        priority="Medium",
        persona="Operator",
        user_story=UserStoryStatement(as_a="operator", i_want="x", so_that="y"),
        acceptance_criteria=["Given a, when b, then c."],
        confidence=0.9,
    )


def _stories(story_ids: tuple[str, ...]) -> UserStoryBuildResult:
    records = {sid: _story(sid) for sid in story_ids}
    artifact = project_user_story_artifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id="DOC-v1",
        doc_version="V1",
        records=records,
    )
    return UserStoryBuildResult(
        artifact=artifact, records=records, coverage={"REQ-1": list(story_ids)}
    )


if __name__ == "__main__":
    unittest.main()
