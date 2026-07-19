"""Snapshot-scoped Stage-4 code/test-data graph with verified activation.

Publication is deliberately two phase: content is written under a BUILDING
``FrameworkSnapshot`` and only a separate verification step may mark it READY
and active.  A failed publication therefore never replaces the prior READY
snapshot used by the next scenario.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.code_graph_schemas import (
    CodeExtractionResult,
    CodeSymbol,
    FrameworkSnapshot,
)
from multi_agentic_graph_rag.domain.codegen_schemas import Stage4TestCaseRecord
from multi_agentic_graph_rag.domain.test_data_schemas import NormalizedTestData


class SnapshotVerificationError(RuntimeError):
    """A BUILDING snapshot does not represent all expected generated files."""


class CodeGraphStore:
    """Persist/query code snapshots and Stage-4 traceability projections."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def _local(self) -> bool:
        return self.settings.neo4j.mode == "local_json"

    @property
    def _local_path(self) -> Path:
        return self.settings.stage4.code_graph_local_path

    def check(self) -> str:
        if self._local:
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS code-graph local_json path={self._local_path}"
        with self._session() as session:
            session.run("RETURN 1").consume()
        return "PASS code-graph connectivity"

    def ensure_schema(self) -> None:
        if self._local:
            return
        statements = (
            "CREATE CONSTRAINT code_snapshot_id IF NOT EXISTS "
            "FOR (s:FrameworkSnapshot) REQUIRE s.snapshot_id IS UNIQUE",
            "CREATE CONSTRAINT code_file_id IF NOT EXISTS "
            "FOR (f:CodeFile) REQUIRE (f.snapshot_id, f.relative_path) IS UNIQUE",
            "CREATE CONSTRAINT code_symbol_id IF NOT EXISTS "
            "FOR (n:CodeSymbol) REQUIRE (n.snapshot_id, n.symbol_id) IS UNIQUE",
            "CREATE INDEX code_symbol_fqn IF NOT EXISTS "
            "FOR (n:CodeSymbol) ON (n.snapshot_id, n.fqn)",
            "CREATE CONSTRAINT test_data_snapshot_id IF NOT EXISTS "
            "FOR (s:TestDataSnapshot) REQUIRE s.snapshot_id IS UNIQUE",
            "CREATE CONSTRAINT test_data_record_id IF NOT EXISTS "
            "FOR (r:TestDataRecord) REQUIRE (r.snapshot_id, r.record_id) IS UNIQUE",
            "CREATE CONSTRAINT scenario_binding_id IF NOT EXISTS "
            "FOR (b:ScenarioDataBinding) REQUIRE (b.snapshot_id, b.binding_id) IS UNIQUE",
            "CREATE CONSTRAINT stage4_test_case_id IF NOT EXISTS "
            "FOR (t:TestCase) REQUIRE t.tc_id IS UNIQUE",
            "CREATE CONSTRAINT code_artifact_id IF NOT EXISTS "
            "FOR (a:CodeArtifact) REQUIRE (a.snapshot_id, a.relative_path) IS UNIQUE",
        )
        with self._session() as session:
            for statement in statements:
                session.run(statement).consume()

    # --- framework snapshot publication ---------------------------------

    def begin_snapshot(
        self,
        snapshot: FrameworkSnapshot,
        result: CodeExtractionResult,
        *,
        test_data_snapshot_id: str | None = None,
    ) -> FrameworkSnapshot:
        """Persist a BUILDING snapshot and its graph content without activating it."""
        if snapshot.snapshot_id != result.snapshot_id:
            raise ValueError("snapshot/result snapshot_id mismatch")
        building = snapshot.model_copy(
            update={
                "status": "building",
                "active": False,
                "test_data_snapshot_id": test_data_snapshot_id or snapshot.test_data_snapshot_id,
            }
        )
        if self._local:
            self._upsert_local(
                "code_snapshot", building.snapshot_id, building.model_dump(mode="json")
            )
            for file in result.files:
                self._upsert_local(
                    "code_file",
                    f"{file.snapshot_id}:{file.relative_path}",
                    file.model_dump(mode="json"),
                )
            for symbol in result.symbols:
                self._upsert_local(
                    "code_symbol",
                    f"{symbol.snapshot_id}:{symbol.symbol_id}",
                    symbol.model_dump(mode="json"),
                )
            for edge in result.edges:
                self._upsert_local(
                    "code_edge",
                    f"{edge.snapshot_id}:{edge.edge_id}",
                    edge.model_dump(mode="json"),
                )
            if building.test_data_snapshot_id:
                self._upsert_local(
                    "relation",
                    f"{building.snapshot_id}:USES_TEST_DATA:{building.test_data_snapshot_id}",
                    {
                        "source": building.snapshot_id,
                        "relation": "USES_TEST_DATA",
                        "target": building.test_data_snapshot_id,
                        "snapshot_id": building.snapshot_id,
                    },
                )
            return building
        self._begin_neo4j(building, result)
        return building

    def verify_snapshot(self, snapshot_id: str, expected_file_hashes: dict[str, str]) -> None:
        """Require every changed Python/Robot file and exact hash before READY."""
        actual = self._file_hashes(snapshot_id)
        missing = sorted(set(expected_file_hashes) - set(actual))
        mismatched = sorted(
            path
            for path, expected in expected_file_hashes.items()
            if path in actual and actual[path] != expected
        )
        if missing or mismatched:
            raise SnapshotVerificationError(
                f"snapshot {snapshot_id} verification failed; "
                f"missing={missing}, mismatched={mismatched}"
            )

    def snapshot_file_hashes(self, snapshot_id: str) -> dict[str, str]:
        """Return the published file manifest used by deterministic verification."""
        return self._file_hashes(snapshot_id)

    def activate_snapshot(self, snapshot_id: str) -> FrameworkSnapshot:
        """Mark a verified snapshot READY+active and retain every older READY row."""
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise SnapshotVerificationError(f"unknown snapshot: {snapshot_id}")
        if snapshot.status != "building":
            if snapshot.status == "ready" and snapshot.active:
                return snapshot
            raise SnapshotVerificationError(
                f"snapshot {snapshot_id} is {snapshot.status}, expected building"
            )
        ready = snapshot.model_copy(update={"status": "ready", "active": True})
        if self._local:
            rows = self._local_rows()
            for row in rows:
                if row.get("_kind") != "code_snapshot":
                    continue
                payload = row.get("payload", {})
                if payload.get("repository_id") == snapshot.repository_id:
                    payload["active"] = False
            _write_rows(self._local_path, rows)
            self._upsert_local("code_snapshot", snapshot_id, ready.model_dump(mode="json"))
            return ready
        with self._session() as session:
            session.run(
                """
                MATCH (target:FrameworkSnapshot {snapshot_id: $id})
                MATCH (other:FrameworkSnapshot {repository_id: target.repository_id})
                SET other.active = false
                SET target.status = 'ready', target.active = true
                """,
                id=snapshot_id,
            ).consume()
        return ready

    def mark_snapshot_failed(self, snapshot_id: str) -> None:
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot is None or snapshot.status == "ready":
            return
        failed = snapshot.model_copy(update={"status": "failed", "active": False})
        if self._local:
            self._upsert_local("code_snapshot", snapshot_id, failed.model_dump(mode="json"))
            return
        with self._session() as session:
            session.run(
                "MATCH (s:FrameworkSnapshot {snapshot_id: $id}) "
                "SET s.status='failed', s.active=false",
                id=snapshot_id,
            ).consume()

    def publish_snapshot(
        self, snapshot: FrameworkSnapshot, result: CodeExtractionResult
    ) -> FrameworkSnapshot:
        """Compatibility convenience: BUILDING -> verify all files -> READY."""
        self.begin_snapshot(
            snapshot,
            result,
            test_data_snapshot_id=snapshot.test_data_snapshot_id,
        )
        self.verify_snapshot(
            snapshot.snapshot_id,
            {file.relative_path: file.content_hash for file in result.files},
        )
        return self.activate_snapshot(snapshot.snapshot_id)

    def get_snapshot(self, snapshot_id: str) -> FrameworkSnapshot | None:
        if self._local:
            row = self._read_local("code_snapshot", snapshot_id)
            return FrameworkSnapshot.model_validate(row) if row else None
        with self._session() as session:
            record = session.run(
                "MATCH (s:FrameworkSnapshot {snapshot_id: $id}) RETURN s", id=snapshot_id
            ).single()
        return FrameworkSnapshot.model_validate(dict(record["s"])) if record else None

    def get_active_ready_snapshot(
        self, repository_id: str | None = None
    ) -> FrameworkSnapshot | None:
        if self._local:
            matches = []
            for row in self._local_rows():
                if row.get("_kind") != "code_snapshot":
                    continue
                snapshot = FrameworkSnapshot.model_validate(row["payload"])
                if (
                    snapshot.status == "ready"
                    and snapshot.active
                    and (repository_id is None or snapshot.repository_id == repository_id)
                ):
                    matches.append(snapshot)
            return matches[-1] if matches else None
        query = "MATCH (s:FrameworkSnapshot {status:'ready', active:true})"
        params: dict[str, Any] = {}
        if repository_id is not None:
            query += " WHERE s.repository_id = $repository_id"
            params["repository_id"] = repository_id
        query += " RETURN s LIMIT 1"
        with self._session() as session:
            record = session.run(query, **params).single()
        return FrameworkSnapshot.model_validate(dict(record["s"])) if record else None

    def find_ready_snapshot(
        self, *, canonical_path: str, filesystem_checksum: str
    ) -> FrameworkSnapshot | None:
        """Find the READY snapshot for exact on-disk content, active or retained."""
        if self._local:
            for row in reversed(self._local_rows()):
                if row.get("_kind") != "code_snapshot":
                    continue
                snapshot = FrameworkSnapshot.model_validate(row["payload"])
                if (
                    snapshot.status == "ready"
                    and snapshot.canonical_path == canonical_path
                    and snapshot.filesystem_checksum == filesystem_checksum
                ):
                    return snapshot
            return None
        with self._session() as session:
            record = session.run(
                """
                MATCH (s:FrameworkSnapshot {status:'ready', canonical_path:$path,
                    filesystem_checksum:$checksum})
                RETURN s ORDER BY s.active DESC LIMIT 1
                """,
                path=canonical_path,
                checksum=filesystem_checksum,
            ).single()
        return FrameworkSnapshot.model_validate(dict(record["s"])) if record else None

    # --- test-data and accepted-TC projection -----------------------------

    def publish_test_data(self, normalized: NormalizedTestData) -> None:
        snapshot_payload = {
            "snapshot_id": normalized.snapshot_id,
            "project": normalized.project,
            "schema_version": normalized.schema_version,
            "workbook_checksum": normalized.workbook_checksum,
            "checksum": normalized.checksum,
            "status": "ready",
        }
        if self._local:
            self._upsert_local("test_data_snapshot", normalized.snapshot_id, snapshot_payload)
            for record in normalized.records:
                self._upsert_local(
                    "test_data_record",
                    f"{normalized.snapshot_id}:{record.record_id}",
                    {"snapshot_id": normalized.snapshot_id, **record.model_dump(mode="json")},
                )
                self._local_relation(
                    normalized.snapshot_id, "HAS_RECORD", record.record_id, normalized.snapshot_id
                )
            for binding in normalized.bindings:
                payload = {"snapshot_id": normalized.snapshot_id, **binding.model_dump(mode="json")}
                self._upsert_local(
                    "scenario_data_binding",
                    f"{normalized.snapshot_id}:{binding.binding_id}",
                    payload,
                )
                self._local_relation(
                    binding.scenario_id,
                    "HAS_DATA_BINDING",
                    binding.binding_id,
                    normalized.snapshot_id,
                )
                self._local_relation(
                    binding.binding_id,
                    "FOR_PROFILE",
                    binding.execution_profile_id,
                    normalized.snapshot_id,
                )
                for record_id in _binding_record_ids(binding.model_dump(mode="json")):
                    self._local_relation(
                        binding.binding_id,
                        "USES_RECORD",
                        record_id,
                        normalized.snapshot_id,
                    )
            return
        self._publish_test_data_neo4j(normalized, snapshot_payload)

    def publish_test_case(self, record: Stage4TestCaseRecord) -> None:
        """Project an accepted TC and preserve every story/requirement edge."""
        if record.status != "ACCEPTED" or record.framework_snapshot_id is None:
            raise ValueError("only accepted test cases with a framework snapshot may be projected")
        payload = record.model_dump(mode="json")
        if self._local:
            self._upsert_local("test_case", str(record.tc_id), payload)
            self._local_relation(str(record.tc_id), "IMPLEMENTS", record.scenario_id)
            for story_id in record.story_ids:
                self._local_relation(str(record.tc_id), "TRACES_TO", story_id)
            for requirement_id in record.requirement_ids:
                self._local_relation(str(record.tc_id), "VERIFIES", requirement_id)
            self._local_relation(str(record.tc_id), "USES_TEST_DATA", record.test_data_snapshot_id)
            self._local_relation(str(record.tc_id), "GENERATED_IN", record.framework_snapshot_id)
            for path, file_hash in record.generated_file_hashes.items():
                self._upsert_local(
                    "code_artifact",
                    f"{record.framework_snapshot_id}:{path}",
                    {
                        "snapshot_id": record.framework_snapshot_id,
                        "relative_path": path,
                        "content_hash": file_hash,
                    },
                )
                self._local_relation(str(record.tc_id), "GENERATED_FILE", path)
            return
        self._publish_test_case_neo4j(record)

    # --- code queries -----------------------------------------------------

    def get_symbol(self, snapshot_id: str, symbol_id: str) -> CodeSymbol | None:
        if self._local:
            row = self._read_local("code_symbol", f"{snapshot_id}:{symbol_id}")
            return CodeSymbol.model_validate(row) if row else None
        with self._session() as session:
            record = session.run(
                "MATCH (n:CodeSymbol {snapshot_id: $s, symbol_id: $i}) RETURN n",
                s=snapshot_id,
                i=symbol_id,
            ).single()
        return CodeSymbol.model_validate(dict(record["n"])) if record else None

    def search_symbols(
        self,
        snapshot_id: str,
        query: str,
        *,
        kinds: list[str] | None = None,
        limit: int = 20,
    ) -> list[CodeSymbol]:
        needle = query.strip().casefold()
        allowed = set(kinds) if kinds else None
        matches: list[CodeSymbol] = []
        for symbol in self._iter_symbols(snapshot_id):
            if allowed is not None and symbol.kind not in allowed:
                continue
            if needle in symbol.fqn.casefold():
                matches.append(symbol)
            if len(matches) >= limit:
                break
        return matches

    def get_neighbors(
        self,
        snapshot_id: str,
        symbol_id: str,
        *,
        relations: list[str] | None = None,
        depth: int = 1,
    ) -> list[CodeSymbol]:
        depth = max(1, min(depth, 4))
        adjacency: dict[str, list[str]] = {}
        for edge in self._iter_edges(snapshot_id):
            if relations and edge["relation"] not in relations:
                continue
            adjacency.setdefault(edge["source_symbol_id"], []).append(edge["target_symbol_id"])
        seen: set[str] = {symbol_id}
        frontier: deque[tuple[str, int]] = deque([(symbol_id, 0)])
        reached: list[str] = []
        while frontier:
            current, level = frontier.popleft()
            if level >= depth:
                continue
            for target in adjacency.get(current, []):
                if target not in seen:
                    seen.add(target)
                    reached.append(target)
                    frontier.append((target, level + 1))
        resolved = [self.get_symbol(snapshot_id, target) for target in reached]
        return [symbol for symbol in resolved if symbol is not None]

    def delete_snapshot(self, snapshot_id: str) -> int:
        """Explicit maintenance only; active or accepted-TC snapshots are retained."""
        snapshot = self.get_snapshot(snapshot_id)
        if snapshot and snapshot.active:
            raise ValueError("cannot delete the active READY snapshot")
        if self._local:
            if any(
                row.get("_kind") == "relation"
                and row.get("payload", {}).get("relation") == "GENERATED_IN"
                and row.get("payload", {}).get("target") == snapshot_id
                for row in self._local_rows()
            ):
                raise ValueError("cannot delete a snapshot referenced by an accepted TC")
            rows = self._local_rows()
            kept = [row for row in rows if row.get("snapshot_id") != snapshot_id]
            removed = len(rows) - len(kept)
            _write_rows(self._local_path, kept)
            return removed
        with self._session() as session:
            referenced = session.run(
                "MATCH (:TestCase)-[:GENERATED_IN]->(s:FrameworkSnapshot {snapshot_id:$s}) "
                "RETURN count(s) AS n",
                s=snapshot_id,
            ).single()
            if referenced and int(referenced["n"]) > 0:
                raise ValueError("cannot delete a snapshot referenced by an accepted TC")
            summary = session.run(
                "MATCH (n) WHERE n.snapshot_id = $s DETACH DELETE n", s=snapshot_id
            ).consume()
            return int(summary.counters.nodes_deleted)

    # --- Neo4j internals --------------------------------------------------

    def _begin_neo4j(self, snapshot: FrameworkSnapshot, result: CodeExtractionResult) -> None:
        with self._session() as session:
            session.run(
                "MERGE (s:FrameworkSnapshot {snapshot_id: $id}) SET s += $props",
                id=snapshot.snapshot_id,
                props=snapshot.model_dump(mode="json"),
            ).consume()
            for file in result.files:
                session.run(
                    "MERGE (f:CodeFile {snapshot_id: $s, relative_path: $p}) SET f += $props",
                    s=file.snapshot_id,
                    p=file.relative_path,
                    props=file.model_dump(mode="json"),
                ).consume()
            for symbol in result.symbols:
                session.run(
                    "MERGE (n:CodeSymbol {snapshot_id: $s, symbol_id: $i}) SET n += $props",
                    s=symbol.snapshot_id,
                    i=symbol.symbol_id,
                    props=symbol.model_dump(mode="json"),
                ).consume()
            for edge in result.edges:
                session.run(
                    """
                    MATCH (a:CodeSymbol {snapshot_id: $s, symbol_id: $src})
                    MATCH (b:CodeSymbol {snapshot_id: $s, symbol_id: $tgt})
                    MERGE (a)-[r:CODE_EDGE {edge_id: $eid}]->(b)
                    SET r += $props
                    """,
                    s=edge.snapshot_id,
                    src=edge.source_symbol_id,
                    tgt=edge.target_symbol_id,
                    eid=edge.edge_id,
                    props=edge.model_dump(mode="json"),
                ).consume()
            if snapshot.test_data_snapshot_id:
                session.run(
                    """
                    MATCH (f:FrameworkSnapshot {snapshot_id:$framework})
                    MATCH (t:TestDataSnapshot {snapshot_id:$test_data})
                    MERGE (f)-[:USES_TEST_DATA]->(t)
                    """,
                    framework=snapshot.snapshot_id,
                    test_data=snapshot.test_data_snapshot_id,
                ).consume()

    def _file_hashes(self, snapshot_id: str) -> dict[str, str]:
        if self._local:
            return {
                str(row["payload"]["relative_path"]): str(row["payload"]["content_hash"])
                for row in self._local_rows()
                if row.get("_kind") == "code_file" and row.get("snapshot_id") == snapshot_id
            }
        with self._session() as session:
            return {
                str(record["path"]): str(record["hash"])
                for record in session.run(
                    "MATCH (f:CodeFile {snapshot_id:$s}) "
                    "RETURN f.relative_path AS path, f.content_hash AS hash",
                    s=snapshot_id,
                )
            }

    def _publish_test_data_neo4j(
        self, normalized: NormalizedTestData, snapshot_payload: dict[str, Any]
    ) -> None:
        with self._session() as session:
            session.run(
                "MERGE (s:TestDataSnapshot {snapshot_id:$id}) SET s += $props",
                id=normalized.snapshot_id,
                props=snapshot_payload,
            ).consume()
            for record in normalized.records:
                session.run(
                    """
                    MATCH (s:TestDataSnapshot {snapshot_id:$snapshot})
                    MERGE (r:TestDataRecord {snapshot_id:$snapshot, record_id:$record_id})
                    SET r += $props
                    MERGE (s)-[:HAS_RECORD]->(r)
                    """,
                    snapshot=normalized.snapshot_id,
                    record_id=record.record_id,
                    props=record.model_dump(mode="json"),
                ).consume()
            for binding in normalized.bindings:
                session.run(
                    """
                    MERGE (scenario:Scenario {scenario_id:$scenario})
                    MERGE (profile:ExecutionProfile {profile_id:$profile})
                    MERGE (b:ScenarioDataBinding {snapshot_id:$snapshot, binding_id:$binding})
                    SET b += $props
                    MERGE (scenario)-[:HAS_DATA_BINDING]->(b)
                    MERGE (b)-[:FOR_PROFILE]->(profile)
                    """,
                    scenario=binding.scenario_id,
                    profile=binding.execution_profile_id,
                    snapshot=normalized.snapshot_id,
                    binding=binding.binding_id,
                    props=binding.model_dump(mode="json"),
                ).consume()
                for record_id in _binding_record_ids(binding.model_dump(mode="json")):
                    session.run(
                        """
                        MATCH (b:ScenarioDataBinding {snapshot_id:$snapshot,binding_id:$binding})
                        MATCH (r:TestDataRecord {snapshot_id:$snapshot,record_id:$record})
                        MERGE (b)-[:USES_RECORD]->(r)
                        """,
                        snapshot=normalized.snapshot_id,
                        binding=binding.binding_id,
                        record=record_id,
                    ).consume()

    def _publish_test_case_neo4j(self, record: Stage4TestCaseRecord) -> None:
        with self._session() as session:
            session.run(
                "MERGE (t:TestCase {tc_id:$id}) SET t += $props",
                id=record.tc_id,
                props=record.model_dump(mode="json"),
            ).consume()
            session.run(
                """
                MATCH (t:TestCase {tc_id:$id})
                MERGE (s:Scenario {scenario_id:$scenario})
                MERGE (t)-[:IMPLEMENTS]->(s)
                """,
                id=record.tc_id,
                scenario=record.scenario_id,
            ).consume()
            for story_id in record.story_ids:
                session.run(
                    "MATCH (t:TestCase {tc_id:$id}) "
                    "MERGE (s:UserStory {story_id:$target}) "
                    "MERGE (t)-[:TRACES_TO]->(s)",
                    id=record.tc_id,
                    target=story_id,
                ).consume()
            for requirement_id in record.requirement_ids:
                session.run(
                    "MATCH (t:TestCase {tc_id:$id}) "
                    "MERGE (r:Requirement {requirement_id:$target}) "
                    "MERGE (t)-[:VERIFIES]->(r)",
                    id=record.tc_id,
                    target=requirement_id,
                ).consume()
            session.run(
                """
                MATCH (t:TestCase {tc_id:$id})
                MATCH (d:TestDataSnapshot {snapshot_id:$data})
                MATCH (f:FrameworkSnapshot {snapshot_id:$framework})
                MERGE (t)-[:USES_TEST_DATA]->(d)
                MERGE (t)-[:GENERATED_IN]->(f)
                """,
                id=record.tc_id,
                data=record.test_data_snapshot_id,
                framework=record.framework_snapshot_id,
            ).consume()
            for path, file_hash in record.generated_file_hashes.items():
                session.run(
                    """
                    MATCH (t:TestCase {tc_id:$id})
                    MERGE (a:CodeArtifact {snapshot_id:$snapshot, relative_path:$path})
                    SET a.content_hash=$hash
                    MERGE (t)-[:GENERATED_FILE]->(a)
                    """,
                    id=record.tc_id,
                    snapshot=record.framework_snapshot_id,
                    path=path,
                    hash=file_hash,
                ).consume()

    # --- shared/local internals ------------------------------------------

    def _iter_symbols(self, snapshot_id: str) -> Iterator[CodeSymbol]:
        if self._local:
            for row in self._local_rows():
                if row.get("_kind") == "code_symbol" and row.get("snapshot_id") == snapshot_id:
                    yield CodeSymbol.model_validate(row["payload"])
            return
        with self._session() as session:
            for record in session.run(
                "MATCH (n:CodeSymbol {snapshot_id: $s}) RETURN n", s=snapshot_id
            ):
                yield CodeSymbol.model_validate(dict(record["n"]))

    def _iter_edges(self, snapshot_id: str) -> list[dict[str, Any]]:
        if self._local:
            return [
                row["payload"]
                for row in self._local_rows()
                if row.get("_kind") == "code_edge" and row.get("snapshot_id") == snapshot_id
            ]
        with self._session() as session:
            return [
                dict(record["r"])
                for record in session.run(
                    "MATCH ()-[r:CODE_EDGE {snapshot_id: $s}]->() RETURN r", s=snapshot_id
                )
            ]

    def _driver(self) -> Any:
        from neo4j import GraphDatabase

        return GraphDatabase.driver(
            self.settings.neo4j.uri,
            auth=(self.settings.neo4j.username, self.settings.neo4j.password),
        )

    @contextmanager
    def _session(self) -> Iterator[Any]:
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            yield session

    def _local_rows(self) -> list[dict[str, Any]]:
        if not self._local_path.exists():
            return []
        return [
            json.loads(line)
            for line in self._local_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        rows = self._local_rows()
        row = {
            "_kind": kind,
            "_key": key,
            "snapshot_id": payload.get("snapshot_id"),
            "payload": payload,
        }
        for index, existing in enumerate(rows):
            if existing.get("_kind") == kind and existing.get("_key") == key:
                rows[index] = row
                break
        else:
            rows.append(row)
        _write_rows(self._local_path, rows)

    def _read_local(self, kind: str, key: str) -> dict[str, Any] | None:
        for row in self._local_rows():
            if row.get("_kind") == kind and row.get("_key") == key:
                payload = row.get("payload")
                return dict(payload) if isinstance(payload, dict) else None
        return None

    def _local_relation(
        self, source: str, relation: str, target: str, snapshot_id: str | None = None
    ) -> None:
        self._upsert_local(
            "relation",
            f"{source}:{relation}:{target}",
            {
                "source": source,
                "relation": relation,
                "target": target,
                "snapshot_id": snapshot_id,
            },
        )


def _binding_record_ids(payload: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in payload.items():
        if key.endswith("_id") and key not in {
            "binding_id",
            "scenario_id",
            "execution_profile_id",
        }:
            if isinstance(value, str) and value:
                values.append(value)
        elif key.endswith("_ids") and isinstance(value, list):
            values.extend(str(item) for item in value if item)
    return list(dict.fromkeys(values))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["CodeGraphStore", "SnapshotVerificationError"]
