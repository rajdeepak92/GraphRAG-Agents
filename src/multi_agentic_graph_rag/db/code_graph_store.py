"""Snapshot-scoped Neo4j code-property graph store (plan §6, §7.2).

Deliberately separate from the document ``Neo4jStore`` (Chunk/Entity/MENTIONS):
the code graph uses its own labels — ``FrameworkSnapshot``, ``CodeFile``,
``CodeSymbol``, ``CodeEdge`` — and its own uniqueness constraints so a short
name like ``start`` is never merged globally (plan §6.6). A ``local_json`` mode
mirrors the document store so the code graph is testable without a live server.
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


class CodeGraphStore:
    """Persist and query the canonical code graph for a framework snapshot."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    @property
    def _local(self) -> bool:
        return self.settings.neo4j.mode == "local_json"

    @property
    def _local_path(self) -> Path:
        return self.settings.stage4.code_graph_local_path

    def check(self) -> str:
        """Validate connectivity or the local path."""
        if self._local:
            self._local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS code-graph local_json path={self._local_path}"
        with self._session() as session:
            session.run("RETURN 1").consume()
        return "PASS code-graph connectivity"

    def ensure_schema(self) -> None:
        """Create snapshot-scoped uniqueness and search indexes (plan §6.6)."""
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
        )
        with self._session() as session:
            for statement in statements:
                session.run(statement).consume()

    def publish_snapshot(
        self, snapshot: FrameworkSnapshot, result: CodeExtractionResult
    ) -> FrameworkSnapshot:
        """Persist files/symbols/edges then mark the snapshot READY (plan §7.2 s.17)."""
        if snapshot.snapshot_id != result.snapshot_id:
            raise ValueError("snapshot/result snapshot_id mismatch")
        ready = snapshot.model_copy(update={"status": "ready"})
        if self._local:
            self._upsert_local("code_snapshot", ready.snapshot_id, ready.model_dump(mode="json"))
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
            return ready
        self._publish_neo4j(ready, result)
        return ready

    def get_symbol(self, snapshot_id: str, symbol_id: str) -> CodeSymbol | None:
        """Return one symbol by its snapshot-scoped identity."""
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
        """Exact/substring symbol search within a snapshot (plan §10.5 s.5)."""
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
        """Traverse outbound edges up to ``depth`` and return reachable symbols."""
        depth = max(1, min(depth, 4))
        edges = self._iter_edges(snapshot_id)
        adjacency: dict[str, list[str]] = {}
        for edge in edges:
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
        """Remove every node/edge belonging to a snapshot. Returns rows removed."""
        if self._local:
            rows = self._local_rows()
            kept = [row for row in rows if row.get("snapshot_id") != snapshot_id]
            removed = len(rows) - len(kept)
            _write_rows(self._local_path, kept)
            return removed
        with self._session() as session:
            summary = session.run(
                "MATCH (n) WHERE n.snapshot_id = $s DETACH DELETE n", s=snapshot_id
            ).consume()
            return int(summary.counters.nodes_deleted)

    # --- Neo4j write ---------------------------------------------------------

    def _publish_neo4j(self, snapshot: FrameworkSnapshot, result: CodeExtractionResult) -> None:
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

    # --- connection + local helpers -----------------------------------------

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
        path = self._local_path
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        # Nest the payload so its own fields (e.g. CodeSymbol.kind) never collide
        # with the row markers used for filtering.
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


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


__all__ = ["CodeGraphStore"]
