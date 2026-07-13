"""Neo4j document/chunk graph projection with a local JSON trace mode."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from multi_agentic_graph_rag.config.settings import AppSettings
from multi_agentic_graph_rag.domain.schemas import (
    DocumentChunk,
    DocumentManifest,
    EntityRecord,
    KnowledgeGraphArtifact,
    RequirementArtifact,
    TestScenarioArtifact,
    TestScenarioBuildResult,
    TextUnit,
    UserStoryArtifact,
    UserStoryBuildResult,
)
from multi_agentic_graph_rag.services.assertion_lifecycle import (
    LifecycleUpdate,
    PriorAssertion,
)

_LUCENE_TOKEN = re.compile(r"[A-Za-z0-9]+")


class Neo4jStore:
    """Coordinate neo4j store behavior within the db boundary."""

    def __init__(self, settings: AppSettings) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (AppSettings): Validated settings that control this operation.
        """
        self.settings = settings

    def check(self) -> str:
        """Check check.

        Returns:
            str: The typed result produced by the operation.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
        if self.settings.neo4j.mode == "local_json":
            self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
            return f"PASS neo4j local_json path={self.settings.neo4j.local_path}"
        with self._driver() as driver:
            driver.verify_connectivity()
        return "PASS neo4j connectivity"

    def project_manifest(self, manifest: DocumentManifest) -> None:
        """Project manifest through the owning storage boundary.

        Args:
            manifest (DocumentManifest): Manifest required by the operation's typed contract.
        """
        if self.settings.neo4j.mode == "local_json":
            self._upsert_local(
                "manifest_projection",
                manifest.document_version_id,
                {"kind": "manifest_projection", "manifest": manifest.model_dump(mode="json")},
            )
            return

        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(_project_manifest_tx, manifest.model_dump(mode="json"))

    def project_artifact(self, artifact: RequirementArtifact) -> None:
        """Compatibility no-op.

        Generated requirement artifacts and their ledger rows are intentionally
        stored in PostgreSQL, not projected into Neo4j.
        """
        _ = artifact

    def ensure_search_index(self) -> None:
        """Create the chunk full-text index and derivative-node uniqueness constraints."""
        if self.settings.neo4j.mode == "local_json":
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.run(
                "CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS "
                "FOR (c:Chunk) ON EACH [c.text, c.normalized_text]"
            )
            # Chunk identity contract: chunk_id is version-scoped (derived from the
            # document-version identifier), so a uniqueness constraint both backs the
            # MERGE-by-chunk_id projection with an index and enforces immutable
            # one-node-per-chunk identity. A separate range index on
            # document_version_id keeps version-scoped retrieval and neighbour
            # expansion from scanning all chunks.
            session.run(
                "CREATE CONSTRAINT chunk_pk IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            )
            session.run(
                "CREATE INDEX chunk_document_version IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.document_version_id)"
            )
            # PostgreSQL is authoritative for generated artifacts, but downstream
            # coverage projection MERGEs Requirement/UserStory/TestScenario nodes;
            # uniqueness constraints keep those derivative nodes deduplicated.
            for label, key in (
                ("Requirement", "requirement_id"),
                ("UserStory", "story_id"),
                ("TestScenario", "scenario_id"),
            ):
                session.run(
                    f"CREATE CONSTRAINT {label.lower()}_pk IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
                )

    def fulltext_search_chunks(
        self,
        query: str,
        document_version_id: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        """BM25/Lucene keyword search over chunks, scoped to one document version."""
        if limit <= 0:
            return []
        lucene_query = _lucene_keyword_query(query)
        if not lucene_query:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_fulltext_search(query, document_version_id, limit)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
                YIELD node, score
                WHERE node.document_version_id = $document_version_id
                RETURN node.chunk_id AS chunk_id, node.text AS text, score AS score
                LIMIT $limit
                """,
                query=lucene_query,
                document_version_id=document_version_id,
                limit=limit,
            )
            return [
                (str(record["chunk_id"]), str(record["text"] or ""), float(record["score"]))
                for record in records
            ]

    def neighbor_chunks(
        self,
        chunk_ids: list[str],
        document_version_id: str,
        window: int,
    ) -> list[tuple[str, str]]:
        """Multi-hop neighbours of the seed chunks within the same document version."""
        if not chunk_ids:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_neighbor_chunks(chunk_ids, document_version_id, window)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $document_version_id})
                      -[:HAS_CHUNK]->(seed:Chunk)
                WHERE seed.chunk_id IN $chunk_ids
                MATCH (v)-[:HAS_CHUNK]->(c:Chunk)
                WHERE abs(c.ordinal - seed.ordinal) <= $window
                   OR (seed.section IS NOT NULL AND c.section = seed.section)
                RETURN DISTINCT c.chunk_id AS chunk_id, c.text AS text, c.ordinal AS ordinal
                ORDER BY ordinal
                """,
                document_version_id=document_version_id,
                chunk_ids=chunk_ids,
                window=window,
            )
            return [(str(record["chunk_id"]), str(record["text"] or "")) for record in records]

    def fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        """Fetch (chunk_id, text) for the requested chunks, preserving input order."""
        if not chunk_ids:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_fetch_chunks(chunk_ids)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (c:Chunk)
                WHERE c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS chunk_id, c.text AS text
                """,
                chunk_ids=chunk_ids,
            )
            by_id = {str(record["chunk_id"]): str(record["text"] or "") for record in records}
        return [(chunk_id, by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in by_id]

    def project_user_story_coverage(
        self,
        artifact: UserStoryArtifact | UserStoryBuildResult,
        evidence_chunk_ids: Mapping[str, list[str]],
    ) -> None:
        """Project validated user-story coverage claim-nodes back into the graph.

        Writes UserStory -> Requirement -> Chunk traceability so a later stage can
        retrieve graph context, and marks each covered requirement.
        """
        evidence = {key: list(value) for key, value in evidence_chunk_ids.items()}
        records = _user_story_records(artifact)
        if self.settings.neo4j.mode == "local_json":
            for story_id, record in records.items():
                self._upsert_local(
                    "user_story_projection",
                    story_id,
                    {
                        "kind": "user_story_projection",
                        "story_id": story_id,
                        "requirement_id": record.requirement_id,
                        "revision_id": record.requirement_revision_id,
                        "source_req_id": record.source_req_id,
                        "project": record.project,
                        "document_version_id": record.document_version_id,
                        "title": record.title,
                        "covered": True,
                        "evidence_chunk_ids": evidence.get(record.requirement_id, []),
                        "generation_context_run_id": record.generation_context_run_id,
                        "retrieved_assertion_ids": list(record.retrieved_assertion_ids),
                        "context_mode": record.context_mode,
                    },
                )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _project_user_story_coverage_tx,
                {"stories": {key: value.model_dump(mode="json") for key, value in records.items()}},
                evidence,
            )

    def project_test_scenario_coverage(
        self,
        artifact: TestScenarioArtifact | TestScenarioBuildResult,
        evidence_chunk_ids: Mapping[str, list[str]],
    ) -> None:
        """Project validated test-scenario claim-nodes back into the graph."""
        evidence = {key: list(value) for key, value in evidence_chunk_ids.items()}
        records = _test_scenario_records(artifact)
        if self.settings.neo4j.mode == "local_json":
            for scenario_id, record in records.items():
                self._upsert_local(
                    "test_scenario_projection",
                    scenario_id,
                    {
                        "kind": "test_scenario_projection",
                        "scenario_id": scenario_id,
                        "story_id": record.story_id,
                        "requirement_id": record.requirement_id,
                        "revision_id": record.requirement_revision_id,
                        "source_req_id": record.source_req_id,
                        "project": record.project,
                        "document_version_id": record.document_version_id,
                        "title": record.title,
                        "scenario_type": record.scenario_type,
                        "priority": record.priority,
                        "confidence": record.confidence,
                        "evidence_chunk_ids": evidence.get(record.requirement_id, []),
                        "generation_context_run_id": record.generation_context_run_id,
                        "retrieved_assertion_ids": list(record.retrieved_assertion_ids),
                        "context_mode": record.context_mode,
                    },
                )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _project_test_scenario_coverage_tx,
                {
                    "scenarios": {
                        key: value.model_dump(mode="json") for key, value in records.items()
                    }
                },
                evidence,
            )

    def cleanup_identity_projections(self, project: str) -> None:
        """Idempotently remove generated derivative projections before rebuilding."""
        if self.settings.neo4j.mode == "local_json":
            rows = [
                row
                for row in self._read_local_rows()
                if not (
                    row.get("project") == project
                    and row.get("kind") in {"user_story_projection", "test_scenario_projection"}
                )
            ]
            self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings.neo4j.local_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.run(
                """
                MATCH (n)
                WHERE n.project = $project
                  AND (n:Requirement OR n:UserStory OR n:TestScenario)
                DETACH DELETE n
                """,
                project=project,
            ).consume()

    def ensure_knowledge_schema(self) -> None:
        """Create the constraints and search indexes for the source-knowledge graph."""
        if self.settings.neo4j.mode == "local_json":
            return
        statements = (
            "CREATE CONSTRAINT entity_pk IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT entity_mention_pk IF NOT EXISTS "
            "FOR (m:EntityMention) REQUIRE m.mention_id IS UNIQUE",
            "CREATE CONSTRAINT assertion_pk IF NOT EXISTS "
            "FOR (a:Assertion) REQUIRE a.assertion_id IS UNIQUE",
            "CREATE CONSTRAINT assertion_evidence_pk IF NOT EXISTS "
            "FOR (ev:AssertionEvidence) REQUIRE ev.evidence_id IS UNIQUE",
            "CREATE CONSTRAINT text_unit_pk IF NOT EXISTS "
            "FOR (u:TextUnit) REQUIRE u.text_unit_id IS UNIQUE",
            "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS "
            "FOR (e:Entity) ON EACH [e.canonical_name, e.aliases_text]",
            "CREATE FULLTEXT INDEX assertion_text_fulltext IF NOT EXISTS "
            "FOR (a:Assertion) ON EACH [a.display_text, a.condition]",
            "CREATE FULLTEXT INDEX text_unit_fulltext IF NOT EXISTS "
            "FOR (u:TextUnit) ON EACH [u.text]",
        )
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            for statement in statements:
                session.run(statement)

    def project_knowledge_graph(self, artifact: KnowledgeGraphArtifact) -> None:
        """Project validated source-knowledge entities/assertions into the graph.

        The knowledge graph is source semantics with exact evidence, not a copy
        of the generated-artifact ledger: PostgreSQL remains the source of truth
        for requirements, user stories, and test scenarios.
        """
        if self.settings.neo4j.mode == "local_json":
            evidence_by_assertion: dict[str, list[dict[str, Any]]] = {}
            for evidence in artifact.evidence:
                evidence_by_assertion.setdefault(evidence.assertion_id, []).append(
                    evidence.model_dump(mode="json")
                )
            for entity in artifact.entities:
                self._upsert_local(
                    "entity_projection",
                    entity.entity_id,
                    {"kind": "entity_projection", **entity.model_dump(mode="json")},
                )
            for mention in artifact.mentions:
                self._upsert_local(
                    "entity_mention_projection",
                    mention.mention_id,
                    {"kind": "entity_mention_projection", **mention.model_dump(mode="json")},
                )
            for assertion in artifact.assertions:
                self._upsert_local(
                    "assertion_projection",
                    assertion.assertion_id,
                    {
                        "kind": "assertion_projection",
                        **assertion.model_dump(mode="json"),
                        "evidence": evidence_by_assertion.get(assertion.assertion_id, []),
                    },
                )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _project_knowledge_graph_tx,
                artifact.model_dump(mode="json"),
            )

    def project_text_units(self, document_version_id: str, text_units: list[TextUnit]) -> None:
        """Project atomic TextUnits with chunk containment and reading order."""
        if self.settings.neo4j.mode == "local_json":
            for unit in text_units:
                self._upsert_local(
                    "text_unit_projection",
                    unit.text_unit_id,
                    {"kind": "text_unit_projection", **unit.model_dump(mode="json")},
                )
            return
        ordered = sorted(text_units, key=lambda unit: unit.ordinal)
        next_pairs = [
            {
                "previous_id": previous.text_unit_id,
                "next_id": current.text_unit_id,
            }
            for previous, current in itertools.pairwise(ordered)
        ]
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _project_text_units_tx,
                document_version_id,
                [unit.model_dump(mode="json") for unit in text_units],
                next_pairs,
            )

    def knowledge_related_chunks(
        self,
        evidence_chunk_ids: list[str],
        document_version_id: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        """Chunks related to the seeds via assertion/entity hops, version-scoped.

        Expansion path: seed chunks -> their assertion evidence -> assertions ->
        subject/object entities -> other assertions of the same document version
        -> their evidence chunks. The score is the count of distinct related
        assertions evidencing the candidate chunk.
        """
        if not evidence_chunk_ids or limit <= 0:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_knowledge_related_chunks(
                evidence_chunk_ids, document_version_id, limit
            )
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (seed:Chunk)<-[:FROM_CHUNK]-(:AssertionEvidence)
                      <-[:SUPPORTED_BY]-(a1:Assertion {document_version_id: $document_version_id})
                WHERE seed.chunk_id IN $chunk_ids
                  AND (a1.status IS NULL OR a1.status = 'active')
                MATCH (a1)-[:SUBJECT|OBJECT]->(e:Entity)
                      <-[:SUBJECT|OBJECT]-(a2:Assertion {
                          document_version_id: $document_version_id
                      })
                WHERE a2.status IS NULL OR a2.status = 'active'
                MATCH (a2)-[:SUPPORTED_BY]->(:AssertionEvidence)-[:FROM_CHUNK]->(c:Chunk)
                WHERE NOT c.chunk_id IN $chunk_ids
                RETURN c.chunk_id AS chunk_id,
                       c.text AS text,
                       count(DISTINCT a2) AS weight
                ORDER BY weight DESC, chunk_id
                LIMIT $limit
                """,
                chunk_ids=evidence_chunk_ids,
                document_version_id=document_version_id,
                limit=limit,
            )
            return [
                (str(record["chunk_id"]), str(record["text"] or ""), float(record["weight"]))
                for record in records
            ]

    def _local_knowledge_related_chunks(
        self,
        evidence_chunk_ids: list[str],
        document_version_id: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        """Execute the local knowledge related chunks operation within its declared architectural
        boundary.

        Args:
            evidence_chunk_ids (list[str]): Evidence chunk ids required by the operation's typed
                                            contract.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.
            limit (int): Bounded limit used for deterministic processing.

        Returns:
            list[tuple[str, str, float]]: The typed result produced by the operation.
        """
        seeds = set(evidence_chunk_ids)
        assertions: list[dict[str, Any]] = [
            row
            for row in self._read_local_rows()
            if row.get("kind") == "assertion_projection"
            and row.get("document_version_id") == document_version_id
            and str(row.get("status", "active")) == "active"
        ]

        def evidence_chunks(row: dict[str, Any]) -> set[str]:
            """Execute the evidence chunks operation within its declared architectural boundary.

            Args:
                row (dict[str, Any]): Validated structured data for the operation.

            Returns:
                set[str]: The typed result produced by the operation.
            """
            chunks: set[str] = set()
            for evidence in row.get("evidence", []):
                trace = evidence.get("source_trace", {})
                chunk_id = str(trace.get("chunk_id", ""))
                if chunk_id:
                    chunks.add(chunk_id)
            return chunks

        def entity_ids(row: dict[str, Any]) -> set[str]:
            """Execute the entity ids operation within its declared architectural boundary.

            Args:
                row (dict[str, Any]): Validated structured data for the operation.

            Returns:
                set[str]: The typed result produced by the operation.
            """
            ids = {str(row.get("subject_entity_id", ""))}
            object_entity = row.get("object_entity_id")
            if object_entity:
                ids.add(str(object_entity))
            return {value for value in ids if value}

        seed_entities: set[str] = set()
        for row in assertions:
            if evidence_chunks(row) & seeds:
                seed_entities |= entity_ids(row)
        if not seed_entities:
            return []

        weight_by_chunk: dict[str, int] = {}
        for row in assertions:
            if not entity_ids(row) & seed_entities:
                continue
            for chunk_id in evidence_chunks(row) - seeds:
                weight_by_chunk[chunk_id] = weight_by_chunk.get(chunk_id, 0) + 1

        text_by_id = {
            str(chunk.get("chunk_id", "")): str(chunk.get("text", ""))
            for chunk in self._local_manifest_chunks(document_version_id)
        }
        ranked = sorted(weight_by_chunk.items(), key=lambda item: (-item[1], item[0]))
        return [
            (chunk_id, text_by_id.get(chunk_id, ""), float(weight))
            for chunk_id, weight in ranked[:limit]
        ]

    def search_assertions_fulltext(
        self,
        query: str,
        document_version_id: str,
        limit: int,
        allowed_predicates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword search over assertions, version-scoped and predicate-filterable.

        Returns structured assertion rows (subject/predicate/object + modality,
        polarity, condition, confidence) carrying a ``search_score`` — never chunk
        text. The knowledge-graph read side of the semantic retrieval path.
        """
        if limit <= 0:
            return []
        predicates = _normalized_predicate_filter(allowed_predicates)
        if self.settings.neo4j.mode == "local_json":
            return self._local_search_assertions(query, document_version_id, limit, predicates)
        lucene_query = _lucene_keyword_query(query)
        if not lucene_query:
            return []
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                CALL db.index.fulltext.queryNodes('assertion_text_fulltext', $query)
                YIELD node, score
                WHERE node.document_version_id = $document_version_id
                  AND ($predicates IS NULL OR node.predicate IN $predicates)
                  AND (node.status IS NULL OR node.status = 'active')
                MATCH (node)-[:SUBJECT]->(s:Entity)
                OPTIONAL MATCH (node)-[:OBJECT]->(o:Entity)
                RETURN node AS assertion, s AS subject, o AS object, score AS search_score
                ORDER BY search_score DESC, node.assertion_id
                LIMIT $limit
                """,
                query=lucene_query,
                document_version_id=document_version_id,
                predicates=predicates,
                limit=limit,
            )
            return [
                _assertion_row(
                    dict(record["assertion"]),
                    dict(record["subject"]) if record["subject"] is not None else {},
                    dict(record["object"]) if record["object"] is not None else None,
                    {"search_score": float(record["search_score"])},
                )
                for record in records
            ]

    def fetch_anchor_assertions(
        self,
        evidence_chunk_ids: list[str],
        document_version_id: str,
    ) -> list[dict[str, Any]]:
        """Assertions directly evidenced by the seed chunks (mandatory anchors)."""
        if not evidence_chunk_ids:
            return []
        if self.settings.neo4j.mode == "local_json":
            return self._local_anchor_assertions(evidence_chunk_ids, document_version_id)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (seed:Chunk)<-[:FROM_CHUNK]-(:AssertionEvidence)
                      <-[:SUPPORTED_BY]-(a:Assertion {document_version_id: $document_version_id})
                WHERE seed.chunk_id IN $chunk_ids
                  AND (a.status IS NULL OR a.status = 'active')
                MATCH (a)-[:SUBJECT]->(s:Entity)
                OPTIONAL MATCH (a)-[:OBJECT]->(o:Entity)
                RETURN DISTINCT a AS assertion, s AS subject, o AS object
                ORDER BY a.assertion_id
                """,
                chunk_ids=evidence_chunk_ids,
                document_version_id=document_version_id,
            )
            return [
                _assertion_row(
                    dict(record["assertion"]),
                    dict(record["subject"]) if record["subject"] is not None else {},
                    dict(record["object"]) if record["object"] is not None else None,
                    {},
                )
                for record in records
            ]

    def expand_entity_assertions(
        self,
        entity_ids: list[str],
        document_version_id: str,
        per_entity_limit: int,
        allowed_predicates: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Bounded one-hop expansion: assertions sharing a seed entity.

        Each seed entity contributes at most ``per_entity_limit`` assertions,
        ranked by confidence, keeping the traversal degree-bounded. Rows carry
        the ``via_entity_id`` they were reached through.
        """
        if not entity_ids or per_entity_limit <= 0:
            return []
        predicates = _normalized_predicate_filter(allowed_predicates)
        if self.settings.neo4j.mode == "local_json":
            return self._local_expand_entity_assertions(
                entity_ids, document_version_id, per_entity_limit, predicates
            )
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                UNWIND $entity_ids AS eid
                MATCH (e:Entity {entity_id: eid})<-[:SUBJECT|OBJECT]-(
                    a:Assertion {document_version_id: $document_version_id})
                WHERE ($predicates IS NULL OR a.predicate IN $predicates)
                  AND (a.status IS NULL OR a.status = 'active')
                WITH eid, a ORDER BY a.confidence DESC, a.assertion_id
                WITH eid, collect(a)[0..$per_entity_limit] AS capped
                UNWIND capped AS a
                MATCH (a)-[:SUBJECT]->(s:Entity)
                OPTIONAL MATCH (a)-[:OBJECT]->(o:Entity)
                RETURN eid AS via_entity_id, a AS assertion, s AS subject, o AS object
                """,
                entity_ids=entity_ids,
                document_version_id=document_version_id,
                predicates=predicates,
                per_entity_limit=per_entity_limit,
            )
            return [
                _assertion_row(
                    dict(record["assertion"]),
                    dict(record["subject"]) if record["subject"] is not None else {},
                    dict(record["object"]) if record["object"] is not None else None,
                    {"via_entity_id": str(record["via_entity_id"])},
                )
                for record in records
            ]

    def hydrate_assertion_evidence(
        self,
        assertion_ids: list[str],
        document_version_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch exact evidence (quote + TextUnit ids + locators) per assertion."""
        if not assertion_ids:
            return {}
        if self.settings.neo4j.mode == "local_json":
            return self._local_assertion_evidence(assertion_ids, document_version_id)
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (a:Assertion {document_version_id: $document_version_id})
                      -[:SUPPORTED_BY]->(ev:AssertionEvidence)
                WHERE a.assertion_id IN $assertion_ids
                OPTIONAL MATCH (ev)-[:FROM_TEXT_UNIT]->(u:TextUnit)
                RETURN a.assertion_id AS assertion_id,
                       ev.evidence_id AS evidence_id,
                       ev.chunk_id AS chunk_id,
                       ev.quote AS quote,
                       ev.page AS page,
                       ev.section AS section,
                       ev.start_char AS start_char,
                       ev.end_char AS end_char,
                       collect(u.text_unit_id) AS text_unit_ids
                ORDER BY ev.evidence_id
                """,
                assertion_ids=assertion_ids,
                document_version_id=document_version_id,
            )
            evidence: dict[str, list[dict[str, Any]]] = {}
            for record in records:
                text_unit_ids = [str(value) for value in record["text_unit_ids"] if value]
                evidence.setdefault(str(record["assertion_id"]), []).append(
                    {
                        "evidence_id": str(record["evidence_id"]),
                        "chunk_id": str(record["chunk_id"] or ""),
                        "quote": str(record["quote"] or ""),
                        "page": record["page"],
                        "section": record["section"],
                        "start_char": record["start_char"],
                        "end_char": record["end_char"],
                        "text_unit_ids": text_unit_ids,
                    }
                )
            return evidence

    def _local_assertions(self, document_version_id: str) -> list[dict[str, Any]]:
        # Current-knowledge reads: superseded/retired assertions are excluded so a
        # stale version returns nothing (explicit historical reads are elsewhere).
        """Execute the local assertions operation within its declared architectural boundary.

        Args:
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.

        Returns:
            list[dict[str, Any]]: The typed result produced by the operation.
        """
        return [
            row
            for row in self._read_local_rows()
            if row.get("kind") == "assertion_projection"
            and row.get("document_version_id") == document_version_id
            and str(row.get("status", "active")) == "active"
        ]

    def _local_assertions_any(self, document_version_id: str) -> list[dict[str, Any]]:
        """All assertion rows for a version regardless of lifecycle status.

        Historical/audit read (includes superseded/retired); the current-knowledge
        reads use :meth:`_local_assertions`, which filters to active only.
        """
        return [
            row
            for row in self._read_local_rows()
            if row.get("kind") == "assertion_projection"
            and row.get("document_version_id") == document_version_id
        ]

    def _local_entities(self) -> dict[str, dict[str, Any]]:
        """Execute the local entities operation within its declared architectural boundary.

        Returns:
            dict[str, dict[str, Any]]: The typed result produced by the operation.
        """
        entities: dict[str, dict[str, Any]] = {}
        for row in self._read_local_rows():
            if row.get("kind") == "entity_projection":
                entities[str(row.get("entity_id", ""))] = row
        return entities

    def _local_assertion_row(
        self,
        assertion: dict[str, Any],
        entities: dict[str, dict[str, Any]],
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute the local assertion row operation within its declared architectural boundary.

        Args:
            assertion (dict[str, Any]): Assertion required by the operation's typed contract.
            entities (dict[str, dict[str, Any]]): Entities required by the operation's typed
                                                  contract.
            extra (dict[str, Any]): Extra required by the operation's typed contract.

        Returns:
            dict[str, Any]: The typed result produced by the operation.
        """
        subject = entities.get(str(assertion.get("subject_entity_id", "")), {})
        object_entity_id = assertion.get("object_entity_id")
        obj = entities.get(str(object_entity_id), {}) if object_entity_id else None
        return _assertion_row(assertion, subject, obj, extra)

    def _local_search_assertions(
        self,
        query: str,
        document_version_id: str,
        limit: int,
        predicates: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Execute the local search assertions operation within its declared architectural boundary.

        Args:
            query (str): Input text processed in memory and excluded from diagnostic logs.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.
            limit (int): Bounded limit used for deterministic processing.
            predicates (list[str] | None): Predicates required by the operation's typed contract.

        Returns:
            list[dict[str, Any]]: The typed result produced by the operation.
        """
        terms = {token.lower() for token in _LUCENE_TOKEN.findall(query)}
        if not terms:
            return []
        entities = self._local_entities()
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for assertion in self._local_assertions(document_version_id):
            if predicates is not None and str(assertion.get("predicate", "")) not in predicates:
                continue
            haystack = " ".join(
                str(assertion.get(field, "") or "")
                for field in ("display_text", "condition", "predicate")
            ).lower()
            score = float(sum(1 for term in terms if term in haystack))
            if score <= 0:
                continue
            row = self._local_assertion_row(assertion, entities, {"search_score": score})
            scored.append((score, str(assertion.get("assertion_id", "")), row))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [row for _, _, row in scored[:limit]]

    def _local_anchor_assertions(
        self,
        evidence_chunk_ids: list[str],
        document_version_id: str,
    ) -> list[dict[str, Any]]:
        """Execute the local anchor assertions operation within its declared architectural boundary.

        Args:
            evidence_chunk_ids (list[str]): Evidence chunk ids required by the operation's typed
                                            contract.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.

        Returns:
            list[dict[str, Any]]: The typed result produced by the operation.
        """
        seeds = set(evidence_chunk_ids)
        entities = self._local_entities()
        rows: list[dict[str, Any]] = []
        for assertion in self._local_assertions(document_version_id):
            chunks = {
                str(evidence.get("source_trace", {}).get("chunk_id", ""))
                for evidence in assertion.get("evidence", [])
            }
            if chunks & seeds:
                rows.append(self._local_assertion_row(assertion, entities, {}))
        rows.sort(key=lambda row: str(row.get("assertion_id", "")))
        return rows

    def _local_expand_entity_assertions(
        self,
        entity_ids: list[str],
        document_version_id: str,
        per_entity_limit: int,
        predicates: list[str] | None,
    ) -> list[dict[str, Any]]:
        """Execute the local expand entity assertions operation within its declared architectural
        boundary.

        Args:
            entity_ids (list[str]): Entity ids required by the operation's typed contract.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.
            per_entity_limit (int): Per entity limit required by the operation's typed contract.
            predicates (list[str] | None): Predicates required by the operation's typed contract.

        Returns:
            list[dict[str, Any]]: The typed result produced by the operation.
        """
        seed_entities = set(entity_ids)
        entities = self._local_entities()
        by_entity: dict[str, list[dict[str, Any]]] = {eid: [] for eid in seed_entities}
        for assertion in self._local_assertions(document_version_id):
            if predicates is not None and str(assertion.get("predicate", "")) not in predicates:
                continue
            ids = {str(assertion.get("subject_entity_id", ""))}
            object_entity_id = assertion.get("object_entity_id")
            if object_entity_id:
                ids.add(str(object_entity_id))
            for eid in ids & seed_entities:
                by_entity[eid].append(assertion)
        rows: list[dict[str, Any]] = []
        for eid in entity_ids:
            ranked = sorted(
                by_entity.get(eid, []),
                key=lambda a: (-float(a.get("confidence", 0.0)), str(a.get("assertion_id", ""))),
            )
            for assertion in ranked[:per_entity_limit]:
                rows.append(self._local_assertion_row(assertion, entities, {"via_entity_id": eid}))
        return rows

    def _local_assertion_evidence(
        self,
        assertion_ids: list[str],
        document_version_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Execute the local assertion evidence operation within its declared architectural
        boundary.

        Args:
            assertion_ids (list[str]): Assertion ids required by the operation's typed contract.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.

        Returns:
            dict[str, list[dict[str, Any]]]: The typed result produced by the operation.
        """
        wanted = set(assertion_ids)
        evidence: dict[str, list[dict[str, Any]]] = {}
        for assertion in self._local_assertions(document_version_id):
            assertion_id = str(assertion.get("assertion_id", ""))
            if assertion_id not in wanted:
                continue
            rows: list[dict[str, Any]] = []
            for item in assertion.get("evidence", []):
                trace = item.get("source_trace", {})
                rows.append(
                    {
                        "evidence_id": str(item.get("evidence_id", "")),
                        "chunk_id": str(trace.get("chunk_id", "")),
                        "quote": str(trace.get("quote", "")),
                        "page": trace.get("page"),
                        "section": trace.get("section"),
                        "start_char": trace.get("start_char"),
                        "end_char": trace.get("end_char"),
                        "text_unit_ids": [str(tu) for tu in item.get("text_unit_ids", [])],
                    }
                )
            evidence[assertion_id] = sorted(rows, key=lambda row: row["evidence_id"])
        return evidence

    def fetch_version_chunks(self, document_version_id: str) -> list[DocumentChunk]:
        """Fetch all chunks of one document version in ordinal order."""
        if self.settings.neo4j.mode == "local_json":
            chunks = self._local_manifest_chunks(document_version_id)
            ordered = sorted(chunks, key=lambda chunk: int(chunk.get("ordinal", 0)))
            return [DocumentChunk.model_validate(chunk) for chunk in ordered]
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $document_version_id})
                      -[:HAS_CHUNK]->(c:Chunk)
                RETURN c.chunk_id AS chunk_id,
                       c.ordinal AS ordinal,
                       c.text AS text,
                       c.normalized_text AS normalized_text,
                       c.page AS page,
                       c.section AS section,
                       c.start_char AS start_char,
                       c.end_char AS end_char,
                       c.source_block_ids AS source_block_ids
                ORDER BY c.ordinal
                """,
                document_version_id=document_version_id,
            )
            return [
                DocumentChunk(
                    chunk_id=str(record["chunk_id"]),
                    ordinal=int(record["ordinal"]),
                    text=str(record["text"] or ""),
                    normalized_text=str(record["normalized_text"] or ""),
                    page=record["page"],
                    section=record["section"],
                    start_char=int(record["start_char"]),
                    end_char=int(record["end_char"]),
                    source_block_ids=[str(item) for item in record["source_block_ids"] or []],
                )
                for record in records
            ]

    def fetch_version_metadata(self, document_version_id: str) -> dict[str, str] | None:
        """Fetch (project, document_id, version) for one projected document version."""
        if self.settings.neo4j.mode == "local_json":
            for row in reversed(self._read_local_rows()):
                if (
                    row.get("kind") == "manifest_projection"
                    and row.get("_local_key") == document_version_id
                ):
                    manifest = row.get("manifest")
                    if isinstance(manifest, dict):
                        return {
                            "project": str(manifest.get("project", "")),
                            "document_id": str(manifest.get("document_id", "")),
                            "version": str(manifest.get("version", "")),
                        }
            return None
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            record = session.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $document_version_id})
                RETURN v.project AS project,
                       v.document_id AS document_id,
                       v.version AS version
                LIMIT 1
                """,
                document_version_id=document_version_id,
            ).single()
            if record is None:
                return None
            return {
                "project": str(record["project"] or ""),
                "document_id": str(record["document_id"] or ""),
                "version": str(record["version"] or ""),
            }

    def fetch_project_entities(self, project: str) -> list[EntityRecord]:
        """Fetch the project's canonical entities for cross-build resolution."""
        if self.settings.neo4j.mode == "local_json":
            entities: list[EntityRecord] = []
            for row in self._read_local_rows():
                if row.get("kind") != "entity_projection" or row.get("project") != project:
                    continue
                payload = {
                    key: value
                    for key, value in row.items()
                    if key not in {"kind", "_local_key", "written_at"}
                }
                entities.append(EntityRecord.model_validate(payload))
            return entities
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (e:Entity {project: $project})
                RETURN e.entity_id AS entity_id,
                       e.project AS project,
                       e.canonical_name AS canonical_name,
                       e.normalized_name AS normalized_name,
                       e.entity_type AS entity_type,
                       e.aliases AS aliases
                """,
                project=project,
            )
            return [
                EntityRecord(
                    entity_id=str(record["entity_id"]),
                    project=str(record["project"]),
                    canonical_name=str(record["canonical_name"] or ""),
                    normalized_name=str(record["normalized_name"] or ""),
                    entity_type=str(record["entity_type"] or "concept"),
                    aliases=[str(item) for item in record["aliases"] or []],
                )
                for record in records
            ]

    def has_knowledge_assertions(self, document_version_id: str) -> bool:
        """Preflight: does this document version have any projected assertions?

        Used to fail fast when graph-primary generation is requested for a version
        whose semantic knowledge graph was never built.
        """
        if self.settings.neo4j.mode == "local_json":
            return any(
                row.get("kind") == "assertion_projection"
                and row.get("document_version_id") == document_version_id
                for row in self._read_local_rows()
            )
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            record = session.run(
                """
                MATCH (a:Assertion {document_version_id: $document_version_id})
                RETURN a.assertion_id AS assertion_id
                LIMIT 1
                """,
                document_version_id=document_version_id,
            ).single()
            return record is not None

    def fetch_assertion_lineage(self, document_version_id: str) -> list[PriorAssertion]:
        """Active assertions of a version as (id, key, lineage_key) for lifecycle diff."""
        if self.settings.neo4j.mode == "local_json":
            rows = [
                row
                for row in self._read_local_rows()
                if row.get("kind") == "assertion_projection"
                and row.get("document_version_id") == document_version_id
                and str(row.get("status", "active")) == "active"
            ]
            return [
                PriorAssertion(
                    assertion_id=str(row.get("assertion_id", "")),
                    assertion_key=str(row.get("assertion_key", "")),
                    assertion_lineage_key=str(row.get("assertion_lineage_key", "")),
                )
                for row in rows
            ]
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (a:Assertion {document_version_id: $document_version_id})
                WHERE a.status IS NULL OR a.status = 'active'
                RETURN a.assertion_id AS assertion_id,
                       a.assertion_key AS assertion_key,
                       a.assertion_lineage_key AS assertion_lineage_key
                """,
                document_version_id=document_version_id,
            )
            return [
                PriorAssertion(
                    assertion_id=str(record["assertion_id"]),
                    assertion_key=str(record["assertion_key"] or ""),
                    assertion_lineage_key=str(record["assertion_lineage_key"] or ""),
                )
                for record in records
            ]

    def apply_assertion_lifecycle(self, updates: list[LifecycleUpdate]) -> None:
        """Mark prior-version assertions superseded/retired from a lifecycle diff."""
        if not updates:
            return
        payload = [
            {
                "assertion_id": update.assertion_id,
                "status": update.status,
                "superseded_by_assertion_id": update.superseded_by_assertion_id,
            }
            for update in updates
        ]
        if self.settings.neo4j.mode == "local_json":
            by_id = {update["assertion_id"]: update for update in payload}
            rows = self._read_local_rows()
            for row in rows:
                if row.get("kind") != "assertion_projection":
                    continue
                update = by_id.get(str(row.get("assertion_id", "")))
                if update is None:
                    continue
                row["status"] = update["status"]
                row["superseded_by_assertion_id"] = update["superseded_by_assertion_id"]
            self.settings.neo4j.local_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.run(
                """
                UNWIND $updates AS update
                MATCH (a:Assertion {assertion_id: update.assertion_id})
                SET a.status = update.status,
                    a.superseded_by_assertion_id = update.superseded_by_assertion_id
                """,
                updates=payload,
            )

    def prune_superseded_assertions(self, *, document_id: str, dry_run: bool = True) -> list[str]:
        """Opt-in retention policy for historical knowledge (disabled by default).

        Historical (superseded/retired) assertions are **never** hard-deleted
        unless the caller explicitly passes ``dry_run=False``. The default dry run
        returns the ids that *would* be pruned so the policy can be reviewed before
        any destructive action is taken.
        """
        if self.settings.neo4j.mode == "local_json":
            rows = self._read_local_rows()
            prunable = [
                str(row.get("assertion_id", ""))
                for row in rows
                if row.get("kind") == "assertion_projection"
                and row.get("document_id") == document_id
                and str(row.get("status", "active")) in {"superseded", "retired"}
            ]
            if not dry_run and prunable:
                prunable_set = set(prunable)
                kept = [
                    row
                    for row in rows
                    if not (
                        row.get("kind") == "assertion_projection"
                        and str(row.get("assertion_id", "")) in prunable_set
                    )
                ]
                self.settings.neo4j.local_path.write_text(
                    "".join(json.dumps(row) + "\n" for row in kept),
                    encoding="utf-8",
                )
            return prunable
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (a:Assertion {document_id: $document_id})
                WHERE a.status IN ['superseded', 'retired']
                RETURN a.assertion_id AS assertion_id
                """,
                document_id=document_id,
            )
            prunable = [str(record["assertion_id"]) for record in records]
            if not dry_run and prunable:
                session.run(
                    """
                    MATCH (a:Assertion {document_id: $document_id})
                    WHERE a.status IN ['superseded', 'retired']
                    OPTIONAL MATCH (a)-[:SUPPORTED_BY]->(ev:AssertionEvidence)
                    DETACH DELETE a, ev
                    """,
                    document_id=document_id,
                )
            return prunable

    def active_knowledge_version(self, document_id: str) -> str | None:
        """The document version currently pointed at as active knowledge, if any."""
        if self.settings.neo4j.mode == "local_json":
            for row in reversed(self._read_local_rows()):
                if (
                    row.get("kind") == "active_knowledge_version"
                    and row.get("_local_key") == document_id
                ):
                    value = row.get("document_version_id")
                    return str(value) if value else None
            return None
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            record = session.run(
                """
                MATCH (d:Document {document_id: $document_id})
                      -[:ACTIVE_KNOWLEDGE_VERSION]->(v:DocumentVersion)
                RETURN v.document_version_id AS document_version_id
                LIMIT 1
                """,
                document_id=document_id,
            ).single()
            if record is None:
                return None
            value = record["document_version_id"]
            return str(value) if value else None

    def set_active_knowledge_version(self, *, document_id: str, document_version_id: str) -> None:
        """Point ``(:Document)-[:ACTIVE_KNOWLEDGE_VERSION]->(:DocumentVersion)``.

        Per-document (not one global project pointer, since a project owns many
        documents). Called only after a successful knowledge projection so current
        queries never resolve against a half-written version.
        """
        if self.settings.neo4j.mode == "local_json":
            self._upsert_local(
                "active_knowledge_version",
                document_id,
                {
                    "kind": "active_knowledge_version",
                    "document_id": document_id,
                    "document_version_id": document_version_id,
                },
            )
            return
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            session.execute_write(
                _set_active_knowledge_version_tx, document_id, document_version_id
            )

    def fetch_entities_for_resolution(
        self, *, project: str, document_id: str
    ) -> list[EntityRecord]:
        """Entities to resolve a new build against: those participating in the
        document's prior active knowledge version only.

        This bounds resolution to live, relevant entities instead of every stale
        entity ever created for the project. On the first build (no active version
        yet) the set is empty; deterministic entity ids keep re-runs idempotent
        regardless of the pointer's position.
        """
        prior_version = self.active_knowledge_version(document_id)
        if prior_version is None:
            return []
        if self.settings.neo4j.mode == "local_json":
            wanted: set[str] = set()
            for row in self._local_assertions(prior_version):
                subject = str(row.get("subject_entity_id", ""))
                if subject:
                    wanted.add(subject)
                obj = row.get("object_entity_id")
                if obj:
                    wanted.add(str(obj))
            entities: list[EntityRecord] = []
            for row in self._read_local_rows():
                if row.get("kind") != "entity_projection" or row.get("project") != project:
                    continue
                if str(row.get("entity_id", "")) not in wanted:
                    continue
                payload = {
                    key: value
                    for key, value in row.items()
                    if key not in {"kind", "_local_key", "written_at"}
                }
                entities.append(EntityRecord.model_validate(payload))
            return entities
        with (
            self._driver() as driver,
            driver.session(database=self.settings.neo4j.database) as session,
        ):
            records = session.run(
                """
                MATCH (v:DocumentVersion {document_version_id: $prior_version})
                      -[:HAS_ASSERTION]->(a:Assertion)-[:SUBJECT|OBJECT]->(e:Entity)
                WHERE e.project = $project
                RETURN DISTINCT e.entity_id AS entity_id,
                       e.project AS project,
                       e.canonical_name AS canonical_name,
                       e.normalized_name AS normalized_name,
                       e.entity_type AS entity_type,
                       e.aliases AS aliases
                """,
                prior_version=prior_version,
                project=project,
            )
            return [
                EntityRecord(
                    entity_id=str(record["entity_id"]),
                    project=str(record["project"]),
                    canonical_name=str(record["canonical_name"] or ""),
                    normalized_name=str(record["normalized_name"] or ""),
                    entity_type=str(record["entity_type"] or "concept"),
                    aliases=[str(item) for item in record["aliases"] or []],
                )
                for record in records
            ]

    def _driver(self) -> Any:
        """Execute the driver operation within its declared architectural boundary.

        Returns:
            Any: The typed result produced by the operation.
        """
        from neo4j import GraphDatabase, NotificationDisabledClassification

        # PDF chunks have no ``section`` property (only Markdown/DOCX headings set it),
        # so the multi-hop ``neighbor_chunks`` query referencing ``c.section`` emits a
        # spurious UnknownPropertyKey (UNRECOGNIZED) notification. Disable only that
        # classification; genuine schema/performance notifications still surface.
        return GraphDatabase.driver(
            self.settings.neo4j.uri,
            auth=(self.settings.neo4j.username, self.settings.neo4j.password),
            notifications_disabled_classifications=[
                NotificationDisabledClassification.UNRECOGNIZED,
            ],
        )

    def _append_local(self, payload: dict[str, Any]) -> None:
        """Append local.

        Args:
            payload (dict[str, Any]): Validated structured data for the operation.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
        self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
        payload["written_at"] = datetime.now(UTC).isoformat()
        with self.settings.neo4j.local_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _upsert_local(self, kind: str, key: str, payload: dict[str, Any]) -> None:
        """Execute the upsert local operation within its declared architectural boundary.

        Args:
            kind (str): Kind required by the operation's typed contract.
            key (str): Key required by the operation's typed contract.
            payload (dict[str, Any]): Validated structured data for the operation.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
        self.settings.neo4j.local_path.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        if self.settings.neo4j.local_path.exists():
            rows = [
                json.loads(line)
                for line in self.settings.neo4j.local_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        payload["written_at"] = datetime.now(UTC).isoformat()
        payload["_local_key"] = key
        replaced = False
        for index, row in enumerate(rows):
            if row.get("kind") == kind and row.get("_local_key") == key:
                rows[index] = payload
                replaced = True
                break
        if not replaced:
            rows.append(payload)
        self.settings.neo4j.local_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_local_rows(self) -> list[dict[str, Any]]:
        """Read local rows within the authorized project and version scope.

        Returns:
            list[dict[str, Any]]: The typed result produced by the operation.
        """
        if not self.settings.neo4j.local_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.settings.neo4j.local_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _local_manifest_chunks(self, document_version_id: str) -> list[dict[str, Any]]:
        """Execute the local manifest chunks operation within its declared architectural boundary.

        Args:
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.

        Returns:
            list[dict[str, Any]]: The typed result produced by the operation.
        """
        for row in reversed(self._read_local_rows()):
            if (
                row.get("kind") == "manifest_projection"
                and row.get("_local_key") == document_version_id
            ):
                manifest = row.get("manifest")
                if isinstance(manifest, dict):
                    chunks = manifest.get("chunks")
                    if isinstance(chunks, list):
                        return [chunk for chunk in chunks if isinstance(chunk, dict)]
        return []

    def _local_fulltext_search(
        self,
        query: str,
        document_version_id: str,
        limit: int,
    ) -> list[tuple[str, str, float]]:
        """Execute the local fulltext search operation within its declared architectural boundary.

        Args:
            query (str): Input text processed in memory and excluded from diagnostic logs.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.
            limit (int): Bounded limit used for deterministic processing.

        Returns:
            list[tuple[str, str, float]]: The typed result produced by the operation.
        """
        terms = {token.lower() for token in _LUCENE_TOKEN.findall(query)}
        if not terms:
            return []
        scored: list[tuple[str, str, float]] = []
        for chunk in self._local_manifest_chunks(document_version_id):
            text = str(chunk.get("text", ""))
            haystack = text.lower()
            score = float(sum(1 for term in terms if term in haystack))
            if score > 0:
                scored.append((str(chunk.get("chunk_id", "")), text, score))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:limit]

    def _local_neighbor_chunks(
        self,
        chunk_ids: list[str],
        document_version_id: str,
        window: int,
    ) -> list[tuple[str, str]]:
        """Execute the local neighbor chunks operation within its declared architectural boundary.

        Args:
            chunk_ids (list[str]): Chunk ids required by the operation's typed contract.
            document_version_id (str): Canonical document version id used as a safe operational
                                       anchor.
            window (int): Window required by the operation's typed contract.

        Returns:
            list[tuple[str, str]]: The typed result produced by the operation.
        """
        chunks = self._local_manifest_chunks(document_version_id)
        if not chunks:
            return []
        by_id = {str(chunk.get("chunk_id", "")): chunk for chunk in chunks}
        seeds = [by_id[cid] for cid in chunk_ids if cid in by_id]
        seed_ordinals = {int(seed.get("ordinal", 0)) for seed in seeds}
        seed_sections = {
            str(seed.get("section")) for seed in seeds if seed.get("section") is not None
        }
        selected: list[tuple[str, str]] = []
        for chunk in chunks:
            ordinal = int(chunk.get("ordinal", 0))
            section = chunk.get("section")
            near = any(abs(ordinal - seed) <= window for seed in seed_ordinals)
            same_section = section is not None and str(section) in seed_sections
            if near or same_section:
                selected.append((str(chunk.get("chunk_id", "")), str(chunk.get("text", ""))))
        return selected

    def _local_fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]:
        """Execute the local fetch chunks operation within its declared architectural boundary.

        Args:
            chunk_ids (list[str]): Chunk ids required by the operation's typed contract.

        Returns:
            list[tuple[str, str]]: The typed result produced by the operation.
        """
        by_id: dict[str, str] = {}
        for row in self._read_local_rows():
            if row.get("kind") != "manifest_projection":
                continue
            manifest = row.get("manifest")
            if not isinstance(manifest, dict):
                continue
            for chunk in manifest.get("chunks", []):
                if isinstance(chunk, dict):
                    by_id[str(chunk.get("chunk_id", ""))] = str(chunk.get("text", ""))
        return [(chunk_id, by_id[chunk_id]) for chunk_id in chunk_ids if chunk_id in by_id]


def _lucene_keyword_query(query: str) -> str:
    """Sanitize free text into a Lucene OR keyword query, dropping special chars."""
    tokens = _LUCENE_TOKEN.findall(query)
    return " OR ".join(tokens)


def _normalized_predicate_filter(allowed_predicates: list[str] | None) -> list[str] | None:
    """Normalize an allowed-predicate list; ``None``/empty means no filter."""
    if not allowed_predicates:
        return None
    seen: dict[str, None] = {}
    for predicate in allowed_predicates:
        token = str(predicate).strip().upper()
        if token:
            seen.setdefault(token, None)
    return list(seen) or None


def _assertion_row(
    assertion: Mapping[str, Any],
    subject: Mapping[str, Any],
    obj: Mapping[str, Any] | None,
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    """Flatten an assertion node + subject/object entities into a plain row.

    Shared by the real-driver and local-JSON structured assertion queries so both
    paths return the identical shape consumed by the context assembler.
    """
    object_entity_id = assertion.get("object_entity_id")
    row: dict[str, Any] = {
        "assertion_id": str(assertion.get("assertion_id", "")),
        "assertion_key": str(assertion.get("assertion_key", "")),
        "project": str(assertion.get("project", "")),
        "document_id": str(assertion.get("document_id", "")),
        "document_version_id": str(assertion.get("document_version_id", "")),
        "subject_entity_id": str(assertion.get("subject_entity_id", "")),
        "subject_name": str(subject.get("canonical_name", "")),
        "subject_type": str(subject.get("entity_type", "")),
        "predicate": str(assertion.get("predicate", "")),
        "object_entity_id": str(object_entity_id) if object_entity_id else None,
        "object_name": str(obj.get("canonical_name", "")) if obj else None,
        "object_literal": (
            str(assertion["object_literal"])
            if assertion.get("object_literal") is not None
            else None
        ),
        "modality": str(assertion.get("modality", "")),
        "polarity": str(assertion.get("polarity", "")),
        "explicitness": str(assertion.get("explicitness", "")),
        "condition": (
            str(assertion["condition"]) if assertion.get("condition") is not None else None
        ),
        "confidence": float(assertion.get("confidence", 0.0) or 0.0),
        "display_text": str(assertion.get("display_text", "")),
    }
    row.update(extra)
    return row


def _project_user_story_coverage_tx(
    tx: Any,
    artifact: dict[str, Any],
    evidence_chunk_ids: Mapping[str, list[str]],
) -> None:
    """Project user story coverage tx through the owning storage boundary.

    Args:
        tx (Any): Tx required by the operation's typed contract.
        artifact (dict[str, Any]): Artifact required by the operation's typed contract.
        evidence_chunk_ids (Mapping[str, list[str]]): Evidence chunk ids required by the operation's
                                                      typed contract.
    """
    for story_id, record in artifact["stories"].items():
        requirement_id = record["requirement_id"]
        tx.run(
            """
            MERGE (r:Requirement {requirement_id: $requirement_id})
            SET r.covered = true,
                r.project = $project,
                r.document_id = $document_id,
                r.document_version_id = $document_version_id
            MERGE (s:UserStory {story_id: $story_id})
            SET s.title = $title,
                s.requirement_id = $requirement_id,
                s.revision_id = $revision_id,
                s.source_req_id = $source_req_id,
                s.project = $project,
                s.persona = $persona,
                s.priority = $priority,
                s.document_version_id = $document_version_id
            MERGE (s)-[:COVERS_REQUIREMENT]->(r)
            """,
            story_id=story_id,
            requirement_id=requirement_id,
            revision_id=record.get("requirement_revision_id", ""),
            source_req_id=record.get("source_req_id"),
            project=record["project"],
            document_id=record["document_id"],
            document_version_id=record["document_version_id"],
            title=record["title"],
            persona=record["persona"],
            priority=record["priority"],
        )
        tx.run(
            """
            MATCH (s:UserStory {story_id: $story_id})
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (v)-[:HAS_USER_STORY]->(s)
            """,
            story_id=story_id,
            document_version_id=record["document_version_id"],
        )
        for assertion_id in record.get("retrieved_assertion_ids", []):
            tx.run(
                """
                MATCH (s:UserStory {story_id: $story_id})
                MATCH (a:Assertion {assertion_id: $assertion_id})
                MERGE (s)-[:GROUNDED_BY_ASSERTION]->(a)
                """,
                story_id=story_id,
                assertion_id=assertion_id,
            )
        for chunk_id in evidence_chunk_ids.get(requirement_id, []):
            tx.run(
                """
                MATCH (r:Requirement {requirement_id: $requirement_id})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (r)-[:EVIDENCED_BY_CHUNK]->(c)
                """,
                requirement_id=requirement_id,
                chunk_id=chunk_id,
            )


def _project_test_scenario_coverage_tx(
    tx: Any,
    artifact: dict[str, Any],
    evidence_chunk_ids: Mapping[str, list[str]],
) -> None:
    """Project test scenario coverage tx through the owning storage boundary.

    Args:
        tx (Any): Tx required by the operation's typed contract.
        artifact (dict[str, Any]): Artifact required by the operation's typed contract.
        evidence_chunk_ids (Mapping[str, list[str]]): Evidence chunk ids required by the operation's
                                                      typed contract.
    """
    for scenario_id, record in artifact["scenarios"].items():
        requirement_id = record["requirement_id"]
        tx.run(
            """
            MERGE (t:TestScenario {scenario_id: $scenario_id})
            SET t.title = $title,
                t.scenario_type = $scenario_type,
                t.priority = $priority,
                t.confidence = $confidence,
                t.story_id = $story_id,
                t.requirement_id = $requirement_id,
                t.revision_id = $revision_id,
                t.source_req_id = $source_req_id,
                t.project = $project,
                t.document_version_id = $document_version_id
            MERGE (s:UserStory {story_id: $story_id})
            MERGE (t)-[:VALIDATES_STORY]->(s)
            MERGE (r:Requirement {requirement_id: $requirement_id})
            SET r.project = $project,
                r.document_version_id = $document_version_id
            MERGE (t)-[:COVERS_REQUIREMENT]->(r)
            """,
            scenario_id=scenario_id,
            story_id=record["story_id"],
            requirement_id=requirement_id,
            revision_id=record.get("requirement_revision_id", ""),
            source_req_id=record.get("source_req_id"),
            project=record["project"],
            document_version_id=record["document_version_id"],
            title=record["title"],
            scenario_type=record["scenario_type"],
            priority=record["priority"],
            confidence=record["confidence"],
        )
        tx.run(
            """
            MATCH (t:TestScenario {scenario_id: $scenario_id})
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (v)-[:HAS_TEST_SCENARIO]->(t)
            """,
            scenario_id=scenario_id,
            document_version_id=record["document_version_id"],
        )
        for assertion_id in record.get("retrieved_assertion_ids", []):
            tx.run(
                """
                MATCH (t:TestScenario {scenario_id: $scenario_id})
                MATCH (a:Assertion {assertion_id: $assertion_id})
                MERGE (t)-[:GROUNDED_BY_ASSERTION]->(a)
                """,
                scenario_id=scenario_id,
                assertion_id=assertion_id,
            )
        for chunk_id in evidence_chunk_ids.get(requirement_id, []):
            tx.run(
                """
                MATCH (r:Requirement {requirement_id: $requirement_id})
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (r)-[:EVIDENCED_BY_CHUNK]->(c)
                """,
                requirement_id=requirement_id,
                chunk_id=chunk_id,
            )


def _user_story_records(
    artifact: UserStoryArtifact | UserStoryBuildResult,
) -> dict[str, Any]:
    """Execute the user story records operation within its declared architectural boundary.

    Args:
        artifact (UserStoryArtifact | UserStoryBuildResult): Artifact required by the operation's
                                                             typed contract.

    Returns:
        dict[str, Any]: The typed result produced by the operation.

    Raises:
        ValueError: If validated inputs or required dependencies cannot satisfy the contract.
    """
    if isinstance(artifact, UserStoryBuildResult):
        return dict(artifact.records)
    raise ValueError("projecting user-story coverage requires internal build records")


def _test_scenario_records(
    artifact: TestScenarioArtifact | TestScenarioBuildResult,
) -> dict[str, Any]:
    """Execute the test scenario records operation within its declared architectural boundary.

    Args:
        artifact (TestScenarioArtifact | TestScenarioBuildResult): Artifact required by the
                                                                   operation's typed contract.

    Returns:
        dict[str, Any]: The typed result produced by the operation.

    Raises:
        ValueError: If validated inputs or required dependencies cannot satisfy the contract.
    """
    if isinstance(artifact, TestScenarioBuildResult):
        return dict(artifact.records)
    raise ValueError("projecting test-scenario coverage requires internal build records")


def _project_text_units_tx(
    tx: Any,
    document_version_id: str,
    text_units: list[dict[str, Any]],
    next_pairs: list[dict[str, str]],
) -> None:
    """Project text units tx through the owning storage boundary.

    Args:
        tx (Any): Tx required by the operation's typed contract.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
        text_units (list[dict[str, Any]]): Text units required by the operation's typed contract.
        next_pairs (list[dict[str, str]]): Next pairs required by the operation's typed contract.
    """
    if text_units:
        tx.run(
            """
            UNWIND $text_units AS unit
            MERGE (u:TextUnit {text_unit_id: unit.text_unit_id})
            SET u.document_version_id = unit.document_version_id,
                u.ordinal = unit.ordinal,
                u.unit_type = unit.unit_type,
                u.text = unit.text,
                u.start_char = unit.start_char,
                u.end_char = unit.end_char,
                u.page = unit.page,
                u.section = unit.section
            WITH u, unit
            UNWIND unit.chunk_ids AS chunk_id
            MATCH (c:Chunk {chunk_id: chunk_id})
            MERGE (c)-[:CONTAINS_TEXT_UNIT]->(u)
            """,
            text_units=text_units,
            document_version_id=document_version_id,
        )
    if next_pairs:
        tx.run(
            """
            UNWIND $pairs AS pair
            MATCH (previous:TextUnit {text_unit_id: pair.previous_id})
            MATCH (next:TextUnit {text_unit_id: pair.next_id})
            MERGE (previous)-[:NEXT_TEXT_UNIT]->(next)
            """,
            pairs=next_pairs,
        )


def _project_knowledge_graph_tx(tx: Any, artifact: dict[str, Any]) -> None:
    """Idempotent MERGE writes for the source-knowledge graph.

    Structural relationships only (SUBJECT/OBJECT/SUPPORTED_BY/FROM_CHUNK/
    HAS_MENTION/REFERS_TO/HAS_ASSERTION); the predicate is data on the
    Assertion node, so new predicates never require a schema change.
    """
    entities = [
        {**entity, "aliases_text": " ".join(entity.get("aliases", []))}
        for entity in artifact["entities"]
    ]
    if entities:
        tx.run(
            """
            UNWIND $entities AS entity
            MERGE (e:Entity {entity_id: entity.entity_id})
            SET e.project = entity.project,
                e.canonical_name = entity.canonical_name,
                e.normalized_name = entity.normalized_name,
                e.entity_type = entity.entity_type,
                e.aliases = entity.aliases,
                e.aliases_text = entity.aliases_text
            """,
            entities=entities,
        )
    if artifact["mentions"]:
        tx.run(
            """
            UNWIND $mentions AS mention
            MERGE (m:EntityMention {mention_id: mention.mention_id})
            SET m.chunk_id = mention.chunk_id,
                m.surface_text = mention.surface_text,
                m.start_char = mention.start_char,
                m.end_char = mention.end_char
            WITH m, mention
            MATCH (e:Entity {entity_id: mention.entity_id})
            MERGE (m)-[:REFERS_TO]->(e)
            WITH m, mention
            MATCH (c:Chunk {chunk_id: mention.chunk_id})
            MERGE (c)-[:HAS_MENTION]->(m)
            """,
            mentions=artifact["mentions"],
        )
    if artifact["assertions"]:
        tx.run(
            """
            UNWIND $assertions AS assertion
            MERGE (a:Assertion {assertion_id: assertion.assertion_id})
            SET a.assertion_key = assertion.assertion_key,
                a.assertion_lineage_key = assertion.assertion_lineage_key,
                a.project = assertion.project,
                a.document_id = assertion.document_id,
                a.document_version_id = assertion.document_version_id,
                a.predicate = assertion.predicate,
                a.object_literal = assertion.object_literal,
                a.modality = assertion.modality,
                a.polarity = assertion.polarity,
                a.explicitness = assertion.explicitness,
                a.condition = assertion.condition,
                a.confidence = assertion.confidence,
                a.display_text = assertion.display_text,
                a.status = assertion.status,
                a.previous_assertion_id = assertion.previous_assertion_id,
                a.superseded_by_assertion_id = assertion.superseded_by_assertion_id,
                a.revision_type = assertion.revision_type
            WITH a, assertion
            MATCH (s:Entity {entity_id: assertion.subject_entity_id})
            MERGE (a)-[:SUBJECT]->(s)
            WITH a, assertion
            MATCH (v:DocumentVersion {document_version_id: assertion.document_version_id})
            MERGE (v)-[:HAS_ASSERTION]->(a)
            """,
            assertions=artifact["assertions"],
        )
        tx.run(
            """
            UNWIND $assertions AS assertion
            WITH assertion WHERE assertion.previous_assertion_id IS NOT NULL
            MATCH (a:Assertion {assertion_id: assertion.assertion_id})
            MATCH (prev:Assertion {assertion_id: assertion.previous_assertion_id})
            MERGE (a)-[:PREVIOUS_ASSERTION]->(prev)
            """,
            assertions=artifact["assertions"],
        )
        tx.run(
            """
            UNWIND $assertions AS assertion
            MATCH (a:Assertion {assertion_id: assertion.assertion_id})
            MATCH (o:Entity {entity_id: assertion.object_entity_id})
            MERGE (a)-[:OBJECT]->(o)
            """,
            assertions=artifact["assertions"],
        )
    if artifact["evidence"]:
        evidence_rows = [
            {
                "evidence_id": evidence["evidence_id"],
                "assertion_id": evidence["assertion_id"],
                "text_unit_ids": evidence.get("text_unit_ids", []),
                **evidence["source_trace"],
            }
            for evidence in artifact["evidence"]
        ]
        tx.run(
            """
            UNWIND $evidence AS evidence
            MERGE (ev:AssertionEvidence {evidence_id: evidence.evidence_id})
            SET ev.chunk_id = evidence.chunk_id,
                ev.quote = evidence.quote,
                ev.start_char = evidence.start_char,
                ev.end_char = evidence.end_char,
                ev.page = evidence.page,
                ev.section = evidence.section
            WITH ev, evidence
            MATCH (a:Assertion {assertion_id: evidence.assertion_id})
            MERGE (a)-[:SUPPORTED_BY]->(ev)
            WITH ev, evidence
            MATCH (c:Chunk {chunk_id: evidence.chunk_id})
            MERGE (ev)-[:FROM_CHUNK]->(c)
            """,
            evidence=evidence_rows,
        )
        tx.run(
            """
            UNWIND $evidence AS evidence
            UNWIND evidence.text_unit_ids AS tu_id
            MATCH (ev:AssertionEvidence {evidence_id: evidence.evidence_id})
            MATCH (u:TextUnit {text_unit_id: tu_id})
            MERGE (ev)-[:FROM_TEXT_UNIT]->(u)
            """,
            evidence=evidence_rows,
        )


def _set_active_knowledge_version_tx(tx: Any, document_id: str, document_version_id: str) -> None:
    """Execute the set active knowledge version tx operation within its declared architectural
    boundary.

    Args:
        tx (Any): Tx required by the operation's typed contract.
        document_id (str): Canonical document id used as a safe operational anchor.
        document_version_id (str): Canonical document version id used as a safe operational anchor.
    """
    tx.run(
        """
        MATCH (d:Document {document_id: $document_id})
        OPTIONAL MATCH (d)-[r:ACTIVE_KNOWLEDGE_VERSION]->(:DocumentVersion)
        DELETE r
        WITH d
        MATCH (v:DocumentVersion {document_version_id: $document_version_id})
        MERGE (d)-[:ACTIVE_KNOWLEDGE_VERSION]->(v)
        """,
        document_id=document_id,
        document_version_id=document_version_id,
    )


def _project_manifest_tx(tx: Any, manifest: dict[str, Any]) -> None:
    """Project manifest tx through the owning storage boundary.

    Args:
        tx (Any): Tx required by the operation's typed contract.
        manifest (dict[str, Any]): Manifest required by the operation's typed contract.
    """
    tx.run(
        """
        MERGE (p:Project {project: $project})
        MERGE (d:Document {document_id: $document_id})
        SET d.project = $project,
            d.logical_name = $logical_name,
            d.source_path = $source_path
        MERGE (p)-[:OWNS_DOCUMENT]->(d)
        MERGE (v:DocumentVersion {document_version_id: $document_version_id})
        SET v.project = $project,
            v.document_id = $document_id,
            v.version = $version,
            v.checksum = $checksum,
            v.source_checksum = $checksum,
            v.source_path = $source_path,
            v.parser_fingerprint = $parser_fingerprint,
            v.chunker_fingerprint = $chunker_fingerprint,
            v.created_at = $created_at
        MERGE (d)-[:HAS_VERSION]->(v)
        """,
        project=manifest["project"],
        document_id=manifest["document_id"],
        document_version_id=manifest["document_version_id"],
        logical_name=manifest["logical_name"],
        version=manifest["version"],
        checksum=manifest["source_checksum"],
        source_path=manifest["source_path"],
        parser_fingerprint=manifest["parser_fingerprint"],
        chunker_fingerprint=manifest["chunker_fingerprint"],
        created_at=manifest["created_at"],
    )
    for chunk in manifest["chunks"]:
        tx.run(
            """
            MATCH (v:DocumentVersion {document_version_id: $document_version_id})
            MERGE (c:Chunk {chunk_id: $chunk_id})
            SET c.project = $project,
                c.document_id = $document_id,
                c.document_version_id = $document_version_id,
                c.version = $version,
                c.source_checksum = $source_checksum,
                c.source_path = $source_path,
                c.chunk_id = $chunk_id,
                c.ordinal = $ordinal,
                c.text = $text,
                c.normalized_text = $normalized_text,
                c.page = $page,
                c.section = $section,
                c.start_char = $start_char,
                c.end_char = $end_char,
                c.source_block_ids = $source_block_ids
            MERGE (v)-[:HAS_CHUNK]->(c)
            """,
            project=manifest["project"],
            document_id=manifest["document_id"],
            document_version_id=manifest["document_version_id"],
            version=manifest["version"],
            source_checksum=manifest["source_checksum"],
            source_path=manifest["source_path"],
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            normalized_text=chunk["normalized_text"],
            ordinal=chunk["ordinal"],
            page=chunk.get("page"),
            section=chunk.get("section"),
            start_char=chunk["start_char"],
            end_char=chunk["end_char"],
            source_block_ids=chunk["source_block_ids"],
        )
