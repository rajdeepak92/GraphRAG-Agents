"""Phase A: chunk / TextUnit retention contract.

These tests lock the lexical-source contract the downstream KG-readiness and
cumulative-master work depends on:

* full chunk text is retained once on the Neo4j ``Chunk`` node, and never copied
  onto Assertion / Entity / EntityMention / evidence nodes;
* ``chunk_id`` is the canonical, version-scoped evidence anchor shared across
  Neo4j, Chroma-facing manifests, and evidence;
* chunk identity is protected by a uniqueness constraint plus a
  ``document_version_id`` index;
* assertions resolve through evidence to a chunk of the same document version,
  with exact quote/offset grounding;
* retrieval (fetch, neighbour expansion, full-text) stays version-isolated;
* evidence text is recoverable from Neo4j alone (no Chroma dependency).
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agentic_graph_rag.config.settings import (
    AppSettings,
    Neo4jSettings,
    PathsSettings,
    PostgresSettings,
)
from multi_agentic_graph_rag.db.neo4j_store import (
    Neo4jStore,
    _project_knowledge_graph_tx,
    _project_manifest_tx,
)
from multi_agentic_graph_rag.domain.identifiers import chunk_id as make_chunk_id
from multi_agentic_graph_rag.domain.schemas import (
    AssertionEvidenceRecord,
    AssertionRecord,
    DocumentChunk,
    DocumentManifest,
    EntityMentionRecord,
    EntityRecord,
    KnowledgeGraphArtifact,
    SourceTrace,
)

_CHUNK_TEXT = "The gateway shall collect operating data every 5 seconds."


class ChunkRetentionContractTests(unittest.TestCase):
    # 1. Full chunk text lives once on the Chunk node; not duplicated elsewhere.
    def test_chunk_node_retains_full_text_and_metadata(self) -> None:
        tx = _FakeTx()
        _project_manifest_tx(tx, _manifest("DOC-v1", "V1").model_dump(mode="json"))

        chunk_run = next(run for run in tx.runs if "MERGE (c:Chunk" in run["query"])
        params = chunk_run["params"]
        self.assertEqual(params["text"], _CHUNK_TEXT)
        for key in (
            "chunk_id",
            "project",
            "document_id",
            "document_version_id",
            "ordinal",
            "normalized_text",
            "page",
            "section",
            "start_char",
            "end_char",
            "source_checksum",
            "source_block_ids",
        ):
            with self.subTest(key=key):
                self.assertIn(key, params)

    # 2. No redundant chunk text is copied onto KG nodes; they keep their own
    #    span text (display_text / surface_text / quote), anchored by chunk_id.
    def test_knowledge_nodes_do_not_duplicate_chunk_text(self) -> None:
        tx = _FakeTx()
        _project_knowledge_graph_tx(tx, _kg_artifact("DOC-v1").model_dump(mode="json"))

        for run in tx.runs:
            query = run["query"]
            # Only the AssertionEvidence quote may equal source text, and it is a
            # bounded quotation, not the whole chunk text property assignment.
            self.assertNotIn("a.text =", query)
            self.assertNotIn("e.text =", query)
            self.assertNotIn("m.text =", query)
        assertion_run = next(run for run in tx.runs if "MERGE (a:Assertion" in run["query"])
        self.assertIn("a.display_text = assertion.display_text", assertion_run["query"])
        mention_run = next(run for run in tx.runs if "MERGE (m:EntityMention" in run["query"])
        self.assertIn("m.surface_text = mention.surface_text", mention_run["query"])

    # 3. ensure_search_index protects chunk identity: uniqueness constraint on
    #    chunk_id plus a document_version_id index.
    def test_ensure_search_index_creates_chunk_constraint_and_index(self) -> None:
        store = Neo4jStore(_live_settings())
        sink: list[str] = []
        store._driver = lambda: _FakeDriver(sink)  # type: ignore[method-assign]

        store.ensure_search_index()

        joined = "\n".join(sink)
        self.assertIn("CONSTRAINT chunk_pk", joined)
        self.assertIn("FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE", joined)
        self.assertIn("INDEX chunk_document_version", joined)
        self.assertIn("FOR (c:Chunk) ON (c.document_version_id)", joined)

    # 4. chunk_id is version-scoped, so the uniqueness constraint never collapses
    #    a V1 and V2 chunk that share ordinal + text.
    def test_chunk_id_is_version_scoped(self) -> None:
        v1 = make_chunk_id("DOC-v1", 1, _CHUNK_TEXT)
        v2 = make_chunk_id("DOC-v2", 1, _CHUNK_TEXT)
        self.assertNotEqual(v1, v2)

    # 5. Assertion -> evidence -> chunk integrity: every evidence chunk_id is a
    #    real chunk of the same document version.
    def test_assertion_evidence_resolves_to_version_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Neo4jStore(_settings(Path(temp_dir)))
            store.project_manifest(_manifest("DOC-v1", "V1"))
            store.project_knowledge_graph(_kg_artifact("DOC-v1"))

            chunk_ids = {chunk.chunk_id for chunk in store.fetch_version_chunks("DOC-v1")}
            evidence = store.hydrate_assertion_evidence(["AST-1"], "DOC-v1")

        rows = evidence["AST-1"]
        self.assertTrue(rows)
        for row in rows:
            self.assertIn(row["chunk_id"], chunk_ids)

    # 6. Exact quote + offsets validate against retained chunk text.
    def test_evidence_quote_matches_chunk_offsets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Neo4jStore(_settings(Path(temp_dir)))
            store.project_manifest(_manifest("DOC-v1", "V1"))
            store.project_knowledge_graph(_kg_artifact("DOC-v1"))

            chunks = {c.chunk_id: c for c in store.fetch_version_chunks("DOC-v1")}
            evidence = store.hydrate_assertion_evidence(["AST-1"], "DOC-v1")

        row = evidence["AST-1"][0]
        chunk = chunks[row["chunk_id"]]
        self.assertEqual(chunk.text[row["start_char"] : row["end_char"]], row["quote"])

    # 7. V1 / V2 isolation on chunk fetch and neighbour expansion.
    def test_retrieval_is_version_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Neo4jStore(_settings(Path(temp_dir)))
            store.project_manifest(_manifest("DOC-v1", "V1"))
            store.project_manifest(_manifest("DOC-v2", "V2"))

            v1_chunks = store.fetch_version_chunks("DOC-v1")
            v2_chunks = store.fetch_version_chunks("DOC-v2")
            v1_ids = {c.chunk_id for c in v1_chunks}
            v2_ids = {c.chunk_id for c in v2_chunks}

            neighbours = store.neighbor_chunks(list(v1_ids), "DOC-v1", window=2)

        self.assertTrue(v1_ids)
        self.assertTrue(v2_ids)
        self.assertEqual(v1_ids & v2_ids, set())
        neighbour_ids = {cid for cid, _ in neighbours}
        self.assertTrue(neighbour_ids <= v1_ids)
        self.assertEqual(neighbour_ids & v2_ids, set())

    # 8. Evidence text is recoverable from Neo4j alone (no Chroma lookup needed).
    def test_evidence_text_recoverable_from_neo4j(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Neo4jStore(_settings(Path(temp_dir)))
            store.project_manifest(_manifest("DOC-v1", "V1"))
            evidence_chunk_id = make_chunk_id("DOC-v1", 1, _CHUNK_TEXT)

            recovered = dict(store.fetch_chunks([evidence_chunk_id]))

        self.assertEqual(recovered.get(evidence_chunk_id), _CHUNK_TEXT)


class _FakeTx:
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def run(self, query: str, **params: Any) -> None:
        self.runs.append({"query": query, "params": params})


class _FakeSession:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def run(self, query: str, **params: Any) -> None:
        self._sink.append(query)


class _FakeDriver:
    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def __enter__(self) -> _FakeDriver:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def session(self, database: str | None = None) -> _FakeSession:
        return _FakeSession(self._sink)


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
        postgres=PostgresSettings(
            mode="local_json",
            local_path=root / "runtime" / "postgres.jsonl",
        ),
        neo4j=Neo4jSettings(
            mode="local_json",
            local_path=root / "runtime" / "neo4j.jsonl",
        ),
    )


def _live_settings() -> AppSettings:
    root = Path(tempfile.gettempdir()) / "phase_a_live"
    settings = _settings(root)
    settings.neo4j.mode = "neo4j"
    return settings


def _manifest(document_version_id: str, doc_version: str) -> DocumentManifest:
    return DocumentManifest(
        project="PROJECT",
        document_id="DOC",
        document_version_id=document_version_id,
        logical_name="source",
        version=doc_version,
        source_path="source.txt",
        source_checksum=f"checksum-{doc_version}",
        parser_fingerprint="parser",
        chunker_fingerprint="chunker",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        chunks=[
            DocumentChunk(
                chunk_id=make_chunk_id(document_version_id, 1, _CHUNK_TEXT),
                ordinal=1,
                text=_CHUNK_TEXT,
                normalized_text=_CHUNK_TEXT.lower(),
                page=3,
                section="Monitoring",
                start_char=0,
                end_char=len(_CHUNK_TEXT),
                source_block_ids=["BLOCK-1"],
            )
        ],
    )


def _kg_artifact(document_version_id: str) -> KnowledgeGraphArtifact:
    quote = "The gateway shall collect operating data"
    return KnowledgeGraphArtifact(
        project="PROJECT",
        document_id="DOC",
        document_version_id=document_version_id,
        doc_version="V1",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        entities=[
            EntityRecord(
                entity_id="ENT-GATEWAY",
                project="PROJECT",
                canonical_name="Gateway",
                normalized_name="gateway",
                entity_type="system",
                aliases=["gw"],
            ),
            EntityRecord(
                entity_id="ENT-DATA",
                project="PROJECT",
                canonical_name="Operating Data",
                normalized_name="operating data",
                entity_type="data_object",
            ),
        ],
        mentions=[
            EntityMentionRecord(
                mention_id="MENTION-1",
                entity_id="ENT-GATEWAY",
                chunk_id=make_chunk_id(document_version_id, 1, _CHUNK_TEXT),
                surface_text="gateway",
                start_char=4,
                end_char=11,
            )
        ],
        assertions=[
            AssertionRecord(
                assertion_id="AST-1",
                assertion_key="KEY-1",
                project="PROJECT",
                document_id="DOC",
                document_version_id=document_version_id,
                subject_entity_id="ENT-GATEWAY",
                predicate="COLLECTS",
                object_entity_id="ENT-DATA",
                modality="shall",
                polarity="positive",
                explicitness="explicit",
                confidence=0.95,
                display_text="Gateway COLLECTS Operating Data",
            )
        ],
        evidence=[
            AssertionEvidenceRecord(
                evidence_id="ASTEVID-1",
                assertion_id="AST-1",
                source_trace=SourceTrace(
                    chunk_id=make_chunk_id(document_version_id, 1, _CHUNK_TEXT),
                    quote=quote,
                    start_char=0,
                    end_char=len(quote),
                ),
            )
        ],
    )


if __name__ == "__main__":
    unittest.main()
