"""Simplified PostgreSQL and Neo4j schema regression tests."""

from __future__ import annotations

from pathlib import Path

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.db.postgres import PostgresStore


def test_postgres_schema_is_active_only_and_has_no_legacy_layers() -> None:
    source = Path("src/multi_agentic_graph_rag/db/postgres.py").read_text(encoding="utf-8")
    assert "CHECK (status = 'active')" in source
    assert "requirements_project_source_id" in source
    assert "source_req_id_type IN ('source','generated')" in source
    assert "DROP CONSTRAINT" in source
    assert "CASCADE" not in source
    for removed in (
        "canonical_facts",
        "fact_occurrences",
        "document_versions",
        "requirement_revisions",
        "superseded_by",
    ):
        assert removed not in source


def test_neo4j_schema_contains_only_direct_chunk_semantic_contract() -> None:
    source = Path("src/multi_agentic_graph_rag/db/neo4j_store.py").read_text(encoding="utf-8")
    assert "(c:Chunk)" in source
    assert "(e:Entity)" in source
    assert "MENTIONS" in source
    for removed in ("DocumentVersion", "TextUnit", "Assertion", "Requirement"):
        assert removed not in source


def test_explicit_local_development_reset_removes_the_store(tmp_path: Path) -> None:
    settings = load_config()
    settings.postgres.mode = "local_json"
    settings.postgres.local_path = tmp_path / "postgres.jsonl"
    settings.postgres.local_path.write_text('{"kind":"test"}\n', encoding="utf-8")
    result = PostgresStore(settings).reset_schema()
    assert result == "PASS postgres local_json reset"
    assert not settings.postgres.local_path.exists()
