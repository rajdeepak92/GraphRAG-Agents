"""Reusable MCP prompts for MARAG workflows."""

# mypy: disable-error-code=untyped-decorator

from __future__ import annotations

from typing import Any


def register_prompts(mcp: Any) -> None:
    @mcp.prompt()
    def marag_ingest_brd_srs(project: str, document: str, version: str) -> str:
        return (
            "Ingest this BRD/SRS document into Agentic-GraphRAG.\n\n"
            f"Project: {project}\n"
            f"Document path: {document}\n"
            f"Version: {version}\n\n"
            "Before ingestion:\n"
            "- Check the local Windows MARAG stack.\n"
            "- Start PostgreSQL service if configured and stopped.\n"
            "- Start Neo4j DBMS if configured and stopped.\n"
            "- Use json output.\n"
            "- Return requirement artifact paths.\n"
        )

    @mcp.prompt()
    def marag_generate_user_stories(project: str) -> str:
        return (
            "Generate enterprise-grade user stories from the latest requirement artifact "
            f"for project {project}.\n\n"
            "Use hybrid retrieval and preserve traceability to requirement IDs and evidence "
            "chunks.\n\n"
            "Return artifact path and summary counts.\n"
        )

    @mcp.prompt()
    def marag_generate_test_scenarios(project: str) -> str:
        return (
            f"Generate test scenarios from user stories for project {project}.\n\n"
            "Preserve requirement traceability and evidence links.\n\n"
            "Return artifact path and summary counts.\n"
        )

    @mcp.prompt()
    def marag_full_pipeline(project: str, document: str, version: str) -> str:
        return (
            "Run the full Agentic-GraphRAG workflow.\n\n"
            f"Project: {project}\n"
            f"Document: {document}\n"
            f"Version: {version}\n\n"
            "Steps:\n"
            "1. Check local Windows stack health.\n"
            "2. Start PostgreSQL service if configured.\n"
            "3. Start Neo4j DBMS if configured.\n"
            "4. Ingest document.\n"
            "5. Generate user stories.\n"
            "6. Generate test scenarios.\n"
            "7. Verify generated artifacts.\n"
            "8. Return final artifact paths.\n"
        )
