# Agentic-GraphRAG Working Rules

Read [CLAUDE.md](./CLAUDE.md) for the long-form project memory. Keep this file short and authoritative.

- Treat Agentic-GraphRAG as a production GraphRAG platform with a stable ingestion-first contract.
- Preserve the existing architecture, CLI behavior, strict schemas, auditable IDs, and Neo4j/PostgreSQL source-of-truth design.
- Keep requirement discovery, user-story generation, and test-scenario generation behavior stable unless the user explicitly asks for a change.
- Ingestion performs a best-effort source-knowledge build (entities/assertions/text units) when enabled; `build-knowledge-graph` remains an explicit rebuild command. Neo4j must not become a second generated requirement/story/scenario ledger, which stays in PostgreSQL.
- Knowledge-graph retrieval is flag-gated (`KnowledgeGraphSettings`: `KNOWLEDGE_GRAPH_ENABLED`, `KNOWLEDGE_GRAPH_SHADOW_MODE`, `GRAPH_PRIMARY_STORY`, `GRAPH_PRIMARY_SCENARIO`). The checked-in profile enables graph-primary generation; shadow mode records comparison snapshots.
- Make narrow changes with minimal blast radius. Inspect the relevant files before editing and update tests and docs with code changes.
- HFIL is re-approved only for optional test-scenario review before final persistence.
  Broader HITL and MCP remain retired unless explicitly re-approved.
- The canonical shared literal registries are `multi_agentic_graph_rag/common_defs.py` (runtime literals) and `multi_agentic_graph_rag/common_prompt_defs.py` (prompts); do not re-inline those literals elsewhere.
- Keep provenance and traceability intact. Do not add hidden behavior or speculative replacements.
- Canonical requirement, revision, and evidence IDs are UUIDv7-backed (`REQ-`, `REQREV-`,
  `REQEVID-`). Public requirement artifacts use `5.0-requirements`, one canonical revision row
  with nested evidence. Story/scenario contracts use canonical parent IDs; presentation aliases
  are retired and may appear only in the explicit legacy database contraction migration.
- `requirements.json` is the only production requirement artifact. The internal requirement
  model is persisted relationally, while `identity_resolution.json` carries resolution audit data.
  Project identity repair is dry-run by default, fail-closed on ambiguity, transactional, and
  rebuilds derivative Neo4j projections after apply.
- Prefer repository-local conventions over generic abstractions. Avoid broad refactors unless they are required to remove a bug or satisfy the request.
- On Windows, use `uv run` for checks and project commands.
- Standard validation after code changes: `uv run ruff check .`, `uv run ruff format --check ...`, `uv run mypy src/multi_agentic_graph_rag`, `uv run python -m unittest discover -s tests`, and `uv run python -m compileall -q src`.
- If a request is underspecified, risky, or would widen the blast radius, explain the delta and confirm before making edits.
- When the durable project guidance changes, update this file and the long-form memory doc together.
