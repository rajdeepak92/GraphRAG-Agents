# Multi-Agentic QA Knowledge GraphRAG

This repository implements a project-scoped QA workflow through behavioral
test-scenario generation. Test-case generation, execution, Jira publishing, and
Allure reporting are outside the current scope.

## Workflow

`ingest` is additive and never deletes existing project state. Use the explicit
`project-reset --project <name> --yes` maintenance command when a clean project
is intentionally required. Then:

1. Stage 1.1 parses one source file, creates stable chunks, persists only
   project-scoped `Chunk` nodes in Neo4j, writes chunk embeddings to one Chroma
   collection per project, validates both stores, and publishes
   `requirements/chunk_manifest.json`.
2. Stage 1.2 processes every manifest chunk with exactly one combined
   reasoning-model response containing `chunk_id`, `requirements`, `entities`,
   and `relationships`. Python rejects extra or missing fields, validates
   temporary references, exact source-ID coverage, evidence, and semantic
   grounding, checkpoints the validated response, then projects only allowlisted
   relationship types. Schema or semantic invalidity gets no repair call. The
   first terminal invalid chunk stops Stage 1.2, blocks publication, and marks
   readiness `failed`.
3. Stage 2 retrieves current-run graph and vector evidence for every requirement
   and publishes `user-stories/user-stories.json` plus the diagnostic
   `user-stories/progress_story.json`.
4. Stage 3 retrieves current-run evidence for every story and publishes
   `test-scenario/test-scenarios.json` plus the diagnostic
   `test-scenario/progress_scenario.json`.

All four LangGraph workflows use PostgreSQL checkpointing when PostgreSQL mode
is configured. Item-level state is checkpointed by chunk, requirement, or story,
and external writes are idempotent. `progress_*.json` files are diagnostic
mirrors only; the checkpointer is the sole recovery authority.

## Storage boundaries

- Neo4j stores project-scoped chunks, canonical entities, `MENTIONS` edges, and
  only `USES`, `SUPPORTS`, `CONTROLS`, `COLLECTS_FROM`, `COMMUNICATES_VIA`,
  `CONNECTS_TO`, and `REFERS_TO` semantic relationships.
- ChromaDB stores Stage 1.1 chunk text and embeddings in one normalized
  collection per project. Stage 1.2 never accesses ChromaDB.
- PostgreSQL stores runs, readiness, canonical artifacts, evidence and
  traceability mappings, generation contexts, and LangGraph checkpoints. The
  checkpoint tables are provisioned idempotently by `db-check`/schema setup.
- Local JSON artifacts are validated mirrors under
  `generated/<project>/<run-id>/`.

The source file is transient input. There are no document or document-version
identities, fact/assertion layers, revision lifecycles, fallback relationships,
or human-feedback workflow.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag doctor
uv run python -m multi_agentic_graph_rag db-check
```

Configure either Azure OpenAI or private Hugging Face models in `.env`.
Set `HUGGINGFACE_DEVICE=auto`, `cpu`, or `cuda`; explicit `cuda` fails fast when
the installed PyTorch build cannot access an NVIDIA GPU.
PostgreSQL, Neo4j, and ChromaDB must be reachable for the real workflow.
`local_json` PostgreSQL and Neo4j modes are available for isolated development
tests.

## Commands

Run joined Stage 1.1 and Stage 1.2 additively:

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project customer-portal `
  --document .\documents\requirements.docx
```

The command prints the generated `run_id`. Use it for downstream stages:

```powershell
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --project customer-portal `
  --run-id <RUN-ID>

uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --project customer-portal `
  --run-id <RUN-ID>

uv run python -m multi_agentic_graph_rag coverage `
  --project customer-portal `
  --run-id <RUN-ID>
```

Diagnostics:

```powershell
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag hf-check
uv run python -m multi_agentic_graph_rag hf-check --load-model
uv run python -m multi_agentic_graph_rag doctor
uv run python -m multi_agentic_graph_rag db-check
```

`postgres-reset --yes` is destructive and is only for an explicitly disposable
development PostgreSQL database:

```powershell
uv run python -m multi_agentic_graph_rag postgres-reset --yes
```

Reset one project across Neo4j, ChromaDB, PostgreSQL checkpoints/artifacts, and
generated files only when explicitly intended:

```powershell
uv run python -m multi_agentic_graph_rag project-reset `
  --project customer-portal `
  --yes
```

## Verification

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src\multi_agentic_graph_rag
uv run pytest -q
```

The authoritative implementation contract is in [implementation.md](implementation.md).
