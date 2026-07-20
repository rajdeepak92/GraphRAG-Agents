# Multi-Agentic QA Knowledge GraphRAG

This repository implements a project-scoped QA workflow through behavioral
test-scenario generation and autonomous Stage 4 test-code generation. Test
execution against real hardware, Jira publishing, and Allure reporting remain
outside the current scope.

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
5. Stage 4 freezes the ordered Stage 3 scenario set, the selected reasoning
   provider, exact test data, and framework checksum. It then generates each
   test case serially, validates its plain-Python lifecycle and Robot caller,
   publishes a verified READY framework-graph snapshot, and only then advances
   to the next scenario. A failed case is restored from its own file journal.

All LangGraph workflows use PostgreSQL checkpointing when PostgreSQL mode is
configured. Item-level state is checkpointed by chunk, requirement, story, or
six-digit TC identity, and external writes are idempotent. `progress_*.json`
files are diagnostic mirrors only; the checkpointer is the sole recovery
authority.

Stage 4 writes directly to the selected local framework. Its code path performs
no Git operation and does not inspect repository state. Existing production
functions are immutable. Generated files are limited to module-specific
`tests/`, `tests_robot/`, and `test_lib/` paths; helpers and wrappers may be
added only under `test_lib/<module>/`. Each case keeps an open recovery journal
until validation, Graphify publication, Neo4j verification, and PostgreSQL
persistence have all succeeded.

## Storage boundaries

- Neo4j stores project-scoped chunks, canonical entities, `MENTIONS` edges, and
  only `USES`, `SUPPORTS`, `CONTROLS`, `COLLECTS_FROM`, `COMMUNICATES_VIA`,
  `CONNECTS_TO`, and `REFERS_TO` semantic relationships.
- ChromaDB stores Stage 1.1 chunk text and embeddings in one normalized
  collection per project. Stage 1.2 never accesses ChromaDB.
- PostgreSQL stores runs, readiness, canonical artifacts, evidence and
  traceability mappings, generation contexts, and LangGraph checkpoints. The
  checkpoint tables are provisioned idempotently by `db-check`/schema setup.
- PostgreSQL is the sole allocator for permanent Stage 4 TC IDs from `100001`
  through `999999`. It also stores frozen run manifests, normalized test-data
  snapshots, journal metadata, KG publication attempts, blockers, and accepted
  test cases.
- Neo4j stores BUILDING and READY filesystem-derived framework snapshots,
  typed test-data bindings, code artifacts, and complete scenario/story/
  requirement traceability for accepted test cases.
- Local JSON artifacts are validated mirrors under
  `generated/<project>/<run-id>/`.

The source file is transient input. There are no document or document-version
identities, fact/assertion layers, revision lifecycles, fallback relationships,
or human-feedback workflow.

## Setup

```powershell
uv sync --dev --extra stage4
Copy-Item .env.example .env
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag doctor
uv run python -m multi_agentic_graph_rag db-check
```

Configure Azure OpenAI or Google Gemini for reasoning and embeddings in `.env`
(`REASONING_MODEL_PROVIDER` / `EMBEDDING_MODEL_PROVIDER` = `azure_openai` or
`gemini`). Reranking is always the private Hugging Face reranker
(`RERANKER_MODEL_PROVIDER=huggingface`). Install the cloud extras you need:
`uv sync --dev --extra azure`, `uv sync --dev --extra gemini`, and
`uv sync --dev --extra local-llm` (reranker). Stage 4 provider selection is
explicit on every command (or uses the configured `azure_openai` production
default). It never discovers a provider from ambient credentials and never
falls back from one provider to the other. A transport failure gets one retry
against the same pinned provider and model.
Set `HUGGINGFACE_DEVICE=auto`, `cpu`, or `cuda` for the reranker; explicit
`cuda` fails fast when the installed PyTorch build cannot access an NVIDIA GPU.
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

### Stage 4 autonomous test-code generation

Index the local framework, then ingest the exact test-data document. XLSX files
are opened with formula text visible; any formula in an ingested data region is
a hard error rather than a cached-value fallback.

```powershell
uv run python -m multi_agentic_graph_rag index-framework `
  --framework-path C:\frameworks\customer-portal

uv run python -m multi_agentic_graph_rag ingest-test-data `
  --project customer-portal `
  --document C:\frameworks\customer-portal\test-data\test-data.xlsx
```

Use dry-run first. It performs deterministic configuration, provider,
framework, test-data, Graphify-capability, and persistence readiness checks. It
makes no model request and writes no framework file.

```powershell
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal `
  --run-id RUN-001 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider azure_openai `
  --dry-run
```

The Azure production run uses the configured Azure deployment name and cannot
initialize Gemini:

```powershell
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal `
  --run-id RUN-001 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider azure_openai
```

A Gemini run is a separate, explicitly pinned execution. It uses the configured
`GEMINI_REASONING_MODEL` and cannot initialize Azure OpenAI.

```powershell
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal-gemini `
  --run-id RUN-GEMINI-001 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider gemini
```

Stage 4 does not execute the generated domain test. Validation is limited to
path and additive-AST policy, Python parse/compile, isolated import, lifecycle
contract checks, Robot Framework dry-run, traceability, exact-data checks, and
file hashes. The generated Robot test calls only `Run Test`; Python owns setup,
execution, and guaranteed teardown behavior.

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

Stage 4 maintenance integrations should call
`reset_stage4_project(project, settings)` instead when only code-generation
orchestration state must be removed. That scoped reset removes replayable
run-coordination rows and only each run's `stage-4/` journals and `test-cases/`
review artifacts. Permanent TC reservations/masters, referenced test-data and
READY graph snapshots, and their traceability projections are retained so a
logical key can never be renumbered. It preserves Stages 1-3, retains the TC
sequence, and never deletes generated framework code.
Framework files remain under the user's control and require a separate explicit
filesystem action.

The Stage 4 verification runbook intentionally contains no source-control
command. Inspect results through framework paths,
`generated/<project>/<run-id>/test-cases/test_cases.json`, PostgreSQL, and Neo4j.

## Verification

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy src\multi_agentic_graph_rag
uv run pytest -q
```

The authoritative implementation contract is in [implementation.md](implementation.md).
