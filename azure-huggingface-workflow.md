# Azure OpenAI + Hugging Face Workflow

This is the current Windows PowerShell workflow for Azure OpenAI reasoning and
embeddings, a Hugging Face reranker, PostgreSQL canonical generated-content
storage, Neo4j source-knowledge/projection storage, and local Chroma retrieval.

## 1. Install and configure

Requirements: Python 3.12, `uv`, PostgreSQL, Neo4j, an Azure OpenAI reasoning
deployment, an Azure embeddings deployment, and access to the configured Hugging
Face reranker.

```powershell
Copy-Item .env.example .env
uv sync --dev --extra azure --extra local-llm
```

Set these values in `.env`:

```dotenv
REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface

AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=<reasoning-deployment>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment>

HF_TOKEN=<token-if-required>
HUGGINGFACE_RERANKER_MODEL=BAAI/bge-reranker-base
HUGGINGFACE_OFFLINE=false

POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:pwd@127.0.0.1:5432/marag

NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j
```

Requirement identity recall is configured separately from scenario deduplication:

```dotenv
REQUIREMENT_CANDIDATE_TOP_K=8
REQUIREMENT_RECALL_COSINE_THRESHOLD=0.62
REQUIREMENT_USE_RERANKER=true
```

These values retrieve and reorder candidates only. They never prove identity.
Permanent merging requires compatible structured semantics and, for semantic
duplicates, strict bidirectional entailment. Ambiguity creates a distinct lineage
and an audit warning.

The checked-in example profile enables the source knowledge graph and graph-primary
story/scenario retrieval:

```dotenv
KNOWLEDGE_GRAPH_ENABLED=true
KNOWLEDGE_GRAPH_SHADOW_MODE=true
GRAPH_PRIMARY_STORY=true
GRAPH_PRIMARY_SCENARIO=true
KNOWLEDGE_GRAPH_EXPANSION_K=6
KNOWLEDGE_GRAPH_MIN_ASSERTIONS=3
```

Set `KNOWLEDGE_GRAPH_ENABLED=false` for the explicit chunk-only path. Neo4j holds
source entities/assertions/text units and derivative coverage projections;
PostgreSQL remains authoritative for requirements, stories, and scenarios.

## 2. Validate providers and stores

```powershell
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag hf-check
uv run python -m multi_agentic_graph_rag db-check
uv run python -m multi_agentic_graph_rag doctor
```

`config-check` must report Azure for reasoning/embeddings and Hugging Face for
reranking. `db-check` must report PostgreSQL, Neo4j, and Chroma as available.
Schema creation/migration is idempotent; do not run `postgres-reset` for an
upgrade or identity repair. Reset commands are only for disposable development
stores.

## 3. Ingest a document

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project PROJECT_1 `
  --document documents\inbox\PROJECT_1\source-document.pdf `
  --version 1.0 `
  --logical-name source-document `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --json-output
```

Ingestion parses and chunks the document, indexes chunks in Chroma, acquires a
per-document identity lock, loads prior active revisions before discovery, and
supplies bounded prior requirement memory to the extraction prompt. The LLM emits
atomic semantic candidates; Python validates traces and semantic slots, resolves
identity, allocates UUIDv7 IDs, persists PostgreSQL first, then writes local JSON.

When enabled, ingestion also attempts the evidence-grounded source knowledge graph
build. A graph-build warning does not discard a successfully committed requirement
artifact.

The run directory contains:

```text
generated/PROJECT_1/req/<RUN_ID>/
  requirements.json
  identity_resolution.json
  chunk_manifest.json
  run.log
  run.jsonl
```

`requirements.json` is the sole public requirement contract:
`5.0-requirements`, one object per `(requirement_id, revision_id)`, with all source
occurrences nested as evidence. Newly allocated identifiers use
`REQ-<uuid7>`, `REQREV-<uuid7>`, and `REQEVID-<uuid7>`. There is no
`requirements_full.json` or display-ID namespace.

```powershell
$RUN_ID = (Get-ChildItem generated\PROJECT_1\req |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1).Name

uv run python -m multi_agentic_graph_rag artifact verify `
  generated\PROJECT_1\req\$RUN_ID\requirements.json
```

`identity_resolution.json` records the incoming fingerprint, recalled candidates
and scores, reranker order, deterministic outcome, entailment result when used,
reason, and assigned canonical IDs.

## 4. Generate user stories

```powershell
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --reranker-provider huggingface `
  --json-output

uv run python -m multi_agentic_graph_rag artifact verify-user-stories `
  generated\PROJECT_1\req\$RUN_ID\user_stories.json
```

The loader accepts canonical schema 5 only, groups by canonical identity, requires
exactly one active revision, and never selects the first array entry implicitly.
Story records retain both `requirement_id` and `revision_id`.

## 5. Generate test scenarios

```powershell
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --user-stories generated\PROJECT_1\req\$RUN_ID\user_stories.json `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --reranker-provider huggingface `
  --no-hfil `
  --json-output

uv run python -m multi_agentic_graph_rag artifact verify-test-scenarios `
  generated\PROJECT_1\req\$RUN_ID\test_scenarios.json
```

Use `--hfil --thread-id PROJECT_1-$RUN_ID-hfil` for the optional scenario-review
loop. HFIL is limited to test-scenario review before final persistence.

## 6. Ingest a later document version

Reuse the same project and `--logical-name`:

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project PROJECT_1 `
  --document documents\inbox\PROJECT_1\source-document-v2.pdf `
  --version 2.0 `
  --logical-name source-document `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --json-output
```

Unchanged meaning reuses the requirement and revision IDs. A mutable threshold,
timeout, or limit change normally reuses the requirement ID and creates a new
revision ID while superseding the old active revision. Entity discriminators such
as `Sensor-1` and `Sensor-2`, and incompatible requirement families, remain
separate lineages. Use `--replace-version` only when intentionally replacing an
already stored version label.

## 7. Repair legacy identities

Always inspect the machine-readable dry run first:

```powershell
uv run python -m multi_agentic_graph_rag requirements repair-identities `
  --project PROJECT_1 `
  --dry-run `
  --json-output
```

Apply only after resolving every reported ambiguity and taking a database backup:

```powershell
uv run python -m multi_agentic_graph_rag requirements repair-identities `
  --project PROJECT_1 `
  --apply `
  --json-output
```

Project apply runs under one PostgreSQL advisory transaction (or the equivalent
local file lock), upgrades legacy IDs to UUIDv7, splits incompatible polluted
lineages, merges exact canonical duplicates, remaps evidence/story/scenario
references, rewrites canonical artifact payloads, and cleans/rebuilds Neo4j
derivative projections. Apply aborts on unresolved ambiguity and repeated execution
is idempotent. Do not execute `--apply` against shared/production infrastructure
without an approved change window.

For a standalone legacy `4.0-catalog` file, use the explicit adapter:

```powershell
uv run python -m multi_agentic_graph_rag requirements repair-identities `
  --artifact .\legacy\requirements.json `
  --dry-run `
  --json-output
```

The adapter is the only runtime path that reads legacy catalog/display fields.

## 8. Coverage, mirror repair, and troubleshooting

```powershell
uv run python -m multi_agentic_graph_rag coverage --project PROJECT_1
uv run python -m multi_agentic_graph_rag reconcile --project PROJECT_1
```

`reconcile` rematerializes local canonical JSON from PostgreSQL; it does not make
semantic identity decisions.

- Azure failures: verify endpoint, key, deployment names, API version, and saved
  `llm_response_*.txt` files when response logging is enabled.
- Reranker failures: verify `HF_TOKEN`, the exact Hub model ID, license access,
  and local RAM/GPU capacity.
- PostgreSQL failures: verify `POSTGRES_DSN` and run `db-check`; never fix a
  migration problem with a destructive reset on retained data.
- Neo4j failures: verify Bolt credentials and database name. Requirement/story/
  scenario ledgers remain safe in PostgreSQL if a best-effort graph build fails.
- Chroma failures: rebuild only the local Chroma directory, then re-ingest.

## 9. Repository verification

```powershell
uv lock
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy
uv run pytest -q
```
