# Commands

Use this after cloning the repo on Windows PowerShell.

## 1. Setup Environment

```powershell
git clone <repo-url>
cd Agentic-GraphRAG
Copy-Item .env.example .env
```

Recommended baseline:

```powershell
uv --version
uv sync
uv run marag version
uv run marag config-check
uv run marag doctor
```

If you want optional provider support:

```powershell
uv sync --extra azure
uv sync --extra local-llm
```

Notes:

- `uv sync` installs the core runtime dependencies from `pyproject.toml`.
- `--extra azure` adds `openai` and `azure-identity`.
- `--extra local-llm` adds the Hugging Face / local model stack.

## 2. Sync Dependencies As Needed

Use the smallest dependency set that matches the run mode:

```powershell
# Core runtime only
uv sync

# Azure OpenAI provider support
uv sync --extra azure

# Hugging Face / local model support
uv sync --extra local-llm

# Full local setup
uv sync --extra azure --extra local-llm
```

## 3. Setup Databases And Check

The repo supports two database modes:

- Local trace mode: no external PostgreSQL or Neo4j server is required.
- Real service mode: set `POSTGRES_MODE=postgres` and `NEO4J_MODE=neo4j`.

### Local trace mode

Leave the defaults in place:

```powershell
POSTGRES_MODE=local_json
NEO4J_MODE=local_json
```

Then verify:

```powershell
uv run marag db-check
```

Expected local paths:

- PostgreSQL trace file: `runtime/staging/postgres_records.jsonl`
- Neo4j trace file: `runtime/staging/neo4j_projection.jsonl`
- Chroma persistence: `runtime/databases/chroma/`

### Real PostgreSQL / Neo4j / Chroma

Start the services with your preferred local install, Docker setup, or cloud
endpoint, then point `.env` at them.

Required `.env` values:

```powershell
POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:<password>@127.0.0.1:5432/marag

NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j
```

Then verify:

```powershell
uv run marag db-check
```

## 4. Clean Databases If Previous Data Exists

Use the cleanup that matches your mode.

### Local trace cleanup

This removes the local trace files and local Chroma persistence:

```powershell
Remove-Item -Force runtime\staging\postgres_records.jsonl -ErrorAction SilentlyContinue
Remove-Item -Force runtime\staging\neo4j_projection.jsonl -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force runtime\databases\chroma -ErrorAction SilentlyContinue
```

If you also want to clear prior run artifacts:

```powershell
Remove-Item -Recurse -Force .generated\<PROJECT>\run -ErrorAction SilentlyContinue
```

### Real PostgreSQL cleanup

Run this against the app database:

```powershell
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" `
  -c "TRUNCATE TABLE test_scenarios, test_scenario_artifacts, user_stories, user_story_artifacts, requirement_fact_links, requirement_evidence, requirement_delta_events, requirement_revisions, requirements, fact_occurrences, canonical_facts, requirement_artifacts, document_versions, ingestion_runs RESTART IDENTITY CASCADE;"
```

### Real Neo4j cleanup

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <password> "MATCH (n) DETACH DELETE n;"
```

### Real Chroma cleanup

If you are using the local embedded Chroma store, deleting the persistence
directory is usually the fastest reset:

```powershell
Remove-Item -Recurse -Force runtime\databases\chroma -ErrorAction SilentlyContinue
```

## 5. Configure Hugging Face For Local Dry Runs

For offline local dry-runs, use the local provider defaults and keep Hugging Face
dependencies installed:

```powershell
REASONING_MODEL_PROVIDER=local_heuristic
EMBEDDING_MODEL_PROVIDER=local_hash
RERANKER_MODEL_PROVIDER=none
HUGGINGFACE_OFFLINE=true
HUGGINGFACE_TOKEN=
HUGGINGFACE_REASONING_MODEL=
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

If you want the Hugging Face provider path instead of the local heuristic path,
set the provider names to `huggingface` and point `HUGGINGFACE_REASONING_MODEL`
and `HUGGINGFACE_EMBEDDING_MODEL` at the exact model ids you want to run.

Suggested `.env` block for local dry-runs:

```powershell
APP_ENV=development
LOG_LEVEL=INFO
REASONING_MODEL_PROVIDER=local_heuristic
EMBEDDING_MODEL_PROVIDER=local_hash
RERANKER_MODEL_PROVIDER=none
POSTGRES_MODE=local_json
NEO4J_MODE=local_json
HUGGINGFACE_OFFLINE=true
HUGGINGFACE_TOKEN=
HUGGINGFACE_REASONING_MODEL=
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

## 6. Configure Azure OpenAI

Use Azure when you want hosted reasoning, embeddings, or reranking inputs.
Keep the provider split explicit.

Suggested `.env` block:

```powershell
APP_ENV=development
LOG_LEVEL=INFO

REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=none

AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<azure-openai-key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=<reasoning-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>

POSTGRES_MODE=local_json
NEO4J_MODE=local_json
```

If you also want real databases, change the database settings at the same time:

```powershell
POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:<password>@127.0.0.1:5432/marag

NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j
```

## 7. Run Ingest

```powershell
uv run marag ingest `
  --project PROJECT_1 `
  --document .\documents\inbox\PROJECT_1\sample.txt `
  --version 1.0
```

Useful variations:

```powershell
uv run marag ingest --project PROJECT_1 --document .\documents\inbox\PROJECT_1\sample.txt --version 1.0 --json-output
uv run marag ingest --project PROJECT_1 --document .\documents\inbox\PROJECT_1\sample.txt --version 1.0 --replace-version
```

## 8. Generate User Stories

Stage 3 consumes the requirement artifact and generates user stories per
requirement with hybrid retrieval (Chroma dense + Neo4j BM25 + Neo4j multi-hop)
fused by the reranker. It requires the real stack
(`POSTGRES_MODE=postgres`, `NEO4J_MODE=neo4j`, real reasoning/embedding/reranker
providers) and adds managed tables, so run `postgres-reset --yes` once before
the first run.

Public JSON artifacts use per-project display aliases: `requirements.json` is
`4.0-catalog` with `REQ-###`, `user_stories.json` is `2.0-user-stories` with
`US-###`, and `test_scenarios.json` is `2.0-test-scenarios` with `TS-###`.
Internal workflow state, PostgreSQL rows, Neo4j merges, resume, HFIL, and dedup
continue to use internal hash IDs.

```powershell
uv run marag postgres-reset --yes

# From the local requirement artifact written by ingest:
uv run marag generate-user-stories `
  --requirements generated\PROJECT_1\req\<RUN_ID>\requirements.json `
  --project PROJECT_1 `
  --json-output

# From PostgreSQL (local artifact absent), selected by document version:
uv run marag generate-user-stories --document-version-id <DV-ID> --project PROJECT_1 --json-output
```

Useful variations:

```powershell
uv run marag generate-user-stories --requirements .\path\to\requirements.json --project PROJECT_1 --top-k 4
uv run marag generate-user-stories --requirements .\path\to\requirements.json --project PROJECT_1 --reasoning-provider azure_openai --embedding-provider azure_openai --reranker-provider huggingface
uv run marag artifact verify generated\PROJECT_1\req\<RUN_ID>\requirements.json
uv run marag artifact verify-user-stories generated\PROJECT_1\req\<RUN_ID>\user_stories.json
```

`user_stories.json` is written beside the input requirements file, or under
`generated/<PROJECT>/user_stories/<RUN_ID>/` when loaded from PostgreSQL.

## 9. Generate Test Scenarios

Stage 4 consumes `user_stories.json` and generates implementation-ready test
scenarios per user story. It uses the same hybrid retrieval stack as stage 3:
Chroma dense search, Neo4j full-text search, Neo4j neighbor expansion, and the
reranker. It writes `test_scenarios.json`, persists rows to PostgreSQL, and
projects `TestScenario` claim-nodes into Neo4j.

The stage can load requirement evidence from an explicit `--requirements` file,
from sibling `requirements.json` / `requirements_full.json`, or from PostgreSQL
by document version. If the sibling requirement files are stale or unusable, the
run warns and continues with the next source.

```powershell
uv run marag generate-test-scenarios `
  --user-stories generated\SIIMCS\req\<RUN_ID>\user_stories.json `
  --project SIIMCS `
  --json-output

uv run marag generate-test-scenarios `
  --document-version-id <DV-ID> `
  --project SIIMCS `
  --json-output
```

Useful variations:

```powershell
uv run marag generate-test-scenarios --user-stories .\path\to\user_stories.json --requirements .\path\to\requirements.json --project PROJECT_1 --top-k 4
uv run marag generate-test-scenarios --user-stories .\path\to\user_stories.json --project PROJECT_1 --reasoning-provider azure_openai --embedding-provider azure_openai --reranker-provider huggingface
uv run marag generate-test-scenarios --user-stories .\path\to\user_stories.json --project PROJECT_1 --hfil
uv run marag generate-test-scenarios --user-stories .\path\to\user_stories.json --project PROJECT_1 --no-hfil
uv run marag generate-test-scenarios --user-stories .\path\to\user_stories.json --project PROJECT_1 --hfil --thread-id RUN-123:hfil --emit-md
uv run marag artifact verify-test-scenarios generated\SIIMCS\req\<RUN_ID>\test_scenarios.json
```

Optional `.env` tuning:

```powershell
TEST_SCENARIO_TOP_K=4
TEST_SCENARIO_DENSE_K=8
TEST_SCENARIO_SPARSE_K=8
TEST_SCENARIO_NEIGHBOR_WINDOW=1
TEST_SCENARIO_MAX_NEW_TOKENS=
```

`test_scenarios.json` is written beside the input user-story file, or under
`generated/<PROJECT>/test_scenarios/<RUN_ID>/` when loaded from PostgreSQL.
HFIL runs after permanent scenario IDs are assigned and before persistence.
`test_scenarios.json` remains the canonical public scenario projection;
Markdown is emitted only with `--emit-md`.

## 10. Reconcile Local JSON

Use reconcile to re-materialize local generated artifacts from PostgreSQL after
a crash between DB commit and JSON write, or when local JSON is stale:

```powershell
uv run marag reconcile --project PROJECT_1
uv run marag reconcile --project PROJECT_1 --document-version <DV-ID>
uv run marag reconcile --project PROJECT_1 --json-output
```

## 11. Observability

### Run files

Every command creates a session directory under:

```text
.generated/<PROJECT>/run/
```

Typical files:

- `run_<stamp>.log`
- `run_<stamp>.jsonl`
- `req_<stamp>.json`

### Inspect logs

```powershell
Get-ChildItem .generated\<PROJECT>\run | Sort-Object LastWriteTime -Descending
Get-Content .generated\<PROJECT>\run\run_<stamp>.log -Tail 200
Get-Content .generated\<PROJECT>\run\run_<stamp>.jsonl -Tail 200
```

### Inspect run state

```powershell
uv run marag run status <RUN-ID>
uv run marag run resume <RUN-ID>
```

### Inspect the generated requirement artifact

```powershell
uv run marag artifact verify .generated\<PROJECT>\run\req_<stamp>.json
```

### Inspect database state

Local trace mode:

```powershell
Get-Content runtime\staging\postgres_records.jsonl -Tail 200
Get-Content runtime\staging\neo4j_projection.jsonl -Tail 200
Get-ChildItem runtime\databases\chroma
```

Real PostgreSQL:

```powershell
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select count(*) from ingestion_runs;"
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select count(*) from document_versions;"
```

Real Neo4j:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <password> "MATCH (n) RETURN count(n);"
```

### Where to find file paths after ingest

- Run log: `.generated/<PROJECT>/run/run_<stamp>.log`
- Machine-readable run sidecar: `.generated/<PROJECT>/run/run_<stamp>.jsonl`
- Requirement artifact: `.generated/<PROJECT>/run/req_<stamp>.json`
- Manifest: `runtime/staging/<RUN-ID>/chunk_manifest.json`
- PostgreSQL local trace: `runtime/staging/postgres_records.jsonl`
- Neo4j local trace: `runtime/staging/neo4j_projection.jsonl`
- Chroma persistence: `runtime/databases/chroma/`

## Suggested Order

1. Copy `.env.example` to `.env`
2. `uv sync`
3. `uv run marag version`
4. `uv run marag config-check`
5. `uv run marag doctor`
6. Start or clean databases as needed
7. `uv run marag db-check`
8. Run `uv run marag ingest ...`
9. Run `uv run marag generate-user-stories --requirements ... --project ...`
10. Run `uv run marag generate-test-scenarios --user-stories ... --project ...`
11. Optional: run `uv run marag reconcile --project <PROJECT>` to repair local JSON mirrors.
12. Inspect `.generated/<PROJECT>/run/` and `runtime/staging/`
