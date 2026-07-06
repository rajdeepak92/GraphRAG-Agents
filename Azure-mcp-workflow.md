# Azure OpenAI End-to-End Workflow (with MCP)

Run the full pipeline — **ingestion → requirement discovery → user stories →
test scenarios** — using **Azure OpenAI for reasoning + embedding** and
**Hugging Face for the reranker** (reranker has no Azure provider).

| Role | Provider | Source |
|---|---|---|
| Reasoning | `azure_openai` | Azure deployment (e.g. a GPT-class chat model) |
| Embedding | `azure_openai` | Azure embedding deployment (e.g. `text-embedding-3-large`) |
| Reranker | `huggingface` | `BAAI/bge-reranker-base` (local) |

Stores: **PostgreSQL** (generated artifacts + ledger), **Neo4j** (chunk graph +
traceability claim-nodes), **ChromaDB** (chunk embeddings).

Windows-first; use `uv run`, never generated launchers. All commands are
PowerShell.

---

## 0. Prerequisites

- Python 3.12 (`>=3.12,<3.13`) and [`uv`](https://docs.astral.sh/uv/).
- An **Azure OpenAI resource** with two deployments:
  - a **reasoning/chat** deployment (used for requirement discovery, user-story,
    and test-scenario generation), and
  - an **embedding** deployment (used for Chroma indexing + dense retrieval).
- PostgreSQL 14+ reachable via a libpq URL.
- Neo4j 5/6 reachable over Bolt.
- Some local disk/compute for the **Hugging Face reranker only**
  (`BAAI/bge-reranker-base`, ~1 GB, CPU is fine) — no local LLM weights needed.

---

## 1. Bootstrap

```powershell
git clone <repo-url>
cd Agentic-GraphRAG
Copy-Item .env.example .env
& C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Azure **and** the Hugging Face extra (needed for the reranker), plus the
MCP server:

```powershell
uv sync --dev --extra azure --extra local-llm --extra mcp
uv run python -c "import mcp; print('mcp ok')"
```

- `--extra azure` → `openai` + `azure-identity`.
- `--extra local-llm` → **required** because the reranker is Hugging Face.
- `--extra mcp` → the project-local stdio MCP server (`marag-mcp`).

---

## 2. Configure `.env` (Azure + HF-reranker profile)

```powershell
APP_ENV=development
LOG_LEVEL=INFO

# Reasoning + embedding on Azure; reranker on Hugging Face.
REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface

# Azure OpenAI.
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-secret>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=<reasoning-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>

# Hugging Face reranker (only the reranker model is used locally).
HUGGINGFACE_RERANKER_MODEL=BAAI/bge-reranker-base
HF_TOKEN=                          # only needed if the reranker model is gated
HUGGINGFACE_OFFLINE=false          # set true after the reranker is cached

# Real stores are REQUIRED for a real end-to-end run.
POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:<password>@127.0.0.1:5432/marag

NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

LOG_LLM_RESPONSES=true

# Optional stage-4 (test scenario) retrieval/generation tuning.
TEST_SCENARIO_TOP_K=4
TEST_SCENARIO_DENSE_K=8
TEST_SCENARIO_SPARSE_K=8
TEST_SCENARIO_NEIGHBOR_WINDOW=1
```

Notes:
- Use the **deployment names**, not the underlying model names, for the two Azure
  `*_DEPLOYMENT` values.
- `AZURE_OPENAI_API_VERSION=2024-10-21` is the tested default; match it to what
  your resource supports.
- The reranker still downloads once (Hugging Face). `HF_TOKEN` is only required
  for gated models. Flip `HUGGINGFACE_OFFLINE=true` after it is cached.
- Azure raw completions are saved as `llm_response_*.txt` in the run dir when
  parsing fails (and always when `LOG_LLM_RESPONSES=true`).

---

## 3. Verify config

```powershell
uv run marag version
uv run marag config-check
```

`config-check` confirms the provider split and that the Azure settings are
present. (`hf-check` is only relevant to the local LLM stack; the reranker is
loaded lazily during retrieval.)

---

## 4. One-time schema migration

This build adds managed PostgreSQL tables (including `user_story_evidence` and
`test_scenario_evidence`). Reset once against your disposable/local database,
then confirm all three stores:

```powershell
uv run marag postgres-reset --yes
uv run marag db-check      # checks PostgreSQL, Neo4j, Chroma in order
```

---

## 5. Manual end-to-end run (CLI)

Because the providers are already set in `.env`, no per-command overrides are
needed. (You can still override per run with `--reasoning-provider azure_openai
--embedding-provider azure_openai --reranker-provider huggingface`.)

### 5.1 Ingest → requirement discovery

```powershell
uv run marag ingest `
  --project PROJECT_1 `
  --document .\documents\inbox\PROJECT_1\SIIMCS_BRD_V1.pdf `
  --version 1.0 `
  --json-output
```

Output dir (gitignored): `generated/PROJECT_1/req/<RUN_ID>/` with
`requirements.json`, `requirements_full.json`, `run.log`, `run.jsonl`,
`chunk_manifest.json`.

### 5.2 Generate user stories

```powershell
uv run marag generate-user-stories `
  --requirements generated\PROJECT_1\req\<RUN_ID>\requirements.json `
  --project PROJECT_1 `
  --json-output
```

Hybrid retrieval = Chroma dense (Azure embeddings) + Neo4j BM25 + Neo4j
multi-hop, fused by the **Hugging Face** reranker. Writes `user_stories.json`
beside the requirements file.

### 5.3 Generate test scenarios

```powershell
uv run marag generate-test-scenarios `
  --user-stories generated\PROJECT_1\req\<RUN_ID>\user_stories.json `
  --project PROJECT_1 `
  --json-output
```

Writes `test_scenarios.json` beside the user-stories file.

> Do **not** run `postgres-reset` between the user-story and test-scenario stages —
> the scenario stage may fall back to PostgreSQL for requirement evidence.

### 5.4 Coverage & verification

```powershell
uv run marag coverage --project PROJECT_1
uv run marag artifact verify-user-stories   generated\PROJECT_1\req\<RUN_ID>\user_stories.json
uv run marag artifact verify-test-scenarios generated\PROJECT_1\req\<RUN_ID>\test_scenarios.json
```

Master-sheet traceability checks:

```powershell
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select story_id, requirement_id, run_id, origin from user_stories;"
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select story_id, chunk_id, requirement_id from user_story_evidence;"
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select scenario_id, story_id, requirement_id, chunk_id from test_scenario_evidence;"
```

---

## 6. MCP workflow (Claude CLI, no Docker)

The project ships a project-local stdio MCP server registered in `.mcp.json` as
`marag-mcp`. Secrets stay in `.env`; `.mcp.json` only launches the server, so the
Azure keys are read from `.env` at runtime.

### 6.1 Optional: Windows-native service automation

```powershell
MARAG_MCP_ENABLED=true
MARAG_MCP_STACK_MODE=windows_native

MARAG_POSTGRES_AUTO_START=true
MARAG_POSTGRES_SERVICE_NAME=postgresql-x64-18
MARAG_POSTGRES_HOST=127.0.0.1
MARAG_POSTGRES_PORT=5432

MARAG_NEO4J_AUTO_START=true
MARAG_NEO4J_START_MODE=desktop_dbms
NEO4J_DBMS_HOME=D:\path\to\neo4j\dbms
NEO4J_JAVA_HOME=D:\path\to\neo4j\java
MARAG_NEO4J_HTTP_PORT=7474
MARAG_NEO4J_BOLT_PORT=7687

MARAG_ALLOW_SERVICE_STOP=false
```

Safety defaults: no Docker, no raw SQL/Cypher, no DB reset, no process killing,
no service stop unless `MARAG_ALLOW_SERVICE_STOP=true`.

### 6.2 Register & smoke-check

```powershell
uv run python -m multi_agentic_graph_rag.mcp.server   # Ctrl+C after it starts
claude mcp list                                        # marag-mcp should appear
```

### 6.3 MCP tools

| Tool | Purpose |
|---|---|
| `marag_health_check` | Postgres service/port, Neo4j ports, Chroma, `db-check`. |
| `marag_start_stack` / `marag_stop_stack` | Start/stop the configured local stack (stop gated by flags). |
| `marag_ingest_document` | `project, document, version, [logical_name], [replace_version]`. |
| `marag_generate_user_stories` | `project`, `requirements`/`document_version_id`, provider/top_k overrides. |
| `marag_generate_test_scenarios` | `project`, `user_stories`/`requirements`/`document_version_id`, overrides. |
| `marag_run_full_pipeline` | Stack check/start → ingest → user stories → test scenarios → verify. |
| `marag_find_latest_artifacts` | Latest `requirements.json` / `user_stories.json` / `test_scenarios.json`. |
| `marag_verify_artifact` / `marag_open_run_status` | Verify an artifact / open a run's status. |

### 6.4 One-shot full pipeline from Claude CLI

With `marag-mcp` connected, ask Claude to run the full pipeline
(`marag_run_full_pipeline`):

```text
project=PROJECT_1
document=.\documents\inbox\PROJECT_1\SIIMCS_BRD_V1.pdf
version=1.0
```

The providers come from `.env` (Azure reasoning + embedding, HF reranker), so no
overrides are needed. If you want to force them per call, pass:

```text
reasoning_provider=azure_openai
embedding_provider=azure_openai
reranker_provider=huggingface
```

Then use `marag_find_latest_artifacts` (project=`PROJECT_1`) and
`marag_verify_artifact`.

---

## 7. Observability

```powershell
uv run marag run status <RUN-ID>
Get-Content generated\PROJECT_1\req\<RUN_ID>\run.log   -Tail 200
Get-Content generated\PROJECT_1\req\<RUN_ID>\run.jsonl -Tail 200
```

- Tokens/keys/DSNs are redacted by the logger.
- Azure raw completions land as `llm_response_*.txt` in the run dir on parse
  failure (and always with `LOG_LLM_RESPONSES=true`) — handy for prompt/response
  debugging.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `SchemaMismatchError` on any command | `uv run marag postgres-reset --yes` then `db-check` (expected once for the new evidence tables). |
| Auth / 401 from Azure | Recheck `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and that the **deployment names** (not model names) are correct. |
| `DeploymentNotFound` / 404 | The `*_DEPLOYMENT` value must match a deployment that exists in your resource; confirm `AZURE_OPENAI_API_VERSION` is supported. |
| Rate limits / 429 | Lower concurrency by keeping `DISCOVERY_BATCH_SIZE=1`; retry; or raise your Azure quota. |
| Reranker download/first-run slowness | The HF reranker downloads once; set `HUGGINGFACE_OFFLINE=true` after caching. Ensure `--extra local-llm` was installed. |
| `context_map.json already present` / checkpoint stage error | Fixed in this build (stage-specific `context_story.json` / `context_scenario.json`). Remove stale `context_map.json` from old runs. |
| Empty/degraded retrieval for stories | The document's chunks must be in Neo4j/Chroma; re-ingest the source first. |
| Neo4j `UnknownPropertyKey` (`section`) warning on PDFs | Suppressed in this build via driver notification config. |

## 9. Suggested order

1. `Copy-Item .env.example .env` and set the Azure + HF-reranker profile (§2)
2. `uv sync --dev --extra azure --extra local-llm --extra mcp`
3. `uv run marag version` / `config-check`
4. `uv run marag postgres-reset --yes` → `db-check`
5. `uv run marag ingest ...`
6. `uv run marag generate-user-stories ...`
7. `uv run marag generate-test-scenarios ...`
8. `uv run marag coverage --project ...` + verify artifacts
