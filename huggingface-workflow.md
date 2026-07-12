# Hugging Face Private Models Workflow

This guide sets up Agentic-GraphRAG from scratch with Hugging Face private
models for every model role:

- reasoning: Hugging Face causal/chat model
- embeddings: Hugging Face embedding model
- reranking: Hugging Face cross-encoder/reranker model
- persistence: PostgreSQL
- graph store: Neo4j
- vector store: ChromaDB local persistence

Commands are written for Windows PowerShell from the repository root.

## 1. Prerequisites

Install or verify:

- Python 3.12.
- `uv`.
- Git.
- PostgreSQL server with `psql` available on PATH.
- Neo4j server or Neo4j Desktop DBMS with `cypher-shell` available on PATH.
- A Hugging Face account with access to the private model repositories.
- A Hugging Face token with read access to the private models.
- Enough local CPU/GPU/RAM/disk for the selected private models.

Private Hugging Face model requirements:

- Accept any model license/gated-access terms in the Hugging Face UI.
- Confirm the token can read each private repository.
- Use exact Hub model IDs such as `org/private-reasoning-model`.
- Keep `HUGGINGFACE_OFFLINE=false` until all models are cached locally.

## 2. Clone And Bootstrap

```powershell
git clone <repo-url>
cd Agentic-GraphRAG
Copy-Item .env.example .env
```

Create and activate a Python 3.12 virtual environment:

```powershell
& C:\Users\rdmpr\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the local Hugging Face runtime stack:

```powershell
uv sync --dev --extra local-llm
```

Baseline CLI check:

```powershell
uv run python -m multi_agentic_graph_rag version
uv run python -m multi_agentic_graph_rag doctor
```

## 3. PostgreSQL Setup

Create a local app database and app user if they do not already exist.

Run as a PostgreSQL admin user:

```powershell
psql -h 127.0.0.1 -U postgres -c "CREATE USER marag WITH PASSWORD 'pwd';"
psql -h 127.0.0.1 -U postgres -c "CREATE DATABASE marag OWNER marag;"
psql -h 127.0.0.1 -U postgres -d marag -c "GRANT ALL ON SCHEMA public TO marag;"
```

If the user or database already exists, PostgreSQL may report an error. That is
acceptable if the final connection check passes.

Expected DSN format:

```powershell
POSTGRES_DSN=postgresql://marag:pwd@127.0.0.1:5432/marag
```

## 4. Neo4j Setup

Start Neo4j before running the pipeline.

For a local Neo4j server, confirm Bolt is reachable:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <neo4j-password> "RETURN 1;"
```

If using Neo4j Desktop:

- Start the DBMS from Neo4j Desktop.
- Ensure Bolt is enabled on port `7687`.
- Set `NEO4J_USERNAME=neo4j`.
- Set `NEO4J_PASSWORD=<neo4j-password>`.
- Set `NEO4J_DATABASE=neo4j` unless you created a different database.

## 5. ChromaDB Setup

No separate Chroma service is required. The project persists Chroma under:

```text
runtime/databases/chroma/
```

The directory is created automatically. Delete it only when you intentionally
want to rebuild the local vector index.

## 6. Configure `.env` For Private Hugging Face Models

Edit `.env` and set this profile.

```powershell
APP_ENV=development
LOG_LEVEL=INFO

REASONING_MODEL_PROVIDER=huggingface
EMBEDDING_MODEL_PROVIDER=huggingface
RERANKER_MODEL_PROVIDER=huggingface

POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:pwd@127.0.0.1:5432/marag

NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<neo4j-password>
NEO4J_DATABASE=neo4j

HF_TOKEN=<hf-read-token>
HUGGINGFACE_TOKEN=
HUGGING_FACE_HUB_TOKEN=

HUGGINGFACE_REASONING_MODEL=<org/private-reasoning-model>
HUGGINGFACE_EMBEDDING_MODEL=<org/private-embedding-model>
HUGGINGFACE_RERANKER_MODEL=<org/private-reranker-model>
HUGGINGFACE_OFFLINE=false
HUGGINGFACE_MAX_NEW_TOKENS=4096
DISCOVERY_BATCH_SIZE=1
LOG_LLM_RESPONSES=true

TEST_SCENARIO_TOP_K=4
TEST_SCENARIO_DENSE_K=8
TEST_SCENARIO_SPARSE_K=8
TEST_SCENARIO_NEIGHBOR_WINDOW=1
TEST_SCENARIO_MAX_NEW_TOKENS=

KNOWLEDGE_GRAPH_ENABLED=false
KNOWLEDGE_GRAPH_SHADOW_MODE=true
GRAPH_PRIMARY_STORY=false
GRAPH_PRIMARY_SCENARIO=false
KNOWLEDGE_GRAPH_EXPANSION_K=6

ENABLE_HFIL=false
HFIL_MATCH_THRESHOLD_PCT=60.0
HFIL_OUT_OF_CONTEXT_PCT=5.0
HFIL_COS_FLOOR=0.30
HFIL_COS_CEIL=0.80
DEDUP_RECALL_COSINE=0.55
HFIL_CHECKPOINTER=postgres
HFIL_EMIT_MD=false
```

Notes:

- `HF_TOKEN` is preferred.
- `HUGGINGFACE_TOKEN` and `HUGGING_FACE_HUB_TOKEN` are compatibility aliases.
- Keep `HUGGINGFACE_OFFLINE=false` for first run/cache warmup.
- Set `HUGGINGFACE_OFFLINE=true` only after the private models are already
  cached under `.global_cache`.
- For private/gated models, token access and license acceptance are mandatory.

## 7. Verify Hugging Face Private Model Access

First verify config values:

```powershell
uv run python -m multi_agentic_graph_rag config-check
```

Check Hugging Face dependency/token state without loading weights:

```powershell
uv run python -m multi_agentic_graph_rag hf-check
```

Check private repository visibility through the Hugging Face Hub API:

```powershell
@'
from huggingface_hub import HfApi
from multi_agentic_graph_rag.config.config_loader import load_config

settings = load_config()
api = HfApi(token=settings.huggingface.token)
model_ids = [
    settings.huggingface.reasoning_model,
    settings.huggingface.embedding_model,
    settings.huggingface.reranker_model,
]

for model_id in model_ids:
    info = api.model_info(model_id)
    print(f"PASS {info.id}")
'@ | uv run python -
```

Load the configured Hugging Face models:

```powershell
uv run python -m multi_agentic_graph_rag hf-check --load-model
```

If this fails:

- Confirm `HF_TOKEN` is set in `.env`.
- Confirm the token has read access.
- Confirm license/gated access has been accepted.
- Confirm model IDs are exact.
- Confirm the machine has enough memory/GPU for the reasoning model.

## 8. Database Connection Checks

Run the project database check:

```powershell
uv run python -m multi_agentic_graph_rag db-check
```

Expected:

- PostgreSQL: PASS.
- Neo4j: PASS.
- Chroma: PASS.

Direct PostgreSQL check:

```powershell
psql "postgresql://marag:pwd@127.0.0.1:5432/marag" -c "select current_database(), current_user;"
```

Direct Neo4j check:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <neo4j-password> "MATCH (n) RETURN count(n);"
```

Direct Chroma directory check:

```powershell
Get-ChildItem runtime\databases\chroma -Force
```

## 9. Reset Databases And Runtime State

Use these commands only for disposable/local development data.

Reset PostgreSQL app-managed tables:

```powershell
uv run python -m multi_agentic_graph_rag postgres-reset --yes
```

If the PostgreSQL DSN points to a non-local host and you intentionally want to
reset it:

```powershell
uv run python -m multi_agentic_graph_rag postgres-reset --yes --allow-nonlocal
```

Reset Neo4j:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <neo4j-password> "MATCH (n) DETACH DELETE n;"
```

Reset Chroma:

```powershell
Remove-Item -Recurse -Force runtime\databases\chroma -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force runtime\databases\chroma | Out-Null
```

Optional cleanup of generated local artifacts and run logs:

```powershell
Remove-Item -Recurse -Force generated\<PROJECT> -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .generated\<PROJECT> -ErrorAction SilentlyContinue
```

Recheck after reset:

```powershell
uv run python -m multi_agentic_graph_rag db-check
```

## 10. Prepare An Input Document

Place the source document under the inbox:

```powershell
New-Item -ItemType Directory -Force documents\inbox\PROJECT_1 | Out-Null
Copy-Item .\path\to\source-document.pdf documents\inbox\PROJECT_1\source-document.pdf
```

Supported parsing depends on installed document libraries, but this project
commonly uses text, PDF, DOCX, and related source files.

## 11. Run Ingestion With Hugging Face Models

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project PROJECT_1 `
  --document documents\inbox\PROJECT_1\source-document.pdf `
  --version V1 `
  --logical-name source-document `
  --json-output
```

The command writes:

```text
generated/PROJECT_1/req/<RUN_ID>/
  requirements.json
  requirements_full.json
  chunk_manifest.json
  run.log
  run.jsonl
```

Find the latest run directory:

```powershell
Get-ChildItem generated\PROJECT_1\req | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

Store the run ID in a shell variable:

```powershell
$RUN_ID = (Get-ChildItem generated\PROJECT_1\req | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name
```

Verify the requirement artifact:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify `
  generated\PROJECT_1\req\$RUN_ID\requirements_full.json
```

### Optional: Build The Source-Knowledge Graph

After ingestion you can extract an evidence-grounded source-knowledge graph
(entities plus subject/predicate/object assertions with exact-quote evidence and
atomic text units) and project it into Neo4j. This stage is optional, uses the
reasoning model only, and does not change requirement/story/scenario generation
unless the knowledge-graph flags are enabled (see section 6).

Use the `document_version_id` (`DV-...`) reported by `ingest` (it is also present
in the requirement artifact):

```powershell
uv run python -m multi_agentic_graph_rag build-knowledge-graph `
  --project PROJECT_1 `
  --document-version-id <DV-ID> `
  --json-output
```

The audit copy is written to
`generated\PROJECT_1\kg\<RUN_ID>\knowledge_graph.json`; Neo4j remains the source
of truth for the graph. With `KNOWLEDGE_GRAPH_ENABLED=true` and shadow mode on,
later story/scenario retrieval records comparison snapshots to PostgreSQL without
changing generated output; the `GRAPH_PRIMARY_*` flags opt a stage into
assertion-hop retrieval.

## 12. Generate User Stories With Hugging Face Models

```powershell
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider huggingface `
  --embedding-provider huggingface `
  --reranker-provider huggingface `
  --json-output
```

Verify:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify-user-stories `
  generated\PROJECT_1\req\$RUN_ID\user_stories.json
```

## 13. Generate Test Scenarios With Hugging Face Models

Non-interactive path:

```powershell
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --user-stories generated\PROJECT_1\req\$RUN_ID\user_stories.json `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider huggingface `
  --embedding-provider huggingface `
  --reranker-provider huggingface `
  --no-hfil `
  --json-output
```

Interactive HFIL path:

```powershell
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --user-stories generated\PROJECT_1\req\$RUN_ID\user_stories.json `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider huggingface `
  --embedding-provider huggingface `
  --reranker-provider huggingface `
  --hfil `
  --thread-id PROJECT_1-$RUN_ID-hfil
```

Inside HFIL:

- Type `remove duplicates` to run semantic duplicate detection.
- Type a feedback sentence to request one missing scenario.
- Type `exit` to finalize and persist.

Verify:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify-test-scenarios `
  generated\PROJECT_1\req\$RUN_ID\test_scenarios.json
```

## 14. Coverage And Reconcile

Coverage:

```powershell
uv run python -m multi_agentic_graph_rag coverage --project PROJECT_1
```

Repair local JSON mirrors from PostgreSQL:

```powershell
uv run python -m multi_agentic_graph_rag reconcile --project PROJECT_1
```

Verify DB-backed artifacts after reconcile:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify-user-stories `
  generated\PROJECT_1\req\$RUN_ID\user_stories.json

uv run python -m multi_agentic_graph_rag artifact verify-test-scenarios `
  generated\PROJECT_1\req\$RUN_ID\test_scenarios.json
```

## 15. Run A V2 Document Version

Use the same `--logical-name` to preserve document lineage.

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project PROJECT_1 `
  --document documents\inbox\PROJECT_1\source-document-v2.pdf `
  --version V2 `
  --logical-name source-document `
  --replace-version `
  --json-output
```

Then repeat user-story and test-scenario generation for the new run directory.

Expected behavior:

- New requirements append new child stories/scenarios.
- Updated requirements preserve requirement lineage and update children in place.
- Explicitly outdated requirements cascade-delete child stories/scenarios.
- Unchanged requirements do not duplicate existing children.

## 16. Troubleshooting

`hf-check --load-model` fails with authorization:

- Verify `HF_TOKEN` is loaded by `config-check`.
- Verify token read scope.
- Verify license/gated access was accepted.
- Verify exact private model IDs.

Model download is slow:

- Keep `HUGGINGFACE_OFFLINE=false`.
- Let the first load finish.
- Use `.global_cache` as the persistent cache.
- Set `HUGGINGFACE_OFFLINE=true` only after all required models are cached.

PostgreSQL fails:

- Confirm the server is running.
- Confirm `POSTGRES_DSN`.
- Run the direct `psql` check.
- Run `postgres-reset --yes` only on disposable/local DBs.

Neo4j fails:

- Confirm the DBMS is started.
- Confirm `NEO4J_PASSWORD`.
- Run the direct `cypher-shell` check.

Chroma problems:

- Delete `runtime\databases\chroma`.
- Rerun ingestion to rebuild indexes.

Generation produces invalid JSON:

- Keep `LOG_LLM_RESPONSES=true`.
- Inspect `generated\PROJECT_1\req\$RUN_ID\llm_response_*.txt`.
- Reduce `DISCOVERY_BATCH_SIZE` to `1`.
- Increase `HUGGINGFACE_MAX_NEW_TOKENS` if responses truncate.

