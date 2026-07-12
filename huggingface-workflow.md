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

KNOWLEDGE_GRAPH_ENABLED=true
KNOWLEDGE_GRAPH_SHADOW_MODE=true
GRAPH_PRIMARY_STORY=true
GRAPH_PRIMARY_SCENARIO=true
KNOWLEDGE_GRAPH_EXPANSION_K=6
KNOWLEDGE_GRAPH_MIN_ASSERTIONS=3

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

Knowledge-graph (GraphRAG) profile — **on by default**:

- `KNOWLEDGE_GRAPH_ENABLED=true` makes ingestion build the semantic knowledge
  graph (entities + subject/predicate/object assertions with exact-quote
  evidence) into Neo4j automatically, right after chunk projection and Chroma
  indexing. No separate build step is required for the normal flow.
- `GRAPH_PRIMARY_STORY=true` / `GRAPH_PRIMARY_SCENARIO=true` make user-story and
  test-scenario generation ground on those assertions (retrieved from Neo4j
  content relations) instead of raw chunk text. When a requirement is not
  sufficiently grounded (no mandatory anchor, or fewer than
  `KNOWLEDGE_GRAPH_MIN_ASSERTIONS` selected), that one requirement falls back to
  the legacy chunk retrieval and logs `status="graph_fallback"`.
- `KNOWLEDGE_GRAPH_SHADOW_MODE=true` additionally records retrieval comparison
  snapshots to PostgreSQL; it is independent of the graph-primary flags.
- `KNOWLEDGE_GRAPH_EXPANSION_K` bounds one-hop entity expansion;
  `KNOWLEDGE_GRAPH_MIN_ASSERTIONS` is the grounding-gate minimum.
- **Explicit opt-out (legacy chunk-only path):** set
  `KNOWLEDGE_GRAPH_ENABLED=false`. Ingestion then skips the knowledge-graph
  build and generation uses hybrid chunk retrieval only.
- Every field above is also settable in `config.json` under `knowledge_graph`;
  environment variables win over `config.json`, which wins over built-in
  defaults.

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

With the default GraphRAG profile (`KNOWLEDGE_GRAPH_ENABLED=true`), the same
ingest run also builds the semantic knowledge graph for this document version and
projects it into Neo4j (entities, assertions, text-unit evidence), then moves the
document's active-knowledge pointer to this version. If the knowledge-graph build
fails, the ingestion run fails clearly rather than continuing with an empty
graph. Disable this with `KNOWLEDGE_GRAPH_ENABLED=false` to run the legacy
chunk-only path.

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

### Rebuild The Source-Knowledge Graph (standalone)

With the default profile, the knowledge graph is already built during ingest, so
you normally do not need this step. Run it standalone only to rebuild the graph
for an existing document version (for example after tuning extraction) or when
you ingested with `KNOWLEDGE_GRAPH_ENABLED=false` and now want the graph.

It extracts the same evidence-grounded graph (entities plus
subject/predicate/object assertions with exact-quote evidence and atomic text
units), uses the reasoning model only, is idempotent (safe to re-run), and moves
the active-knowledge pointer on success. Use the `document_version_id` (`DV-...`)
reported by `ingest` (it is also present in the requirement artifact):

```powershell
uv run python -m multi_agentic_graph_rag build-knowledge-graph `
  --project PROJECT_1 `
  --document-version-id <DV-ID> `
  --json-output
```

The audit copy is written to
`generated\PROJECT_1\kg\<RUN_ID>\knowledge_graph.json`; Neo4j remains the source
of truth for the graph.

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

With `GRAPH_PRIMARY_STORY=true` (default), each requirement's context is retrieved
from the Neo4j knowledge graph (anchor assertions plus one-hop entity expansion)
and the prompt renders an "AUTHORITATIVE FACTS" / "RELATED CONTEXT" split. This
requires the knowledge graph to exist for the document version: if it is missing,
the command fails with a `ConfigurationError` telling you to run ingest with
`KNOWLEDGE_GRAPH_ENABLED=true` (or `build-knowledge-graph`), or to set
`KNOWLEDGE_GRAPH_ENABLED=false` for the legacy chunk path. The persisted stories
also record which retrieved context produced them (`generation_context_run_id`,
`retrieved_assertion_ids`, `retrieved_chunk_ids`, context mode), separate from the
requirement-source evidence chunks.

Verify:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify-user-stories `
  generated\PROJECT_1\req\$RUN_ID\user_stories.json
```

## 13. Generate Test Scenarios With Hugging Face Models

With `GRAPH_PRIMARY_SCENARIO=true` (default), scenario context is retrieved from
the knowledge graph using the user-story query and the linked requirement's
evidence as anchors, rendered as "AUTHORITATIVE FACTS FOR THIS STORY" vs.
"RELATED CONSTRAINTS AND CONDITIONS". The same missing-graph preflight and
per-story `graph_fallback` behavior as user stories apply, and scenarios likewise
persist their `generation_context_run_id` / retrieved-context trace.

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

Coverage prints per-requirement status plus a summary line: total active
requirements, requirements with stories and the story-coverage percentage, and
requirements fully covered by scenarios with the end-to-end scenario-coverage
percentage (zero-safe).

```powershell
uv run python -m multi_agentic_graph_rag coverage --project PROJECT_1
```

Scope coverage to a single document version (denominator counts only
requirements present in that version, so a later version is not diluted by
earlier-only requirements). Add `--json-output` for the machine-readable report:

```powershell
uv run python -m multi_agentic_graph_rag coverage `
  --project PROJECT_1 --document-version-id <DV-ID> --json-output
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

The V2 ingest also rebuilds the knowledge graph for the new version and
reconciles it against V1: assertions are matched by a cross-version lineage key,
so a changed value supersedes the prior assertion, a dropped fact is retired, and
a genuinely new fact is added. The active-knowledge pointer moves to V2 only after
the projection succeeds, and story/scenario retrieval then reads **current
(V2) knowledge only** — superseded and retired assertions are excluded.

Then repeat user-story and test-scenario generation for the new run directory.

Expected behavior:

- New requirements append new child stories/scenarios.
- Updated requirements preserve requirement lineage and update children in place.
- Explicitly outdated requirements cascade-delete child stories/scenarios.
- Unchanged requirements do not duplicate existing children.
- Knowledge-graph assertions supersede/retire/add across versions; entity
  resolution reuses the prior active version's entities (not every stale entity),
  and historical assertions are retained (not hard-deleted) unless an explicit,
  disabled-by-default retention pass is run.

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

Generation fails with "has no semantic knowledge graph":

- The document version was ingested without the knowledge graph while
  graph-primary generation is on. Re-ingest with `KNOWLEDGE_GRAPH_ENABLED=true`,
  or run `build-knowledge-graph --project ... --document-version-id <DV-ID>`.
- Or set `KNOWLEDGE_GRAPH_ENABLED=false` to use the legacy chunk-only path.

Stories/scenarios look chunk-based instead of assertion-grounded:

- Check the run log for `status="graph_fallback"` entries: a requirement/story
  missed the grounding gate (no mandatory anchor, or fewer than
  `KNOWLEDGE_GRAPH_MIN_ASSERTIONS` assertions) and fell back to chunk retrieval.
- Confirm the knowledge graph was actually built for the version (it is, by
  default, during ingest) and lower `KNOWLEDGE_GRAPH_MIN_ASSERTIONS` if the
  source is sparse.

