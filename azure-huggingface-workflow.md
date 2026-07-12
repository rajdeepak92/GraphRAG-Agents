# Azure OpenAI + Hugging Face Reranker Workflow

This guide sets up Agentic-GraphRAG from scratch with:

- reasoning: Azure OpenAI chat/completions deployment
- embeddings: Azure OpenAI embeddings deployment
- reranking: Hugging Face reranker model
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
- Azure OpenAI resource with a chat/reasoning deployment.
- Azure OpenAI resource with an embeddings deployment.
- Azure OpenAI API key for the resource.
- Hugging Face account/token if the reranker model is private or gated.

Azure requirements:

- `AZURE_OPENAI_ENDPOINT` must be the resource endpoint, for example
  `https://my-resource.openai.azure.com/`.
- `AZURE_OPENAI_REASONING_DEPLOYMENT` is the Azure deployment name, not
  necessarily the base model name.
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` is the Azure deployment name.
- The API key must belong to the same Azure OpenAI resource as the deployments.
- The API version must support both selected deployments.

Hugging Face reranker requirements:

- Use an exact Hub model ID for `HUGGINGFACE_RERANKER_MODEL`.
- If private/gated, accept license terms and use a token with read access.

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

Install Azure OpenAI and Hugging Face runtime dependencies:

```powershell
uv sync --dev --extra azure --extra local-llm
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

Expected DSN format:

```powershell
POSTGRES_DSN=postgresql://marag:pwd@127.0.0.1:5432/marag
```

## 4. Neo4j Setup

Start Neo4j before running the pipeline.

Verify Bolt connectivity:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <neo4j-password> "RETURN 1;"
```

Use these `.env` values for the default local Neo4j database:

```powershell
NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<neo4j-password>
NEO4J_DATABASE=neo4j
```

## 5. ChromaDB Setup

No separate Chroma service is required. The project persists Chroma under:

```text
runtime/databases/chroma/
```

The directory is created automatically during pipeline runs.

## 6. Configure `.env` For Azure + Hugging Face

Edit `.env` and set this profile.

```powershell
APP_ENV=development
LOG_LEVEL=INFO

REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface

POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:pwd@127.0.0.1:5432/marag

NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<neo4j-password>
NEO4J_DATABASE=neo4j

AZURE_OPENAI_ENDPOINT=https://<resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<azure-openai-key>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=<chat-or-reasoning-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>

HF_TOKEN=<hf-read-token-if-reranker-private>
HUGGINGFACE_TOKEN=
HUGGING_FACE_HUB_TOKEN=
HUGGINGFACE_REASONING_MODEL=
HUGGINGFACE_EMBEDDING_MODEL=
HUGGINGFACE_RERANKER_MODEL=<org/private-or-public-reranker-model>
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

- Keep `RERANKER_MODEL_PROVIDER=huggingface`; stage 3 and stage 4 require a
  real reranker provider.
- Leave `HUGGINGFACE_REASONING_MODEL` and `HUGGINGFACE_EMBEDDING_MODEL` blank
  to avoid accidental local model loading in this Azure profile.
- If the reranker is private, set `HF_TOKEN`.
- Azure deployment names are configured separately from Hugging Face model IDs.

## 7. Configuration And Provider Checks

Verify the project loads the intended providers:

```powershell
uv run python -m multi_agentic_graph_rag config-check
```

Expected provider lines:

```text
reasoning_model.provider = azure_openai
embedding_model.provider = azure_openai
reranker_model.provider = huggingface
```

Check Hugging Face dependency/token state for the reranker:

```powershell
uv run python -m multi_agentic_graph_rag hf-check
```

Check reranker repository access if it is private:

```powershell
@'
from huggingface_hub import HfApi
from multi_agentic_graph_rag.config.config_loader import load_config

settings = load_config()
api = HfApi(token=settings.huggingface.token)
info = api.model_info(settings.huggingface.reranker_model)
print(f"PASS {info.id}")
'@ | uv run python -
```

## 8. Azure OpenAI Connection Checks

Run a direct Azure embeddings check:

```powershell
@'
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.llm_models.azure_openai import AzureOpenAIEmbeddingModel

settings = load_config()
embedder = AzureOpenAIEmbeddingModel(settings.azure_openai)
vectors = embedder.embed_documents(["connection check"])
print(f"PASS embeddings vectors={len(vectors)} dimensions={len(vectors[0])}")
'@ | uv run python -
```

Run a direct Azure reasoning check:

```powershell
@'
from pydantic import BaseModel, ConfigDict

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.llm_models.azure_openai import AzureOpenAIReasoningModel

class Ping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ok: bool

settings = load_config()
reasoner = AzureOpenAIReasoningModel(settings.azure_openai)
reasoner.set_system_message("Return strict JSON only.")
result = reasoner.generate_structured(prompt='Return {"ok": true}.', schema=Ping)
print(f"PASS reasoning ok={result.ok}")
'@ | uv run python -
```

If either check fails:

- Confirm endpoint includes `https://` and ends with `.openai.azure.com/`.
- Confirm the API key belongs to that resource.
- Confirm deployment names are exact.
- Confirm the API version supports the deployments.
- Confirm network/firewall/proxy access to the Azure endpoint.

## 9. Database Connection Checks

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

## 10. Reset Databases And Runtime State

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

## 11. Prepare An Input Document

Place the source document under the inbox:

```powershell
New-Item -ItemType Directory -Force documents\inbox\PROJECT_1 | Out-Null
Copy-Item .\path\to\source-document.pdf documents\inbox\PROJECT_1\source-document.pdf
```

## 12. Run Ingestion With Azure OpenAI

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project PROJECT_1 `
  --document documents\inbox\PROJECT_1\source-document.pdf `
  --version V1 `
  --logical-name source-document `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
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

Store the latest run ID:

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
  --reasoning-provider azure_openai `
  --json-output
```

The audit copy is written to
`generated\PROJECT_1\kg\<RUN_ID>\knowledge_graph.json`; Neo4j remains the source
of truth for the graph. With `KNOWLEDGE_GRAPH_ENABLED=true` and shadow mode on,
later story/scenario retrieval records comparison snapshots to PostgreSQL without
changing generated output; the `GRAPH_PRIMARY_*` flags opt a stage into
assertion-hop retrieval.

## 13. Generate User Stories With Azure + Hugging Face Reranker

```powershell
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --reranker-provider huggingface `
  --json-output
```

Verify:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify-user-stories `
  generated\PROJECT_1\req\$RUN_ID\user_stories.json
```

## 14. Generate Test Scenarios With Azure + Hugging Face Reranker

Non-interactive path:

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
```

Interactive HFIL path:

```powershell
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --user-stories generated\PROJECT_1\req\$RUN_ID\user_stories.json `
  --requirements generated\PROJECT_1\req\$RUN_ID\requirements.json `
  --project PROJECT_1 `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
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

## 15. Coverage And Reconcile

Coverage:

```powershell
uv run python -m multi_agentic_graph_rag coverage --project PROJECT_1
```

Repair local JSON mirrors from PostgreSQL:

```powershell
uv run python -m multi_agentic_graph_rag reconcile --project PROJECT_1
```

Verify artifacts after reconcile:

```powershell
uv run python -m multi_agentic_graph_rag artifact verify-user-stories `
  generated\PROJECT_1\req\$RUN_ID\user_stories.json

uv run python -m multi_agentic_graph_rag artifact verify-test-scenarios `
  generated\PROJECT_1\req\$RUN_ID\test_scenarios.json
```

## 16. Run A V2 Document Version

Use the same `--logical-name` to preserve document lineage.

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project PROJECT_1 `
  --document documents\inbox\PROJECT_1\source-document-v2.pdf `
  --version V2 `
  --logical-name source-document `
  --replace-version `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --json-output
```

Then repeat user-story and test-scenario generation for the new run directory.

Expected behavior:

- New requirements append new child stories/scenarios.
- Updated requirements preserve requirement lineage and update children in place.
- Explicitly outdated requirements cascade-delete child stories/scenarios.
- Unchanged requirements do not duplicate existing children.

## 17. Troubleshooting

Azure embeddings fail:

- Confirm `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`.
- Confirm the deployment is an embedding model deployment.
- Confirm the API key/resource endpoint pair is correct.

Azure reasoning fails:

- Confirm `AZURE_OPENAI_REASONING_DEPLOYMENT`.
- Confirm the deployment supports chat completions.
- Confirm the API version supports the deployment.
- Inspect saved `llm_response_*.txt` files when `LOG_LLM_RESPONSES=true`.

Hugging Face reranker fails:

- Confirm `HF_TOKEN` if the reranker is private.
- Confirm `HUGGINGFACE_RERANKER_MODEL`.
- Confirm local memory/GPU is sufficient for the reranker.

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

Provider mismatch:

- Run `config-check`.
- Confirm `REASONING_MODEL_PROVIDER=azure_openai`.
- Confirm `EMBEDDING_MODEL_PROVIDER=azure_openai`.
- Confirm `RERANKER_MODEL_PROVIDER=huggingface`.

