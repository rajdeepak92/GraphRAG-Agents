# Agentic-GraphRAG Runbook

## 1. Clone And Bootstrap

```powershell
git clone <repo-url>
cd Agentic-GraphRAG
Copy-Item .env.example .env
& C:\Users\rdmpr\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2. Sync Dependencies

Use the Hugging Face runtime:

```powershell
uv sync --dev --extra local-llm
```

Use Azure OpenAI for reasoning and embeddings:

```powershell
uv sync --dev --extra azure --extra local-llm
```

## 3. Configure `.env`

For Azure OpenAI, set these values in `.env`:

```powershell
REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface

AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-secret>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=<reasoning-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>

LOG_LLM_RESPONSES=true
```

For the Hugging Face runtime, keep the provider values set to `huggingface`
and populate the model names and token fields in `.env.example`.

For local service automation, use the PostgreSQL and Neo4j variables only:

```powershell
MARAG_POSTGRES_AUTO_START=true
MARAG_POSTGRES_SERVICE_NAME=postgresql-x64-18
MARAG_NEO4J_AUTO_START=true
MARAG_NEO4J_START_MODE=desktop_dbms
NEO4J_DBMS_HOME=D:\path\to\neo4j\dbms
NEO4J_JAVA_HOME=D:\path\to\neo4j\java
MARAG_ALLOW_SERVICE_STOP=false
MARAG_POSTGRES_ALLOW_STOP=false
```

## 4. First Verification

```powershell
uv run python -m multi_agentic_graph_rag version
uv run python -m multi_agentic_graph_rag config-check
```

If you are running the Hugging Face stack, also run:

```powershell
uv run python -m multi_agentic_graph_rag hf-check --load-model
```

## 5. Database Checks

```powershell
uv run python -m multi_agentic_graph_rag db-check
```

This checks PostgreSQL, Neo4j, and Chroma in that order.

## 6. Cleanup Local Databases

PostgreSQL app-managed schema only:

```powershell
uv run python -m multi_agentic_graph_rag postgres-reset --yes
uv run python -m multi_agentic_graph_rag db-check
```

Neo4j disposable reset:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <password> "MATCH (n) DETACH DELETE n;"
```

Chroma local store reset:

```powershell
Remove-Item -Recurse -Force runtime\databases\chroma -ErrorAction SilentlyContinue
```

Full local cleanup, if needed:

```powershell
Remove-Item -Recurse -Force generated, .generated, .global_cache\hf_home -ErrorAction SilentlyContinue
```

## 7. Run Ingest

```powershell
uv run marag ingest `
  --project <PROJECT> `
  --document <PATH-TO-DOCUMENT> `
  --version <VERSION> `
  --json-output
```

Optional flags:

```powershell
--logical-name <NAME>
--replace-version
--reasoning-provider azure_openai
--embedding-provider azure_openai
```

Generated output is local-only under:

```text
generated/<PROJECT>/req/<RUN_ID>/
```

That folder contains compact `requirements.json`, full audit
`requirements_full.json`, `run.log`, `run.jsonl`, `chunk_manifest.json`, and
any saved `llm_response_*.txt` files.

## 8. Generate User Stories

User-story generation is a standalone stage (`generate-user-stories`) that
consumes the requirement artifact and produces user stories per requirement.
It uses hybrid retrieval - Chroma dense search, Neo4j full-text search, Neo4j
multi-hop over the chunk graph, and the reranker - then writes the artifact to
JSON and PostgreSQL.

Required stack: real reasoning, embedding, and reranker providers,
`POSTGRES_MODE=postgres`, and `NEO4J_MODE=neo4j`.

From a local requirement artifact:

```powershell
uv run marag generate-user-stories `
  --requirements generated\<PROJECT>\req\<RUN_ID>\requirements.json `
  --project <PROJECT> `
  --json-output
```

From PostgreSQL when the local artifact is absent:

```powershell
uv run marag generate-user-stories `
  --document-version-id <DV-ID> `
  --project <PROJECT> `
  --json-output
```

Optional flags:

```powershell
--reasoning-provider azure_openai
--embedding-provider azure_openai
--reranker-provider huggingface
--top-k 4
```

Output: `user_stories.json` is written beside the input requirements file, or
under `generated/<PROJECT>/user_stories/<RUN_ID>/` when loaded from PostgreSQL.

Persistence:

- PostgreSQL: `user_story_artifacts` and `user_stories`.
- Neo4j: `(:UserStory)-[:COVERS_REQUIREMENT]->(:Requirement)`,
  `(:Requirement)-[:EVIDENCED_BY_CHUNK]->(:Chunk)`, and
  `(:DocumentVersion)-[:HAS_USER_STORY]->(:UserStory)`.

## 9. Generate Test Scenarios

Test-scenario generation is a standalone stage (`generate-test-scenarios`) that
consumes `user_stories.json` and produces `test_scenarios.json`. It uses the
same hybrid retrieval stack as stage 3 and writes rows to PostgreSQL plus
coverage claim-nodes to Neo4j.

Requirement evidence loading is lenient unless `--requirements` is supplied.
The stage tries sibling `requirements.json`, sibling `requirements_full.json`,
then PostgreSQL by document version. If none are usable, it warns and
continues with dense+sparse retrieval only.

```powershell
uv run marag generate-test-scenarios `
  --user-stories generated\<PROJECT>\req\<RUN_ID>\user_stories.json `
  --project <PROJECT> `
  --json-output

uv run marag generate-test-scenarios `
  --document-version-id <DV-ID> `
  --project <PROJECT> `
  --json-output
```

Optional tuning:

```powershell
TEST_SCENARIO_TOP_K=4
TEST_SCENARIO_DENSE_K=8
TEST_SCENARIO_SPARSE_K=8
TEST_SCENARIO_NEIGHBOR_WINDOW=1
TEST_SCENARIO_MAX_NEW_TOKENS=
```

## 10. Observability

Inspect the latest run:

```powershell
uv run python -m multi_agentic_graph_rag run status <RUN-ID>
uv run python -m multi_agentic_graph_rag artifact verify generated\<PROJECT>\req\<RUN-ID>\requirements_full.json
Get-Content generated\<PROJECT>\req\<RUN-ID>\run.log -Tail 200
Get-Content generated\<PROJECT>\req\<RUN-ID>\run.jsonl -Tail 200
```

What to expect:

- `run.jsonl` contains structured events for the command session.
- `run.log` contains the human-readable stream.
- Tokens, API keys, and DSNs are redacted by the logger.
- Azure raw completions are saved as `llm_response_*.txt` in the run folder
  when parsing fails, and successful responses are also captured when
  `LOG_LLM_RESPONSES=true`.

## 11. Notes

- `run status` still supports the legacy `.generated/<PROJECT>/run/` path.
- The repo keeps generated outputs and runtime data out of version control.
- `generate-user-stories` writes its session log under `.generated/<PROJECT>/run/`; the `user_stories.json` artifact lands beside the input requirements file.
- `generate-test-scenarios` writes its session log under `.generated/<PROJECT>/run/`; the `test_scenarios.json` artifact lands beside the input user-story file.

