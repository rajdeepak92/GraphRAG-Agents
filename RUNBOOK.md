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

Use Azure OpenAI for reasoning and embeddings, while keeping the Hugging Face
reranker:

```powershell
uv sync --dev --extra azure --extra local-llm
```

Add the MCP extra when using Claude CLI tool automation:

```powershell
uv sync --dev --extra mcp
uv run python -c "import mcp; print('mcp ok')"
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
and populate the Hugging Face model names and token fields in `.env.example`.

For no-Docker MCP automation on Windows, also configure the local service names
and Neo4j DBMS paths:

```powershell
MARAG_MCP_ENABLED=true
MARAG_MCP_STACK_MODE=windows_native
MARAG_POSTGRES_AUTO_START=true
MARAG_POSTGRES_SERVICE_NAME=postgresql-x64-18
MARAG_NEO4J_AUTO_START=true
MARAG_NEO4J_START_MODE=desktop_dbms
NEO4J_DBMS_HOME=D:\path\to\neo4j\dbms
NEO4J_JAVA_HOME=D:\path\to\neo4j\java
MARAG_ALLOW_SERVICE_STOP=false
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

## 5A. Claude CLI MCP Without Docker

The project-local MCP server is registered in `.mcp.json` as `marag-mcp`.
Secrets stay in `.env`; `.mcp.json` only starts the local stdio server:

```powershell
uv run python -m multi_agentic_graph_rag.mcp.server
claude mcp list
```

Use these MCP tools from Claude CLI:

- `marag_health_check` checks PostgreSQL service/port, Neo4j ports, Chroma persistence, and `marag db-check`.
- `marag_start_stack` starts only the configured PostgreSQL Windows service and configured Neo4j DBMS path.
- `marag_find_latest_artifacts` returns latest `requirements.json`, `user_stories.json`, and `test_scenarios.json`.
- `marag_run_full_pipeline` runs stack check/start, ingest, user stories, test scenarios, and artifact verification.

Safety defaults: no Docker, no raw SQL/Cypher tools, no database reset, no
process killing, and no service stop unless `MARAG_ALLOW_SERVICE_STOP=true`.

Storage responsibilities:

- Neo4j is the document/chunk graph knowledge base for multi-hop reasoning, and
  also holds validated user-story and test-scenario coverage claim-nodes linked
  back to their evidence chunks once downstream stages have run.
- ChromaDB stores chunk embeddings, chunk text, and chunk/document metadata for
  semantic search.
- PostgreSQL stores full generated `requirements_full.json` payloads, the
  requirement ledger tables, and fallback copies of generated requirement
  artifacts.

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
uv run python -m multi_agentic_graph_rag ingest `
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

Example:

```powershell
uv run python -m multi_agentic_graph_rag ingest --project PROJECT_1 --document .\documents\inbox\PROJECT_1\SIIMCS_BRD_V1.pdf --version 1.0 --json-output
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
consumes the requirement artifact and produces enterprise-grade user stories per
requirement. For each requirement it runs hybrid retrieval — Chroma dense +
Neo4j BM25 full-text + Neo4j multi-hop over the chunk graph — fused by the
cross-encoder reranker, then generates stories under a strict schema.

> One-time reset: this stage adds new PostgreSQL managed tables
> (`user_stories`, `user_story_artifacts`). The schema self-validates, so run a
> reset once against your disposable/local database before the first run:
>
> ```powershell
> uv run python -m multi_agentic_graph_rag postgres-reset --yes
> uv run python -m multi_agentic_graph_rag db-check
> ```

Required stack (enforced): real reasoning + embedding + reranker providers,
`POSTGRES_MODE=postgres`, and `NEO4J_MODE=neo4j`. Retrieval degrades gracefully
per requirement (falls back to the requirement's own evidence chunks, then to
the requirement text), so a run is never blocked by an empty retrieval — but for
grounded stories the document's chunks must already be in Neo4j/Chroma, so
re-ingest the source first if they are missing.

From a local requirement artifact (the file written by ingest):

```powershell
uv run marag generate-user-stories `
  --requirements generated\<PROJECT>\req\<RUN_ID>\requirements.json `
  --project <PROJECT> `
  --json-output
```

From PostgreSQL when the local artifact is absent, select by document version:

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

Output: `user_stories.json` is written **beside the input requirements file**
(or under `generated/<PROJECT>/user_stories/<RUN_ID>/` when loaded from
PostgreSQL). Each story is keyed by a permanent `US-*` id and carries its
`requirement_id`, full content, `doc_version`, and `project`; a `coverage` map
links each requirement to its story ids.

Persistence:

- PostgreSQL: the full artifact in `user_story_artifacts`, plus one index row
  per story in `user_stories`.
- Neo4j: validated coverage claim-nodes —
  `(:UserStory)-[:COVERS_REQUIREMENT]->(:Requirement)`,
  `(:Requirement)-[:EVIDENCED_BY_CHUNK]->(:Chunk)`, and
  `(:DocumentVersion)-[:HAS_USER_STORY]->(:UserStory)` — and each covered
  requirement is marked `covered=true`.

Verify and inspect:

```powershell
uv run marag artifact verify-user-stories generated\<PROJECT>\req\<RUN_ID>\user_stories.json
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select count(*) from user_stories;"
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <password> "MATCH (s:UserStory)-[:COVERS_REQUIREMENT]->(r:Requirement) RETURN count(s), count(r);"
```

## 9. Generate Test Scenarios

Test-scenario generation is a standalone stage (`generate-test-scenarios`) that
consumes `user_stories.json` and produces `test_scenarios.json`. For each user
story it retrieves hybrid context with Chroma dense search, Neo4j BM25
full-text, Neo4j chunk neighbors, and the reranker, then makes one strict
reasoning-LLM call with one fed-back retry on validation failure.

Required stack (enforced): real reasoning + embedding + reranker providers,
`POSTGRES_MODE=postgres`, and `NEO4J_MODE=neo4j`.

Optional tuning:

```powershell
TEST_SCENARIO_TOP_K=4
TEST_SCENARIO_DENSE_K=8
TEST_SCENARIO_SPARSE_K=8
TEST_SCENARIO_NEIGHBOR_WINDOW=1
TEST_SCENARIO_MAX_NEW_TOKENS=
```

From a local user-story artifact:

```powershell
uv run marag generate-test-scenarios `
  --user-stories generated\<PROJECT>\req\<RUN_ID>\user_stories.json `
  --project <PROJECT> `
  --json-output
```

From PostgreSQL when the local artifact is absent:

```powershell
uv run marag generate-test-scenarios `
  --document-version-id <DV-ID> `
  --project <PROJECT> `
  --json-output
```

Requirement evidence loading is lenient unless `--requirements` is supplied.
The stage tries sibling `requirements.json`, sibling `requirements_full.json`,
then PostgreSQL by document version. If none are usable, it warns and continues
with dense+sparse retrieval only. Do not run `postgres-reset` between
user-story generation and test-scenario generation when that fallback is needed.

Persistence:

- PostgreSQL: the full artifact in `test_scenario_artifacts`, plus one index
  row per scenario in `test_scenarios`.
- Neo4j: validated coverage claim-nodes —
  `(:TestScenario)-[:VALIDATES_STORY]->(:UserStory)`,
  `(:TestScenario)-[:COVERS_REQUIREMENT]->(:Requirement)`,
  `(:Requirement)-[:EVIDENCED_BY_CHUNK]->(:Chunk)`, and
  `(:DocumentVersion)-[:HAS_TEST_SCENARIO]->(:TestScenario)`.

Verify and inspect:

```powershell
uv run marag artifact verify-test-scenarios generated\<PROJECT>\req\<RUN_ID>\test_scenarios.json
psql "postgresql://marag:<password>@127.0.0.1:5432/marag" -c "select count(*) from test_scenarios;"
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <password> "MATCH (t:TestScenario)-[:VALIDATES_STORY]->(:UserStory) RETURN count(t);"
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
- `generate-user-stories` writes its session log under
  `.generated/<PROJECT>/run/`; the `user_stories.json` artifact lands beside the
  input requirements file.
- `generate-test-scenarios` writes its session log under
  `.generated/<PROJECT>/run/`; the `test_scenarios.json` artifact lands beside
  the input user-story file.
