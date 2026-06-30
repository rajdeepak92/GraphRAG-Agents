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

Storage responsibilities:

- Neo4j is the document/chunk graph knowledge base for multi-hop reasoning.
- ChromaDB stores chunk embeddings, chunk text, and chunk/document metadata for
  semantic search.
- PostgreSQL stores generated `requirements.json` payloads, the requirement
  ledger tables, and fallback copies of generated requirement artifacts.

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

That folder contains `requirements.json`, `run.log`, `run.jsonl`,
`chunk_manifest.json`, and any saved `llm_response_*.txt` files.

## 8. Observability

Inspect the latest run:

```powershell
uv run python -m multi_agentic_graph_rag run status <RUN-ID>
uv run python -m multi_agentic_graph_rag artifact verify generated\<PROJECT>\req\<RUN-ID>\requirements.json
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

## 9. Notes

- `run status` still supports the legacy `.generated/<PROJECT>/run/` path.
- The repo keeps generated outputs and runtime data out of version control.
