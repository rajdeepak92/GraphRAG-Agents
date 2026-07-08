# Multi-Agentic Graph RAG

This repository is ingestion-first. The public package is `multi_agentic_graph_rag`.
On Windows with App Control enabled, use `uv run` rather than generated launchers.

## Setup

```powershell
Copy-Item .env.example .env
& C:\Users\rdmpr\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
uv sync --dev --extra local-llm
```

The default profile uses Hugging Face Qwen for requirement discovery, Hugging
Face BGE-M3 for embeddings, a Hugging Face BGE reranker, PostgreSQL, Neo4j,
and Chroma. `HF_TOKEN=` may be blank for public models while online. Use
`HUGGINGFACE_OFFLINE=true` only after the configured models are cached.

## Baseline Checks

```powershell
uv run python -m multi_agentic_graph_rag version
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag hf-check --load-model
uv run python -m multi_agentic_graph_rag db-check
```

For a lighter model check that does not load weights:

```powershell
uv run python -m multi_agentic_graph_rag hf-check
```

## Ingest

```powershell
uv run python -m multi_agentic_graph_rag ingest `
  --project <PROJECT> `
  --document <DOCUMENT_PATH> `
  --version <VERSION> `
  --json-output
```

Generated requirement artifacts and logs are written under:

```text
generated/<PROJECT>/req/<RUN_ID>/
```

That directory is local-only and gitignored. It contains compact
`requirements.json`, full audit `requirements_full.json`, `run.log`,
`run.jsonl`, `chunk_manifest.json`, and any saved `llm_response_*.txt` files.

When requirement discovery parsing or source-trace validation fails, the full
raw model response is saved beside the run artifacts. Set
`LOG_LLM_RESPONSES=true` to also capture raw successful responses for
debugging.

## Database Requirements

For a real ingest run, all three services must be configured and reachable:

- Neo4j for the document/chunk graph knowledge base used in multi-hop reasoning
  and validated downstream claim-node projections for user stories and test
  scenarios.
- ChromaDB for chunk text embeddings and chunk/document metadata used in
  semantic vector search only.
- PostgreSQL for generated artifacts and ledger rows: requirements, user
  stories, test scenarios, document-version manifests, and run records.

Use a libpq/psycopg-style PostgreSQL URL, for example:

```powershell
POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:<password>@127.0.0.1:5432/marag
```

The runtime profile is controlled by provider-specific settings:

- `REASONING_MODEL_PROVIDER`: `huggingface` or `azure_openai`
- `EMBEDDING_MODEL_PROVIDER`: `huggingface` or `azure_openai`
- `RERANKER_MODEL_PROVIDER`: `huggingface`

Azure OpenAI uses `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_API_VERSION`, and the reasoning and embedding deployment names.
Hugging Face uses `HF_TOKEN`, `HUGGINGFACE_REASONING_MODEL`,
`HUGGINGFACE_EMBEDDING_MODEL`, `HUGGINGFACE_RERANKER_MODEL`,
`HUGGINGFACE_OFFLINE`, `HUGGINGFACE_MAX_NEW_TOKENS`, and
`DISCOVERY_BATCH_SIZE`. `HUGGINGFACE_TOKEN` and `HUGGING_FACE_HUB_TOKEN`
remain accepted as backward-compatible aliases.

Requirement discovery runs one chunk at a time. A per-run coverage ledger
(`DISCOVERY_LEDGER_ENABLED`, default on; sized by `DISCOVERY_LEDGER_MAX_ENTRIES`
and `DISCOVERY_LEDGER_TOP_K`) injects previously discovered requirements into
later chunk prompts so the model converges paraphrased restatements onto the
same `requirement_key`/`req_text` instead of drifting. It never skips a
requirement or its source evidence: the deterministic requirement builder
remains the sole authority for deduplication, evidence accumulation, and
revision tracking. Set `DISCOVERY_LEDGER_ENABLED=false` to restore the previous
stateless behavior.

## Static Checks

```powershell
uv run ruff check .
uv run python -m unittest discover -s tests
uv run python -m compileall -q src
uv run mypy src/multi_agentic_graph_rag
```

## Database Reset

Use this only against a disposable/local PostgreSQL app database:

```powershell
uv run python -m multi_agentic_graph_rag postgres-reset --yes
uv run python -m multi_agentic_graph_rag db-check
```

## Test-Scenario HFIL

HFIL is disabled by default. Enable it only for interactive review of
test-scenario generation:

```powershell
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --user-stories generated\<PROJECT>\req\<RUN_ID>\user_stories.json `
  --project <PROJECT> `
  --hfil
```

Use `--no-hfil` for the non-interactive regression path. `--emit-md` writes an
optional Markdown report, but `test_scenarios.json` remains the canonical
artifact. `--thread-id <ID>` can resume a stable LangGraph HFIL thread.

HFIL supports `remove duplicates`, feedback comments, and `exit`. Review turns
do not write PostgreSQL or JSON; final persistence happens after `exit`.

## Reconcile

Generated artifacts are committed to PostgreSQL first and mirrored to local
JSON after the DB commit. Valid local JSON is preferred on read when it matches
PostgreSQL metadata/payload checks. If local JSON is missing or stale, repair it:

```powershell
uv run python -m multi_agentic_graph_rag reconcile --project <PROJECT>
```

For Neo4j and Chroma resets:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j -p <password> "MATCH (n) DETACH DELETE n;"
Remove-Item -Recurse -Force runtime\databases\chroma -ErrorAction SilentlyContinue
```

## Inspect

```powershell
uv run python -m multi_agentic_graph_rag run status <RUN-ID>
uv run python -m multi_agentic_graph_rag artifact verify generated\<PROJECT>\req\<RUN-ID>\requirements_full.json
Get-Content generated\<PROJECT>\req\<RUN-ID>\run.log -Tail 200
Get-Content generated\<PROJECT>\req\<RUN-ID>\run.jsonl -Tail 200
```

`run status` still has a backward-compatible read path for old
`.generated/<PROJECT>/run/` JSONL files.
