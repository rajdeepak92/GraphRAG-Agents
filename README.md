# Multi-Agentic Graph RAG

This rebuild is ingestion-first. The public package remains `multi_agentic_graph_rag`
and the CLI command remains `marag`.

## Setup

```powershell
uv sync
Copy-Item .env.example .env
uv run marag version
uv run marag config-check
uv run marag doctor
uv run marag db-check
```

The default `config.json` uses local trace files for PostgreSQL and Neo4j plus a
real local Chroma collection so smoke checks work without external services. To
use real services, set `POSTGRES_MODE=postgres` and `NEO4J_MODE=neo4j` in `.env`
with the matching credentials.

## Ingest

```powershell
uv run marag ingest --project PROJECT_1 --document .\documents\inbox\PROJECT_1\sample.txt --version 1.0
```

Generated requirement JSON is written under:

```text
generated/inbox/<PROJECT>/requirements/
```

Runtime manifests, run logs, and local trace stores are written under `runtime/`.

## Providers

Provider choices are separate:

- `REASONING_MODEL_PROVIDER`: `local_heuristic`, `azure_openai`, or `huggingface`
- `EMBEDDING_MODEL_PROVIDER`: `local_hash`, `azure_openai`, or `huggingface`
- `RERANKER_MODEL_PROVIDER`: currently `none`

Azure OpenAI and Hugging Face adapters sit behind the same model ports. OpenAI
direct and Gemini are future adapters.

## Future User Stories

User-story generation is intentionally not implemented yet. The future contract
is: read requirement JSON when present, regenerate from PostgreSQL if missing,
retrieve semantic context from Chroma, expand graph context from Neo4j, rerank
candidates, then call the reasoning model.
