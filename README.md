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
.generated/<PROJECT>/run/
```

Runtime manifests stay under `runtime/staging/`, while run logs and run
metadata are written under `.generated/<PROJECT>/run/`.

## Requirement Ledger

Ingestion now extracts a nested `chunks[] -> facts[] -> requirements[]` structure
from the reasoning model. Python-owned artifact generation converts that into:

- `FACT-*` fact occurrence IDs tied to one chunk quote and source span.
- Canonical fact groups for repeated fact text across chunks.
- Stable `REQ-*` requirement lineage IDs derived from a hybrid requirement key.
- `REQREV-*` revision IDs for exact requirement statement versions.
- Evidence rows and delta events for `new`, `duplicate`, `changed`, and
  `superseded` lifecycle outcomes.

Storage ownership is intentionally split:

- Neo4j owns chunk nodes and graph relationships for GraphRAG traversal.
- ChromaDB owns semantic vector search over chunks by `chunk_id`.
- PostgreSQL owns generated requirement ledger state, revisions, evidence
  references, lifecycle status, delta events, and the JSON requirement artifact
  snapshot. PostgreSQL stores chunk references, quotes, offsets, pages, sections,
  and source paths; it is not the primary chunk-body store.

## Providers

Provider choices are separate:

- `REASONING_MODEL_PROVIDER`: `local_heuristic`, `azure_openai`, or `huggingface`
- `EMBEDDING_MODEL_PROVIDER`: `local_hash`, `azure_openai`, or `huggingface`
- `RERANKER_MODEL_PROVIDER`: currently `none`

Azure OpenAI and Hugging Face adapters sit behind the same model ports. OpenAI
direct and Gemini are future adapters.

## Future User Stories

User-story generation is intentionally not implemented yet. The future contract
is: read authoritative requirement lifecycle state from PostgreSQL, retrieve
semantic context from Chroma, expand graph context from Neo4j, rerank candidates,
then call the reasoning model.
