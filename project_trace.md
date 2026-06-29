# Project Trace

## CLI

- `marag version`: `src/multi_agentic_graph_rag/cli.py`
- `marag config-check`: `src/multi_agentic_graph_rag/cli.py`
- `marag doctor`: `src/multi_agentic_graph_rag/cli.py`
- `marag db-check`: `src/multi_agentic_graph_rag/cli.py`
- `marag ingest`: `src/multi_agentic_graph_rag/cli.py`
- `marag run status`: `src/multi_agentic_graph_rag/cli.py`
- `marag run resume`: `src/multi_agentic_graph_rag/cli.py`
- `marag artifact verify`: `src/multi_agentic_graph_rag/cli.py`

## Ingestion Flow

- Agent contract: `src/multi_agentic_graph_rag/agents/ingestion_document_agent.py`
- LangGraph workflow: `src/multi_agentic_graph_rag/workflows/ingestion_graph.py`
- Runtime logging: `src/multi_agentic_graph_rag/observability/logging.py`

## IDs And Schemas

- Chunk ID generation: `src/multi_agentic_graph_rag/domain/identifiers.py`
- Fact ID generation: `src/multi_agentic_graph_rag/domain/identifiers.py`
- Requirement ID generation: `src/multi_agentic_graph_rag/domain/identifiers.py`
- Strict schemas: `src/multi_agentic_graph_rag/domain/schemas.py`

## Source Processing

- Parsing: `src/multi_agentic_graph_rag/services/parsing.py`
- Chunking: `src/multi_agentic_graph_rag/services/chunking.py`
- Manifest building and writing: `src/multi_agentic_graph_rag/services/manifest.py`

## Requirement Discovery

- Bounded agent: `src/multi_agentic_graph_rag/agents/requirement_discovery_agent.py`
- Source trace verification: `src/multi_agentic_graph_rag/agents/requirement_discovery_agent.py`
- Provider ports: `src/multi_agentic_graph_rag/llm_models/ports.py`
- Azure OpenAI adapter: `src/multi_agentic_graph_rag/llm_models/azure_openai.py`
- Hugging Face adapter: `src/multi_agentic_graph_rag/llm_models/huggingface.py`

## Stores

- PostgreSQL persistence: `src/multi_agentic_graph_rag/db/postgres.py`
- Neo4j projection: `src/multi_agentic_graph_rag/db/neo4j_store.py`
- Chroma indexing: `src/multi_agentic_graph_rag/db/chroma_store.py`
- Artifact writing and verification: `src/multi_agentic_graph_rag/services/artifacts.py`

## Generated Outputs

- Requirement JSON path: `generated/inbox/<PROJECT>/requirements/requirements_<timestamp>_<version>.json`
- Runtime manifests: `runtime/staging/<RUN-ID>/chunk_manifest.json`
- Runtime logs: `runtime/logs/<RUN-ID>.jsonl`
