# Azure OpenAI and private Hugging Face workflow

The workflow supports Azure OpenAI or private Hugging Face adapters for
reasoning and embeddings, with a Hugging Face reranker.

## Provider configuration

Copy `.env.example` to `.env`, then select providers:

```dotenv
REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface
```

For Azure OpenAI:

```dotenv
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_KEY=<secret>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=<reasoning-deployment>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment>
```

For private Hugging Face models:

```dotenv
HF_TOKEN=<secret>
HUGGINGFACE_REASONING_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
HUGGINGFACE_EMBEDDING_MODEL=BAAI/bge-m3
HUGGINGFACE_RERANKER_MODEL=BAAI/bge-reranker-base
HUGGINGFACE_OFFLINE=false
```

Do not commit `.env`.

## Validate the stack

```powershell
uv sync --extra azure --extra local-llm
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag hf-check
uv run python -m multi_agentic_graph_rag doctor
uv run python -m multi_agentic_graph_rag db-check
```

`hf-check --load-model` performs the heavier local model initialization check.

## Execute the workflow

```powershell
$Project = "customer-portal"
$Source = ".\documents\requirements.docx"

$Ingest = uv run python -m multi_agentic_graph_rag ingest `
  --project $Project `
  --document $Source
```

The JSON output contains `run_id`, `chunk_manifest`, and `requirements`.
Use the returned run ID:

```powershell
$RunId = "<RUN-ID>"

uv run python -m multi_agentic_graph_rag generate-user-stories `
  --project $Project `
  --run-id $RunId

uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --project $Project `
  --run-id $RunId

uv run python -m multi_agentic_graph_rag coverage `
  --project $Project `
  --run-id $RunId
```

Stage 1.1 uses only the configured embedding adapter. Stage 1.2 uses only the
configured reasoning adapter. Stages 2 and 3 use reasoning, query embeddings,
the reranker, Neo4j, ChromaDB, and PostgreSQL.

## Operational rules

- Stage 1.2 receives exactly one manifest chunk per primary call.
- A successful but invalid Stage 1.2 response is terminal; it is not repaired by
  another model call.
- Stages 2 and 3 may make one targeted output-repair call without repeating
  retrieval.
- All retrieval is project-scoped and intersected with the selected run’s
  manifest chunk IDs.
- Readiness must be `ready` and its `build_run_id` must match the requested run.
- Empty grounded result arrays are valid; unsupported content is rejected.
