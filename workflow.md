# End-to-End Workflow: Setup, Configuration & Pipeline Run

This guide walks through configuring the platform for **two supported provider
combinations** and running the full pipeline from raw requirements document to
generated test code:

```
Stage 1.1  ─►  Stage 1.2  ─►  Stage 2         ─►  Stage 3              ─►  Stage 4
(ingest &      (requirement    (user stories)      (test scenarios)         (test code)
 chunk &        discovery)
 embed)
```

Two model roles are cloud-provider-selectable; the third is fixed:

| Role        | Config 1        | Config 2        | Notes |
|-------------|-----------------|-----------------|-------|
| Reasoning   | **Gemini**      | **Azure OpenAI**| structured generation for discovery / stories / scenarios / test code |
| Embedding   | **Gemini**      | **Azure OpenAI**| vectors for hybrid retrieval + Chroma |
| Reranker    | **HuggingFace** | **HuggingFace** | always HuggingFace — no cloud reranker exists |

> Provider selection is **explicit and has no fallback**. If a provider is
> misconfigured the run fails fast rather than silently switching providers.

---

## 0. Prerequisites (both configs)

### 0.1 Backing services must be reachable
The real pipeline requires **PostgreSQL**, **Neo4j**, and **ChromaDB**. Start
them (Docker or local) and confirm the connection settings in `.env`
(`POSTGRES_DSN`, `NEO4J_URI`, `NEO4J_PASSWORD`, `CHROMA_PERSIST_DIR`).

### 0.2 Python environment (uv)
```powershell
# Base + reranker + Stage-4 extras are always needed.
# Add the cloud extra for the config you are running (gemini OR azure).

# --- Config 1 (Gemini): ---
uv sync --dev --extra gemini --extra local-llm --extra stage4

# --- Config 2 (Azure): ---
uv sync --dev --extra azure --extra local-llm --extra stage4
```
- `gemini` → `google-genai`
- `azure` → `openai`, `tiktoken`, `azure-identity`
- `local-llm` → `sentence-transformers`, `torch` (the **HuggingFace reranker**)
- `stage4` → `robotframework`, `openpyxl` (Stage-4 test generation)

### 0.3 HuggingFace reranker (shared by both configs)
The reranker downloads `BAAI/bge-reranker-base` on first use into the
HuggingFace cache. These vars apply to **both** configs:
```powershell
$env:RERANKER_MODEL_PROVIDER = "huggingface"
$env:HUGGINGFACE_RERANKER_MODEL = "BAAI/bge-reranker-base"
$env:HUGGINGFACE_DEVICE = "auto"     # auto | cpu | cuda
# $env:HF_TOKEN = "hf_..."           # only if the reranker model is gated/private
```
> The model cache was recently cleared, so the **first** stage that reranks
> (Stage 2/3/4 retrieval) re-downloads `bge-reranker-base` (~1.1 GB). This is a
> one-time cost.

---

## 1. Configuration 1 — Gemini reasoning + Gemini embedding + HF reranker

### 1.1 `.env` (copy from `.env.example`, then set)
```powershell
Copy-Item .env.example .env
```
Set these keys in `.env` (or export as environment variables):
```ini
# --- Provider selection ---
REASONING_MODEL_PROVIDER=gemini
EMBEDDING_MODEL_PROVIDER=gemini
RERANKER_MODEL_PROVIDER=huggingface

# --- Gemini (Developer API, API-key auth) ---
GEMINI_API_KEY=your-gemini-api-key
GEMINI_REASONING_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# --- HuggingFace reranker ---
HUGGINGFACE_RERANKER_MODEL=BAAI/bge-reranker-base
HUGGINGFACE_DEVICE=auto

# --- Stage 4 reasoning provider ---
STAGE4_REASONING_PROVIDER=gemini
```

### 1.2 Verify configuration before running
```powershell
uv run python -m multi_agentic_graph_rag config-check   # echoes selected providers
uv run python -m multi_agentic_graph_rag doctor         # Stage-4 dependency checks
uv run python -m multi_agentic_graph_rag db-check        # Postgres / Neo4j / Chroma reachable
uv run python -m multi_agentic_graph_rag hf-check --load-model   # loads the HF reranker
```
`config-check` should report `reasoning_provider: gemini`,
`embedding_provider: gemini`, `reranker_provider: huggingface`.

### 1.3 Run the pipeline (Stage 1.1 → 1.2 → 2 → 3)
```powershell
# Stage 1.1 (ingest + chunk + embed) AND Stage 1.2 (requirement discovery)
# run together in one command. It prints the run_id used by every later stage.
uv run python -m multi_agentic_graph_rag ingest `
  --project customer-portal `
  --document .\documents\requirements.docx
# -> note the "run_id" in the JSON output, e.g. RUN-abc123

# Stage 2 — user stories
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --project customer-portal `
  --run-id RUN-abc123

# Stage 3 — test scenarios
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --project customer-portal `
  --run-id RUN-abc123

# (optional) traceability / coverage report across 1.2 -> 2 -> 3
uv run python -m multi_agentic_graph_rag coverage `
  --project customer-portal `
  --run-id RUN-abc123
```
> Providers come from `.env`. To override per-command, add
> `--reasoning-provider gemini` / `--embedding-provider gemini`.

### 1.4 Stage 4 — test scenarios → test code
Stage 4 turns Stage-3 scenarios into runnable test code against a real test
framework. Three steps:
```powershell
# (a) Index the target automation framework (AST graph, no LLM cost)
uv run python -m multi_agentic_graph_rag index-framework `
  --framework-path C:\frameworks\customer-portal

# (b) Ingest the approved test-data document (.xlsx or normalized .json).
#     Formulas in a data region are a hard error (no cached-value fallback).
uv run python -m multi_agentic_graph_rag ingest-test-data `
  --project customer-portal `
  --document C:\frameworks\customer-portal\test-data\test-data.xlsx

# (c) DRY RUN first — deterministic readiness checks, no model calls, no writes.
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal `
  --run-id RUN-abc123 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider gemini `
  --dry-run

# (d) Real run — generates frozen Stage-4 test cases with Gemini.
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal `
  --run-id RUN-abc123 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider gemini
```
Stage 4 does not execute the generated domain test; it validates path/additive-AST
policy, Python parse/compile, isolated import, lifecycle contracts, Robot
Framework dry-run, traceability, exact-data checks, and file hashes.

---

## 2. Configuration 2 — Azure OpenAI reasoning + Azure embedding + HF reranker

### 2.1 `.env`
```powershell
Copy-Item .env.example .env   # skip if already present
```
Set:
```ini
# --- Provider selection ---
REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface

# --- Azure OpenAI (API-key auth) ---
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=your-reasoning-deployment
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your-embedding-deployment

# --- HuggingFace reranker ---
HUGGINGFACE_RERANKER_MODEL=BAAI/bge-reranker-base
HUGGINGFACE_DEVICE=auto

# --- Stage 4 reasoning provider ---
STAGE4_REASONING_PROVIDER=azure_openai
```
> The Azure API version must support strict structured outputs
> (`2024-08-01-preview` or newer, or GA `v1`); the reasoning adapter fails fast
> otherwise.

### 2.2 Verify
```powershell
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag doctor
uv run python -m multi_agentic_graph_rag db-check
uv run python -m multi_agentic_graph_rag hf-check --load-model
```
`config-check` should report `reasoning_provider: azure_openai`,
`embedding_provider: azure_openai`, `reranker_provider: huggingface`.

### 2.3 Run the pipeline (Stage 1.1 → 1.2 → 2 → 3)
```powershell
# Stage 1.1 + Stage 1.2
uv run python -m multi_agentic_graph_rag ingest `
  --project customer-portal `
  --document .\documents\requirements.docx
# -> note the run_id, e.g. RUN-abc123

# Stage 2
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --project customer-portal --run-id RUN-abc123

# Stage 3
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --project customer-portal --run-id RUN-abc123

# (optional) coverage
uv run python -m multi_agentic_graph_rag coverage `
  --project customer-portal --run-id RUN-abc123
```
> To override per-command: `--reasoning-provider azure_openai`
> `--embedding-provider azure_openai`.

### 2.4 Stage 4 — test scenarios → test code
```powershell
# (a) Index framework
uv run python -m multi_agentic_graph_rag index-framework `
  --framework-path C:\frameworks\customer-portal

# (b) Ingest test data
uv run python -m multi_agentic_graph_rag ingest-test-data `
  --project customer-portal `
  --document C:\frameworks\customer-portal\test-data\test-data.xlsx

# (c) Dry run (no model calls / no writes)
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal `
  --run-id RUN-abc123 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider azure_openai `
  --dry-run

# (d) Real run
uv run python -m multi_agentic_graph_rag generate-test-code `
  --project customer-portal `
  --run-id RUN-abc123 `
  --framework-path C:\frameworks\customer-portal `
  --execution-profile EP-DEFAULT `
  --test-data C:\frameworks\customer-portal\test-data\test-data.xlsx `
  --reasoning-provider azure_openai
```

---

## 3. Stage reference

| Stage | Command | What it does | Provider(s) used |
|-------|---------|--------------|------------------|
| 1.1 | `ingest` | Parse document, chunk, embed, persist manifest + vectors | embedding |
| 1.2 | `ingest` (same call) | Requirement discovery + relationship extraction + semantic projection | reasoning, embedding |
| 2   | `generate-user-stories` | Generate user stories from discovered requirements | reasoning, embedding, **reranker** |
| 3   | `generate-test-scenarios` | Generate test scenarios from user stories | reasoning, embedding, **reranker** |
| —   | `coverage` | Traceability report across 1.2 → 2 → 3 | (read-only) |
| 4   | `index-framework` → `ingest-test-data` → `generate-test-code` | Convert scenarios into framework test code | reasoning, embedding, **reranker** |

The **HuggingFace reranker** is invoked inside hybrid retrieval during Stages 2,
3, and 4 (it re-orders retrieved context), regardless of which cloud provider
serves reasoning/embedding.

---

## 4. Provider-override cheat-sheet

Precedence: **per-command flag** > **environment variable / `.env`** > default.

| What | Env var | Per-command flag |
|------|---------|------------------|
| Reasoning provider | `REASONING_MODEL_PROVIDER` | `--reasoning-provider {azure_openai\|gemini}` |
| Embedding provider | `EMBEDDING_MODEL_PROVIDER` | `--embedding-provider {azure_openai\|gemini}` |
| Reranker provider  | `RERANKER_MODEL_PROVIDER` (always `huggingface`) | — |
| Stage-4 reasoning  | `STAGE4_REASONING_PROVIDER` | `--reasoning-provider` on `generate-test-code` |

> `--reasoning-provider huggingface` is **rejected** — HuggingFace is a reranker
> only; reasoning/embedding accept `azure_openai` or `gemini`.

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `REASONING_MODEL_PROVIDER=gemini requires GEMINI_API_KEY` | set `GEMINI_API_KEY` |
| `... requires google-genai; install with: uv sync --dev --extra gemini` | install the `gemini` extra |
| `EMBEDDING_MODEL_PROVIDER=azure_openai requires embedding deployment` | set `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` |
| `Azure strict Structured Outputs readiness failed` | use API version `2024-08-01-preview`+ / GA `v1` and a supported deployment |
| `RERANKER_MODEL_PROVIDER=huggingface requires sentence-transformers` | install the `local-llm` extra |
| `HUGGINGFACE_DEVICE=cuda ... CUDA is unavailable` | install CUDA-enabled PyTorch or set `HUGGINGFACE_DEVICE=cpu` |
| Reranker slow / re-downloading on first run | expected once — `bge-reranker-base` populates the HF cache |
| Unsupported reasoning/embedding provider error | provider must be `azure_openai` or `gemini` (not `huggingface`) |
