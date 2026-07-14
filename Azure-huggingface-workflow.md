# Azure Reasoning + Azure Embeddings + Hugging Face Reranker Workflow

This guide defines one clean, supported runtime profile:

- Reasoning model: Azure OpenAI (`azure_openai`)
- Embedding model: Azure OpenAI (`azure_openai`)
- Reranker: Hugging Face (`huggingface`) authenticated with `HF_TOKEN`
- Generated-artifact source of truth: PostgreSQL
- Source-knowledge and traceability graph: Neo4j
- Dense chunk index: local persistent ChromaDB

Run every PowerShell command from the repository root. Replace example deployment names,
passwords, project names, document paths, and document-version IDs before running commands.
Never commit `.env` or paste credentials into commands, logs, screenshots, or issue reports.

## Prerequisites

Install or provision the following before starting:

1. PowerShell 7 or Windows PowerShell.
2. `uv` on `PATH`.
3. Python 3.12, or permission for `uv` to install it.
4. An Azure OpenAI resource with:
   - one chat/reasoning deployment;
   - one embeddings deployment;
   - an endpoint, API key, and API version supported by both deployments.
5. A Hugging Face account/token that can download the configured reranker model.
6. A running PostgreSQL server.
7. A running Neo4j DBMS with Bolt enabled.

The direct CLI checks database connectivity and does not start PostgreSQL or Neo4j. Start both
services before `db-check` or `ingest`.

## 1. Synchronize Azure and Hugging Face Dependencies

The `azure` extra installs the Azure OpenAI client. The `local-llm` extra installs
`sentence-transformers`, Transformers, PyTorch, and Hugging Face Hub support required by the
local reranker.

```powershell
uv sync --python 3.12 --dev --extra azure --extra local-llm
```

`uv sync` creates or updates the repository-local `.venv` and locks the environment to
`uv.lock`. Do not install model libraries separately with `pip` unless intentionally debugging
dependency resolution.

## 2. Activate and Verify the Virtual Environment

Activation is optional when using `uv run`, but it is useful for interactive checks.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\.venv\Scripts\Activate.ps1
python --version
uv run python -m multi_agentic_graph_rag version
```

Expected Python version: `3.12.x`.

To leave the environment later:

```powershell
deactivate
```

## 3. Configure `.env`

Create the local file, then edit it. The application loads repository-root `.env` values and
lets process environment variables override them.

```powershell
Copy-Item .env.example .env
notepad .env
```

Use this profile. Replace every `YOUR_...` value. `HF_TOKEN` is the only Hugging Face token
variable required by this workflow.

```dotenv
APP_ENV=development
LOG_LEVEL=INFO

# Strict provider selection for this workflow.
REASONING_MODEL_PROVIDER=azure_openai
EMBEDDING_MODEL_PROVIDER=azure_openai
RERANKER_MODEL_PROVIDER=huggingface

# Azure OpenAI. Values are Azure deployment names, not base model names.
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE_NAME.openai.azure.com/
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_REASONING_DEPLOYMENT=YOUR_REASONING_DEPLOYMENT
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=YOUR_EMBEDDING_DEPLOYMENT

# Hugging Face reranker only.
HF_TOKEN=YOUR_HUGGING_FACE_TOKEN
HUGGINGFACE_RERANKER_MODEL=BAAI/bge-reranker-base
HUGGINGFACE_OFFLINE=false

# Never enable raw model-response diagnostics for routine operation.
LOG_LLM_RESPONSES=false

# PostgreSQL is the canonical generated-artifact ledger.
POSTGRES_MODE=postgres
POSTGRES_DSN=postgresql://marag:YOUR_URL_ENCODED_PASSWORD@127.0.0.1:5432/marag

# Neo4j stores source knowledge and derivative traceability projections.
NEO4J_MODE=neo4j
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=YOUR_NEO4J_PASSWORD
NEO4J_DATABASE=neo4j

# Ingest builds semantic knowledge; downstream generation must use it.
KNOWLEDGE_GRAPH_ENABLED=true
KNOWLEDGE_GRAPH_SHADOW_MODE=true
GRAPH_PRIMARY_STORY=true
GRAPH_PRIMARY_SCENARIO=true
KNOWLEDGE_GRAPH_EXPANSION_K=6
KNOWLEDGE_GRAPH_MIN_ASSERTIONS=3

# Requirement recall and structured classification remain strictly bounded.
REQUIREMENT_USE_RERANKER=true
REQUIREMENT_CANDIDATE_TOP_K=2
REQUIREMENT_MAX_ENTAILMENT_CALLS=200
REQUIREMENT_MAX_STRUCTURED_ATTEMPTS=2
REQUIREMENT_RECALL_COSINE_THRESHOLD=0.62

# Optional retrieval bounds.
USER_STORY_TOP_K=4
USER_STORY_DENSE_K=8
USER_STORY_SPARSE_K=8
USER_STORY_NEIGHBOR_WINDOW=1
TEST_SCENARIO_TOP_K=4
TEST_SCENARIO_DENSE_K=8
TEST_SCENARIO_SPARSE_K=8
TEST_SCENARIO_NEIGHBOR_WINDOW=1

# Keep optional test-scenario review disabled for unattended runs.
ENABLE_HFIL=false
```

If a password contains `@`, `:`, `/`, `?`, `#`, or `%`, URL-encode it in `POSTGRES_DSN`.
Do not place quotes around dotenv values unless the quotes are intended to be part of the value.

Validate provider selection without printing secrets:

```powershell
uv run python -m multi_agentic_graph_rag config-check
uv run python -m multi_agentic_graph_rag doctor
```

Confirm that the safe output identifies Azure for reasoning and embeddings and Hugging Face for
reranking.

## 4. Configure, Validate, and Clean the Databases

### 4.1 Create the PostgreSQL application database

Start PostgreSQL, then open an administrator session:

```powershell
psql -h 127.0.0.1 -U postgres
```

Run the following SQL inside `psql`, replacing the password with a unique local-development
secret:

```sql
CREATE ROLE marag WITH LOGIN PASSWORD 'YOUR_POSTGRES_PASSWORD';
CREATE DATABASE marag OWNER marag;
\q
```

If the role or database already exists, do not recreate it. Instead, confirm that the role can
connect to the database and that `POSTGRES_DSN` contains the corresponding URL-encoded password.

The application creates and validates its managed tables when `db-check` runs. Do not create the
application schema manually.

### 4.2 Create the Neo4j database

1. Open Neo4j Desktop or your local Neo4j service manager.
2. Create a local DBMS and set a non-empty password for the `neo4j` user.
3. Start the DBMS.
4. Confirm that Bolt listens on `127.0.0.1:7687`.
5. Put the same URI, username, password, and database name in `.env`.

Neo4j stores `Project`, `Document`, `DocumentVersion`, `Chunk`, `TextUnit`, `Entity`,
`Assertion`, and evidence relationships. Canonical requirements, user stories, and test
scenarios remain PostgreSQL-owned; Neo4j holds only their derivative traceability projections.

### 4.3 Initialize ChromaDB and validate all stores

ChromaDB is embedded and needs no separate service. This command creates its local collection,
validates Neo4j connectivity, and creates/validates the PostgreSQL schema:

```powershell
uv run python -m multi_agentic_graph_rag db-check
```

Do not continue until PostgreSQL, Neo4j, and Chroma all report `PASS`.

### 4.4 Clean up a disposable local environment

These commands are destructive. Use them only for the local `marag` database and local Neo4j
DBMS configured above. Never run them against shared, staging, or production databases.

Reset only the application-managed PostgreSQL schema and recreate it:

```powershell
uv run python -m multi_agentic_graph_rag postgres-reset --yes
uv run python -m multi_agentic_graph_rag db-check
```

Delete all nodes and relationships from the disposable Neo4j database. `cypher-shell` prompts
for the Neo4j password so it is not placed on the command line:

```powershell
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j "MATCH (n) DETACH DELETE n;"
```

Delete the local Chroma index only after all MARAG processes have stopped:

```powershell
$ChromaPath = (Resolve-Path .).Path + "\runtime\databases\chroma"
if (Test-Path -LiteralPath $ChromaPath) {
    Remove-Item -LiteralPath $ChromaPath -Recurse -Force
}
uv run python -m multi_agentic_graph_rag db-check
```

The cleanup above does not remove `generated\` artifacts or run logs. Remove those separately
only when their retention requirements permit it.

## 5. Check the Azure Reasoning Deployment

This check loads `.env`, constructs the configured provider through the production factory, sends
a minimal native strict-JSON-Schema request, and validates the response again with Pydantic. It
prints only safe status metadata.

`generate_structured` intentionally has no prompt-only compatibility form. Every call must supply
an operation-specific `system_message` plus safe `operation` and `request_id` diagnostic anchors.
Calling it with only `prompt` and `schema` fails locally with `TypeError` before Azure is contacted;
that indicates a stale caller, not an Azure outage.

```powershell
@'
from typing import Literal

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.domain.schemas import StrictModel
from multi_agentic_graph_rag.llm_models.factory import create_reasoning_model


class AzureReasoningProbe(StrictModel):
    status: Literal["ok"]


settings = load_config()
assert settings.reasoning_model.provider == "azure_openai"
model = create_reasoning_model(settings)
result = model.generate_structured(
    prompt='{"requested_status":"ok"}',
    schema=AzureReasoningProbe,
    system_message=(
        "You are an Azure Structured Outputs readiness probe. "
        "Return only the fields required by the supplied response schema."
    ),
    operation="azure_readiness_probe",
    request_id="manual-probe-001",
    max_attempts=2,
)
assert result.status == "ok"
print(f"PASS reasoning provider={model.provider_name} structured_output=true")
'@ | uv run python -
```

The probe arguments exercise the same production contract used by the agents:

- `system_message` scopes the higher-priority instruction to this operation, so the request cannot
  inherit requirement-discovery or another agent's schema instructions.
- `operation` labels logs and response diagnostics without exposing prompt or source text.
- `request_id` separates this request from other structured calls and prevents diagnostic filename
  collisions.
- `max_attempts=2` bounds Azure SDK transport attempts. Azure schema violations are not repaired by
  a prompt-only fallback; native strict Structured Outputs plus local Pydantic validation remain
  mandatory.

If this fails, verify the Azure endpoint, API version, reasoning deployment name and model version
support Structured Outputs, deployment availability, and API-key authorization. Do not print the
key or full exception payload.

Ingestion also runs the adapter readiness check before reading the source document. Azure API
versions older than `2024-08-01-preview` fail configuration immediately. A deployment that rejects
`response_format=json_schema` with `strict=true` also fails with a capability-specific
configuration error; it is never downgraded to prompt-only JSON parsing. Refusals and content
filtering are terminal for that structured request. Diagnostic filenames contain the safe
operation, request ID, unique adapter call number, and structured attempt number so separate calls
cannot overwrite one another.

## 6. Check the Azure Embedding Deployment

This sends one non-sensitive probe string and prints only the vector count and dimension. It does
not print the embedding.

```powershell
@'
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.llm_models.factory import create_embedding_model

settings = load_config()
assert settings.embedding_model.provider == "azure_openai"
model = create_embedding_model(settings)
vectors = model.embed_documents(["MARAG embedding connectivity probe"])
assert len(vectors) == 1
assert len(vectors[0]) > 0
print(
    f"PASS embedding provider={model.provider_name} "
    f"vectors={len(vectors)} dimensions={len(vectors[0])}"
)
'@ | uv run python -
```

Changing the Azure embedding deployment changes the embedding space. For a clean switch, clear
and rebuild Chroma before ingesting documents with the new deployment.

## 7. Check the Hugging Face Reranker

The first run downloads the configured reranker into the repository cache. `HF_TOKEN` must be
present in `.env` and authorized for gated models.

```powershell
@'
from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.llm_models.factory import create_reranker_model

settings = load_config()
assert settings.reranker_model.provider == "huggingface"
assert settings.huggingface.token
model = create_reranker_model(settings)
documents = [
    "The database connectivity check validates PostgreSQL and Neo4j.",
    "The weather forecast is unrelated to database connectivity.",
]
order = model.rerank("database connectivity validation", documents)
assert sorted(order) == list(range(len(documents)))
print(
    f"PASS reranker provider={model.provider_name} "
    f"documents={len(documents)} top_index={order[0]}"
)
'@ | uv run python -
```

After the model is cached, `HUGGINGFACE_OFFLINE=true` may be used for an offline run. Test offline
mode explicitly before depending on it.

## 8. Ingest a Document and Build Source Knowledge

Place the authorized source document under `documents\inbox`. Ingestion performs parsing,
chunking, Azure embedding, Chroma indexing, requirement discovery with Azure reasoning,
PostgreSQL-first artifact persistence, the Neo4j document/chunk projection, and a best-effort
semantic knowledge build containing entities, assertions, evidence, and text units.

```powershell
$Project = "YOUR_PROJECT"
$Document = "documents\inbox\YOUR_DOCUMENT.pdf"
$Version = "1.0"

uv run python -m multi_agentic_graph_rag ingest `
  --project $Project `
  --document $Document `
  --version $Version `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai
```

Record the emitted `document_version_id` (`DV-...`) and requirements artifact path. Requirement
discovery is part of this command; there is no second requirement-discovery command.

Ingestion commits requirements even if the best-effort semantic knowledge build encounters a
recoverable failure. Before graph-primary generation, confirm that the semantic build completed.
If needed, rebuild it explicitly:

```powershell
$DocumentVersionId = "DV-REPLACE_WITH_INGEST_OUTPUT"

uv run python -m multi_agentic_graph_rag build-knowledge-graph `
  --project $Project `
  --document-version-id $DocumentVersionId `
  --reasoning-provider azure_openai
```

Verify the version-scoped source projection. The command prompts for the Neo4j password:

```powershell
$Cypher = @"
MATCH (v:DocumentVersion {document_version_id: '$DocumentVersionId'})
OPTIONAL MATCH (v)-[:HAS_CHUNK]->(c:Chunk)
OPTIONAL MATCH (v)-[:HAS_ASSERTION]->(a:Assertion)
OPTIONAL MATCH (d:Document)-[:ACTIVE_KNOWLEDGE_VERSION]->(active:DocumentVersion {
  document_version_id: '$DocumentVersionId'
})
RETURN v.document_version_id AS document_version_id,
       count(DISTINCT c) AS chunks,
       count(DISTINCT a) AS assertions,
       count(DISTINCT active) AS active_versions;
"@
cypher-shell -a bolt://127.0.0.1:7687 -u neo4j $Cypher
```

Expected checks:

- exactly one matching document version;
- `chunks` is greater than zero;
- `assertions` meets the configured semantic-knowledge readiness threshold;
- `active_versions` equals `1`.

Counts alone do not prove semantic correctness. Review a bounded sample of `Entity`, `Assertion`,
and `AssertionEvidence` nodes in Neo4j Browser against the source document before treating a new
document class or prompt configuration as production-ready.

### 8.1 Troubleshoot knowledge-extraction grounding failures

Azure must return the native structured root fields `entities` and `assertions`. Python then
enforces the source-grounding contract before anything is projected into Neo4j:

- every declared entity name must occur in the normalized chunk text;
- every assertion subject must resolve to an entity declared in that same response;
- a declared `object_name` resolves to that same entity set;
- every assertion quote must map back to an exact contiguous source span.

Exact normalized entity-name matches always win. As a narrow compatibility rule, an assertion
reference with one extra leading `the` may resolve to the otherwise exact declared entity name.
For example, `The Smart Industrial IoT Monitoring & Control System` may reference a declared
`Smart Industrial IoT Monitoring & Control System`; the declared name remains canonical. This
does not alter global entity normalization, entity IDs, aliases, mention spans, or source quotes.
The matcher does not remove `a` or `an`, ignore punctuation, accept paraphrases, or retain an
unknown subject.

A `TraceValidationError` causes one corrected structured call with the validation feedback. If
the second response still fails, the command exits without activating a partial knowledge graph.
When restricted raw-response capture is enabled for an authorized diagnostic run, the two files
use safe operation/request anchors similar to:

```text
llm_response_knowledge_extraction.chunk_CHUNK-..._call-000001_attempt-1.txt
llm_response_knowledge_extraction.chunk_CHUNK-..._call-000002_attempt-1.txt
```

Keep these files private because they contain source-derived model output. Confirm that their root
fields are `entities` and `assertions`, then compare the failing assertion's subject,
`object_name`, and quote with both `entities[].name` and the matching chunk. Do not weaken quote
grounding to make a paraphrase pass.

After correcting configuration or updating the application, rerun the same rebuild command. The
Neo4j projection uses idempotent merges, and the guarded build replaces the failed readiness state;
no `--replace-version` flag or manual deletion is required:

```powershell
uv run python -m multi_agentic_graph_rag build-knowledge-graph `
  --project $Project `
  --document-version-id $DocumentVersionId `
  --reasoning-provider azure_openai
```

## 9. Generate User Stories with Graph-Primary Retrieval

With `GRAPH_PRIMARY_STORY=true`, the command fails closed if the selected document version has no
ready semantic knowledge graph. It retrieves authoritative assertions and bounded related
assertions from Neo4j, combines them with Azure embedding retrieval and Hugging Face reranking,
then invokes Azure reasoning. PostgreSQL remains the canonical story ledger.

```powershell
uv run python -m multi_agentic_graph_rag generate-user-stories `
  --project $Project `
  --document-version-id $DocumentVersionId `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --reranker-provider huggingface `
  --top-k 4
```

Record the emitted `user_stories.json` path. Do not disable `GRAPH_PRIMARY_STORY` to bypass a
knowledge-build failure; repair or rebuild the selected document version instead.

## 10. Generate Test Scenarios with Graph-Primary Retrieval

With `GRAPH_PRIMARY_SCENARIO=true`, scenario generation retrieves Neo4j assertion relations for
the same document version, uses Azure embeddings and the Hugging Face reranker, and invokes Azure
reasoning. It consumes canonical user stories and persists `test_scenarios.json` separately.

```powershell
uv run python -m multi_agentic_graph_rag generate-test-scenarios `
  --project $Project `
  --document-version-id $DocumentVersionId `
  --reasoning-provider azure_openai `
  --embedding-provider azure_openai `
  --reranker-provider huggingface `
  --top-k 4 `
  --no-hfil
```

Use `--hfil` only when an operator is present to review test scenarios before final persistence.
The production artifacts remain PostgreSQL-first and version-scoped regardless of review mode.

## Clean End-to-End Run Order

For a new local environment, use this order:

1. `uv sync --python 3.12 --dev --extra azure --extra local-llm`
2. Create `.env` with the strict provider profile.
3. Start PostgreSQL and Neo4j.
4. `uv run python -m multi_agentic_graph_rag config-check`
5. `uv run python -m multi_agentic_graph_rag doctor`
6. `uv run python -m multi_agentic_graph_rag db-check`
7. Run the Azure reasoning, Azure embedding, and Hugging Face reranker probes.
8. Run `ingest` and capture its `DV-...` identifier.
9. Verify or explicitly rebuild the semantic knowledge graph.
10. Run graph-primary user-story generation.
11. Run graph-primary test-scenario generation.

Do not reuse a `document_version_id` from another project or document version. Do not replace an
immutable version unless the repository's explicit replacement policy is intentionally required.
