# Agentic-GraphRAG

Multi-Agentic Knowledge-Graph RAG platform for converting source documents into traceable, validated requirement artifacts through a deterministic, phase-based ingestion architecture.

The current implementation contains the completed **Phase 1** and **Phase 2** foundation:

- **Phase 1:** Repository, package, CLI, dependency, and development-tool foundation.
- **Phase 2:** Centralized validated configuration, runtime path management, cache redirection, provider selection, and configuration diagnostics.

---

## 1. Current Repository Status

Repository:

```text
rdmpro2020/Agentic-GraphRAG
```

Current default branch:

```text
main
```

Current completed milestone:

```text
phase-2-complete
```

Current latest main commit:

```text
5a971418 feat: add phase 2 configuration and runtime path management
```

Previous Phase 1 baseline commit:

```text
a225cc7f chore: establish phase 1 repository foundation
```

Project package name:

```text
multi-agentic-graph-rag
```

Python import package:

```python
import multi_agentic_graph_rag
```

CLI command:

```powershell
marag
```

---

## 2. Project Purpose

Agentic-GraphRAG is designed as a production-oriented document-to-requirement pipeline.

Target workflow:

```text
Document ingestion
    ↓
Deterministic LangGraph workflow
    ↓
PostgreSQL canonical registry
Neo4j graph projection
ChromaDB semantic retrieval index
Filesystem immutable artifacts
    ↓
Requirement discovery
    ↓
Validated requirements_<timestamp>_<version>.json
```

The project separates deterministic workflow orchestration from bounded LLM-based extraction.

The intended long-term system will support:

- Document ingestion.
- Source file registration.
- Parsing and chunking.
- Traceable chunk manifests.
- Canonical persistence in PostgreSQL.
- Graph projection into Neo4j.
- Semantic chunk indexing in ChromaDB.
- Structured LLM requirement discovery.
- Evidence verification.
- Fact and requirement normalization.
- Permanent requirement and fact ID allocation.
- Atomic generated JSON artifacts.
- Resume and reconciliation support.

---

## 3. Architectural Ownership Model

The project does **not** treat every database as the source of truth.

Storage ownership is split by responsibility:

| Storage | Responsibility |
|---|---|
| PostgreSQL | Canonical transactional registry for projects, documents, versions, runs, chunks, facts, requirements, evidence, artifacts, and projection status |
| Neo4j | Relationship model and traceability graph for traversal, impact analysis, and multi-hop reasoning |
| ChromaDB | Rebuildable semantic retrieval index for chunk search |
| Filesystem | Preserved source snapshots, manifests, generated JSON artifacts, logs, staging, locks, and local runtime state |
| LangGraph checkpoints | Resumable workflow execution state |

Canonical rule:

```text
PostgreSQL controls completion state.
Neo4j, ChromaDB, and generated artifacts are projections registered back in PostgreSQL.
```

---

## 4. Current Implementation Scope

The current repository implements foundational infrastructure only.

Implemented:

- Python 3.12 project pinning.
- `uv`-managed installable Python package.
- `src/` package layout.
- Hatchling build backend.
- Version loading through `importlib.metadata`.
- Typer/Rich CLI.
- `marag version`.
- `marag config-check`.
- `marag doctor`.
- Central validated configuration model.
- Config precedence model.
- Provider enum validation.
- Runtime path bootstrap.
- Cache environment redirection.
- Secret masking in diagnostics.
- `.env.example`.
- `config.json`.
- `.gitignore`.
- Ruff, mypy, and pre-commit hooks.
- Local runtime placeholder directories with `.gitkeep`.

Not implemented yet:

- PostgreSQL schema and repositories.
- Alembic migrations.
- Neo4j client and projections.
- Chroma adapter and embedding indexing.
- LangGraph ingestion workflow.
- Document parsing and chunking.
- LLM requirement discovery.
- Requirement reconciliation.
- Artifact generation.
- Resume/reconciliation commands.

Those belong to later phases.

---

## 5. Repository Layout

Current repository layout:

```text
Agentic-GraphRAG/
├── .env.example
├── .gitignore
├── .global_cache/
│   └── .gitkeep
├── .pre-commit-config.yaml
├── .python-version
├── README.md
├── config.json
├── documents/
│   └── inbox/
│       └── .gitkeep
├── generated_artifacts/
│   └── .gitkeep
├── pyproject.toml
├── runtime/
│   ├── databases/
│   │   └── chroma/
│   │       └── .gitkeep
│   ├── locks/
│   │   └── .gitkeep
│   ├── logs/
│   │   └── .gitkeep
│   └── staging/
│       └── .gitkeep
├── src/
│   └── multi_agentic_graph_rag/
│       ├── __init__.py
│       ├── __main__.py
│       ├── bootstrap.py
│       ├── cli.py
│       └── config/
│           ├── __init__.py
│           ├── check.py
│           ├── paths.py
│           ├── providers.py
│           └── settings.py
└── uv.lock
```

---

## 6. Python and Package Requirements

Python is pinned to:

```text
3.12
```

Project constraint:

```toml
requires-python = ">=3.12,<3.13"
```

The distribution package is:

```toml
name = "multi-agentic-graph-rag"
```

The import package is:

```python
multi_agentic_graph_rag
```

The CLI entry point is:

```toml
[project.scripts]
marag = "multi_agentic_graph_rag.cli:app"
```

---

## 7. Runtime Dependencies

Current runtime dependencies include:

```text
alembic
asyncpg
charset-normalizer
chromadb
langchain-core
langchain-text-splitters
langgraph
neo4j
pydantic
pydantic-settings
pymupdf
python-docx
rich
sqlalchemy
structlog
tenacity
typer
```

Optional Azure dependencies:

```text
azure-identity
openai
```

Optional local LLM / Hugging Face dependencies:

```text
accelerate
huggingface-hub
sentence-transformers
torch
transformers
```

Development dependencies:

```text
ruff
mypy
pre-commit
```

---

## 8. Phase 1 Summary

Phase 1 established the repository and environment foundation.

Completed work:

- Initialized Git repository.
- Confirmed `main` as the default branch.
- Configured Git identity.
- Initialized `uv` project.
- Pinned Python 3.12 in `.python-version`.
- Created installable `src/` layout package.
- Used distribution name `multi-agentic-graph-rag`.
- Used Python import package name `multi_agentic_graph_rag`.
- Configured Hatchling build backend.
- Set project version to `0.1.0`.
- Implemented package version loading through `importlib.metadata`.
- Added `marag` CLI entry point.
- Implemented:
  - `marag version`
  - `marag config-check`
  - `marag doctor`
- Added `.gitignore`.
- Added `.env.example`.
- Added placeholder runtime directories.
- Added `.gitkeep` files for empty tracked directories.
- Added Ruff, mypy, and pre-commit configuration.
- Added locked dependencies through `uv.lock`.

Phase 1 validation commands:

```powershell
uv sync
uv lock --check
uv run marag version
uv run marag config-check
uv run marag doctor
uv run python -c "import multi_agentic_graph_rag"
uv run ruff check
uv run ruff format --check
uv run mypy src
uv build
```

---

## 9. Phase 2 Summary

Phase 2 added centralized configuration and runtime path management.

Completed work:

- Added `config.json`.
- Expanded `.env.example`.
- Created `multi_agentic_graph_rag.config` package.
- Added provider enums.
- Added settings models.
- Added path bootstrap helpers.
- Added config diagnostic helpers.
- Updated `marag config-check` to validate real Phase 2 settings.
- Added CLI override support for config validation.
- Added config precedence:
  - CLI arguments
  - Environment variables / `.env`
  - `config.json`
  - Safe code defaults
- Added runtime directory creation from approved path settings.
- Added cache environment redirection.
- Added provider selection validation.
- Added secret masking in configuration diagnostics.
- Added strict unknown-field rejection through Pydantic models.
- Fixed Ruff formatting and import ordering.
- Fixed strict mypy issues.
- Pushed Phase 2 directly to `main`.
- Tagged the milestone as `phase-2-complete`.

Phase 2 commit:

```text
5a971418 feat: add phase 2 configuration and runtime path management
```

Phase 2 tag:

```text
phase-2-complete
```

---

## 10. Configuration Precedence

Configuration is resolved using this precedence:

```text
CLI arguments
    ↓
Environment variables / .env
    ↓
config.json
    ↓
Safe code defaults
```

This allows safe defaults in source control while keeping secrets local.

---

## 11. Configuration Files

### 11.1 `config.json`

`config.json` contains non-secret application policy defaults.

Current categories:

```json
{
  "application": {},
  "paths": {},
  "providers": {},
  "embedding": {},
  "requirement_discovery": {},
  "chunking": {},
  "observability": {}
}
```

`config.json` should not contain:

- API keys.
- Database passwords.
- Neo4j passwords.
- Hugging Face tokens.
- Private document content.

### 11.2 `.env.example`

`.env.example` documents all supported local environment variables.

Copy it to `.env` for local development:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`.

`.gitignore` excludes:

```text
.env
.env.*
```

and explicitly keeps:

```text
!.env.example
```

---

## 12. Provider Selection

Provider settings are explicit and separate.

Do not use a single overloaded `LLM_PROVIDER`.

Current provider variables:

```env
REASONING_LLM_PROVIDER=azure_openai
EMBEDDING_PROVIDER=huggingface
VECTOR_STORE_PROVIDER=chroma_local
GRAPH_STORE_PROVIDER=neo4j_local
```

Supported reasoning LLM providers:

```text
azure_openai
huggingface
```

Supported embedding providers:

```text
azure_openai
huggingface
```

Supported vector store providers:

```text
chroma_local
chroma_http
chroma_cloud
```

Supported graph store providers:

```text
neo4j_local
neo4j_remote
```

Unknown provider values are rejected by configuration validation.

---

## 13. Runtime Path Management

Approved runtime directories:

```text
.global_cache/
documents/inbox/
generated_artifacts/
runtime/databases/chroma/
runtime/staging/
runtime/logs/
runtime/locks/
```

`marag config-check` creates missing approved runtime directories.

Runtime path rules:

- Paths are resolved relative to the project root unless absolute.
- Paths are validated to remain under the project root.
- Unapproved path escape is rejected.
- Cache paths are placed under `.global_cache`.
- Durable runtime data is kept outside `.global_cache`.

---

## 14. Cache Redirection

The application redirects cache-related environment variables to `.global_cache`.

Configured cache variables:

```text
HF_HOME
HF_HUB_CACHE
TRANSFORMERS_CACHE
TORCH_HOME
XDG_CACHE_HOME
UV_CACHE_DIR
```

Resolved structure:

```text
.global_cache/
├── huggingface/
│   └── hub/
├── transformers/
├── torch/
└── uv/
```

Rule:

```text
.global_cache is disposable.
Do not store durable generated artifacts or ChromaDB data in .global_cache.
```

Durable Chroma state belongs under:

```text
runtime/databases/chroma/
```

Generated artifacts belong under:

```text
generated_artifacts/
```

---

## 15. CLI Commands

### 15.1 Version

```powershell
uv run marag version
```

Expected output:

```text
marag 0.1.0
```

### 15.2 Configuration Check

```powershell
uv run marag config-check
```

Validates:

- Project root.
- Runtime directories.
- Cache paths.
- Required selected-provider settings.
- PostgreSQL DSN presence.
- Neo4j settings.
- Azure settings when `REASONING_LLM_PROVIDER=azure_openai`.
- Provider enum values.
- Secret masking.

### 15.3 Configuration Check with Overrides

```powershell
uv run marag config-check `
  --reasoning-provider huggingface `
  --embedding-provider huggingface `
  --vector-store-provider chroma_local `
  --graph-store-provider neo4j_local
```

### 15.4 Doctor

```powershell
uv run marag doctor
```

Validates:

- Python version.
- `uv` executable.
- `git` executable.
- Installed package metadata.

---

## 16. Local Development Setup

Clone and enter the repository:

```powershell
git clone git@gitlab.com:rdmpro2020/Agentic-GraphRAG.git
cd Agentic-GraphRAG
```

Install dependencies:

```powershell
uv sync
```

Create local `.env`:

```powershell
Copy-Item .env.example .env
notepad .env
```

Run basic checks:

```powershell
uv lock --check
uv run marag version
uv run marag config-check
uv run marag doctor
```

Run quality gates:

```powershell
uv run ruff check
uv run ruff format --check
uv run mypy src
uv build
```

---

## 17. Example `.env`

### Azure reasoning + Hugging Face embeddings

```env
APP_ENV=development
LOG_LEVEL=INFO

REASONING_LLM_PROVIDER=azure_openai
EMBEDDING_PROVIDER=huggingface
VECTOR_STORE_PROVIDER=chroma_local
GRAPH_STORE_PROVIDER=neo4j_local

POSTGRES_DSN=postgresql+asyncpg://marag:<password>@127.0.0.1:5432/marag

NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

AZURE_OPENAI_ENDPOINT=<azure-openai-endpoint>
AZURE_OPENAI_API_KEY=<azure-openai-api-key>
AZURE_OPENAI_API_VERSION=<azure-openai-api-version>
AZURE_OPENAI_REASONING_DEPLOYMENT=<reasoning-deployment>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=

HUGGINGFACE_TOKEN=
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_REASONING_MODEL=
HUGGINGFACE_OFFLINE=false
```

### Hugging Face reasoning + Hugging Face embeddings

```env
APP_ENV=development
LOG_LEVEL=INFO

REASONING_LLM_PROVIDER=huggingface
EMBEDDING_PROVIDER=huggingface
VECTOR_STORE_PROVIDER=chroma_local
GRAPH_STORE_PROVIDER=neo4j_local

POSTGRES_DSN=postgresql+asyncpg://marag:<password>@127.0.0.1:5432/marag

NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=neo4j

AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_REASONING_DEPLOYMENT=
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=

HUGGINGFACE_TOKEN=
HUGGINGFACE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_REASONING_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct
HUGGINGFACE_OFFLINE=false
```

---

## 18. Configuration Diagnostics Behavior

`marag config-check` prints a Rich table.

Example validated categories:

```text
Project root
Path: .global_cache
Path: inbox
Path: generated_artifacts
Path: chroma
Path: staging
Path: logs
Path: locks
HF_HOME
HF_HUB_CACHE
TRANSFORMERS_CACHE
TORCH_HOME
XDG_CACHE_HOME
UV_CACHE_DIR
POSTGRES_DSN
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_REASONING_DEPLOYMENT
Reasoning provider
Embedding provider
Vector store provider
Graph store provider
```

Secrets are masked:

```text
POSTGRES_DSN       postgresql+asyncpg://marag:********@127.0.0.1:5432/marag
NEO4J_PASSWORD     ********
AZURE_OPENAI_KEY   ********
```

---

## 19. Development Quality Gates

The repository uses local pre-commit hooks:

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
      - id: ruff-format-check
      - id: mypy
```

Manual quality gate:

```powershell
uv run ruff check
uv run ruff format --check
uv run mypy src
uv build
```

Commit gate:

```powershell
git commit -m "message"
```

Pre-commit runs:

```text
Ruff lint
Ruff format check
mypy strict type check
```

---

## 20. Build

Build the package:

```powershell
uv build
```

Build backend:

```text
hatchling
```

Wheel package root:

```text
src/multi_agentic_graph_rag
```

---

## 21. Versioning

The project version is defined in `pyproject.toml`:

```toml
version = "0.1.0"
```

Runtime version loading is performed through package metadata:

```python
from importlib.metadata import version
```

The project does **not** load version information from `.env`.

---

## 22. Security Rules

Current repository security practices:

- `.env` is ignored.
- `.env.example` is committed.
- Secrets are not stored in `config.json`.
- Diagnostic output masks passwords, API keys, tokens, and DSN passwords.
- Runtime caches are ignored.
- Generated artifacts are ignored.
- Local Chroma database files are ignored.
- Runtime logs, staging files, and locks are ignored.
- Full private documents should not be logged by default.
- Full prompts containing private document text should not be logged by default.

---

## 23. Git Ignore Policy

Ignored local-only data:

```text
.venv/
.env
.env.*
.global_cache/*
documents/inbox/*
generated_artifacts/*
runtime/databases/chroma/*
runtime/staging/*
runtime/locks/*
runtime/logs/*
*.log
```

Tracked placeholders:

```text
.global_cache/.gitkeep
documents/inbox/.gitkeep
generated_artifacts/.gitkeep
runtime/databases/chroma/.gitkeep
runtime/staging/.gitkeep
runtime/locks/.gitkeep
runtime/logs/.gitkeep
```

---

## 24. Current Source Modules

### `multi_agentic_graph_rag.__init__`

Responsibilities:

- Defines distribution name.
- Exposes `__version__`.
- Loads installed version through `importlib.metadata`.

### `multi_agentic_graph_rag.__main__`

Responsibilities:

- Supports:

```powershell
uv run python -m multi_agentic_graph_rag
```

### `multi_agentic_graph_rag.cli`

Responsibilities:

- Defines Typer CLI app.
- Implements:
  - `version`
  - `config-check`
  - `doctor`
- Renders diagnostic tables through Rich.
- Supports provider override options for config validation.

### `multi_agentic_graph_rag.bootstrap`

Responsibilities:

- Defines diagnostic result contract.
- Performs configuration checks.
- Performs environment doctor checks.
- Creates approved runtime directories.
- Sets cache environment variables.

### `multi_agentic_graph_rag.config.providers`

Responsibilities:

- Defines provider enums:
  - `ReasoningLLMProvider`
  - `EmbeddingProvider`
  - `VectorStoreProvider`
  - `GraphStoreProvider`

### `multi_agentic_graph_rag.config.paths`

Responsibilities:

- Finds project root.
- Resolves paths under project root.
- Rejects path escape.
- Creates approved directories.
- Sets cache environment variables.

### `multi_agentic_graph_rag.config.settings`

Responsibilities:

- Defines full application settings.
- Loads settings from `config.json`, `.env`, environment, and CLI overrides.
- Validates provider values.
- Validates log level.
- Resolves runtime paths.
- Defines cache paths.
- Produces diagnostic checks.
- Masks secrets.

### `multi_agentic_graph_rag.config.check`

Responsibilities:

- Reports runtime directory existence.
- Converts settings to a redacted dictionary.

---

## 25. Current Validation Snapshot

The Phase 2 implementation has passed:

```text
uv sync
uv lock --check
uv run marag version
uv run marag config-check
uv run marag doctor
uv run ruff check
uv run ruff format --check
uv run mypy src
pre-commit Ruff lint
pre-commit Ruff format check
pre-commit mypy strict type check
```

Current successful CLI outputs include:

```text
marag 0.1.0
```

Configuration diagnostics now resolve:

```text
D:\Repos\Agentic-GraphRAG
```

and validate provider defaults:

```text
Reasoning provider     azure_openai
Embedding provider     huggingface
Vector store provider  chroma_local
Graph store provider   neo4j_local
```

---

## 26. Important Current Limitation

The current `config-check` validates the **presence and structure** of selected provider configuration.

It does not yet verify:

- PostgreSQL network connectivity.
- Neo4j network connectivity.
- Azure OpenAI deployment existence.
- Hugging Face model download.
- Chroma collection creation.
- Actual ingestion workflow execution.

Those checks belong to later phases.

---

## 27. Next Planned Phase

The next implementation phase is:

```text
Phase 3 — Domain contracts, identifiers and versioning
```

Primary goal:

```text
Define stable Pydantic domain models before implementing databases or workflow nodes.
```

Expected Phase 3 work:

- Add command models:
  - `IngestDocumentCommand`
  - `ResumeRunCommand`
  - `RebuildProjectionCommand`
  - `VerifyArtifactCommand`
- Add immutable domain models:
  - `Project`
  - `Document`
  - `DocumentVersion`
  - `ParsedDocument`
  - `Chunk`
  - `ChunkManifest`
  - `FactCandidate`
  - `RequirementCandidate`
  - `EvidenceReference`
  - `DiscoveryBatch`
  - `DiscoveryResult`
  - `RequirementArtifact`
  - `IngestionRun`
  - `RunStep`
- Add identifier strategy:
  - Internal UUIDs.
  - Public traceability IDs.
  - Deterministic chunk IDs.
  - Permanent fact and requirement IDs allocated only after validation.
- Add document versioning rules.
- Add run state machine.
- Generate JSON Schema from Pydantic models.
- Ensure domain code has no SQLAlchemy, Neo4j, Chroma, Azure, or Transformers imports.

---

## 28. Non-Negotiable Design Rules

The following rules guide all future phases:

```text
LangGraph nodes orchestrate application services.
LangGraph nodes do not contain large database queries or parsing implementations.
Domain code cannot import SQLAlchemy, Neo4j, Chroma, Azure, or Transformers.
The LLM never creates permanent IDs.
The LLM never writes files or databases.
PostgreSQL records every run and step transition.
Chroma is always rebuildable.
Neo4j writes are idempotent and constraint-backed.
Full documents are not stored in LangGraph state.
Embedding vectors are not stored in LangGraph state.
Configuration is validated once at the composition root.
Secrets exist only in environment variables or an external secret manager.
Same-version content replacement is never implicit.
A requirement without verified evidence is not marked validated.
A run is not completed until mandatory artifacts and projections are verified.
Only importable Python source directories receive __init__.py.
```

---

## 29. Useful Commands

### Install

```powershell
uv sync
```

### Validate lockfile

```powershell
uv lock --check
```

### Print version

```powershell
uv run marag version
```

### Validate configuration

```powershell
uv run marag config-check
```

### Validate local environment

```powershell
uv run marag doctor
```

### Run module directly

```powershell
uv run python -m multi_agentic_graph_rag
```

### Import package

```powershell
uv run python -c "import multi_agentic_graph_rag; print(multi_agentic_graph_rag.__version__)"
```

### Check provider selection

```powershell
uv run python -c "from multi_agentic_graph_rag.config.settings import load_settings; s = load_settings(); print(s.providers.vector_store_provider)"
```

### Check project root

```powershell
uv run python -c "from multi_agentic_graph_rag.config.paths import find_project_root; print(find_project_root())"
```

### Lint

```powershell
uv run ruff check
```

### Format check

```powershell
uv run ruff format --check
```

### Format write

```powershell
uv run ruff format
```

### Type check

```powershell
uv run mypy src
```

### Build

```powershell
uv build
```

### Full local gate

```powershell
uv sync
uv lock --check
uv run marag version
uv run marag config-check
uv run marag doctor
uv run ruff check
uv run ruff format --check
uv run mypy src
uv build
```

---

## 30. Current Development Baseline

The repository is currently a clean Phase 2 foundation.

It is ready for Phase 3 once the following remains true:

```text
All configuration validation passes.
All runtime paths resolve under the project root.
Secrets remain outside Git.
Provider values are explicit enums.
Ruff passes.
Ruff format check passes.
mypy strict passes.
Package build passes.
```

---

## 31. Phase 3 Completion Summary

Phase 3 has been implemented and validated.

Completed milestone:

```text
phase-3-complete
```

Phase 3 added the infrastructure-free domain contract layer for the Multi-Agentic GraphRAG platform.

Primary objective:

```text
Define stable Pydantic domain contracts, identifier rules, document-versioning rules, evidence-trace contracts, requirement artifact contracts, and run-state transition validation before implementing PostgreSQL, Neo4j, ChromaDB, LangGraph workflow nodes, document parsing, chunking, or LLM extraction services.
```

Phase 3 converts the repository from a Phase 2 configuration foundation into a project with stable core business contracts for:

- Document ingestion commands.
- Provider override commands.
- Project identity.
- Logical document identity.
- Document version identity.
- Parsed document output.
- Parsed text block traceability.
- Chunk contracts.
- Chunk manifest contracts.
- Evidence references.
- Fact candidates.
- Validated facts.
- Requirement candidates.
- Validated requirements.
- Discovery batches.
- Discovery results.
- Requirement artifacts.
- Ingestion run lifecycle tracking.
- Run step tracking.
- Public traceability identifiers.
- Temporary LLM identifiers.
- Deterministic chunk identifiers.
- Run state transition validation.
- JSON Schema generation from domain models.

Phase 3 is intentionally still **not** an ingestion implementation.

It does not yet implement:

- PostgreSQL tables.
- Alembic migrations.
- SQLAlchemy repositories.
- Neo4j client writes.
- Chroma indexing.
- LangGraph workflow execution.
- PDF/DOCX/TXT parsing.
- Actual chunk creation.
- LLM extraction.
- Artifact writing.
- Resume command execution.

Those belong to later phases.

---

## 32. Phase 3 Source Modules

Phase 3 added the domain package:

```text
src/multi_agentic_graph_rag/domain/
```

Added domain modules:

```text
src/multi_agentic_graph_rag/domain/__init__.py
src/multi_agentic_graph_rag/domain/chunks.py
src/multi_agentic_graph_rag/domain/commands.py
src/multi_agentic_graph_rag/domain/documents.py
src/multi_agentic_graph_rag/domain/enums.py
src/multi_agentic_graph_rag/domain/errors.py
src/multi_agentic_graph_rag/domain/evidence.py
src/multi_agentic_graph_rag/domain/facts.py
src/multi_agentic_graph_rag/domain/identifiers.py
src/multi_agentic_graph_rag/domain/requirements.py
src/multi_agentic_graph_rag/domain/runs.py
```

Added Phase 3 schema verification script:

```text
scripts/phase3_schema_check.py
```

Generated schema draft output directory:

```text
generated_artifacts/schema_drafts/
```

Generated schema files:

```text
generated_artifacts/schema_drafts/IngestDocumentCommand.schema.json
generated_artifacts/schema_drafts/ParsedDocument.schema.json
generated_artifacts/schema_drafts/ChunkManifest.schema.json
generated_artifacts/schema_drafts/FactCandidate.schema.json
generated_artifacts/schema_drafts/RequirementCandidate.schema.json
generated_artifacts/schema_drafts/DiscoveryBatch.schema.json
generated_artifacts/schema_drafts/DiscoveryResult.schema.json
generated_artifacts/schema_drafts/RequirementArtifact.schema.json
generated_artifacts/schema_drafts/IngestionRun.schema.json
generated_artifacts/schema_drafts/RunStep.schema.json
```

---

## 33. Phase 3 Domain Package Purpose

The `domain` package defines the core business vocabulary and validation contracts for the platform.

The domain package must remain infrastructure-free.

It must not import:

```text
sqlalchemy
asyncpg
neo4j
chromadb
azure
openai
transformers
torch
langgraph runtime clients
```

The domain package is intentionally separate from:

```text
config/
application/
infrastructure/
agents/
interfaces/
```

Reason:

```text
Domain contracts define what the system means by project, document, version, chunk, evidence, fact, requirement, artifact, ingestion run, and run step.

They are not database tables.
They are not API request schemas.
They are not LLM-provider-specific schemas.
They are not workflow-node implementations.
They are not persistence models.
```

Domain models are allowed to be Pydantic models because they act as validated business contracts.

They are not placed under a generic `schemas/` directory because they represent stable business concepts, not boundary-specific request/response payloads.

Boundary-specific schemas should later live closer to their boundary, for example:

```text
agents/requirement_discovery/schemas.py
interfaces/api/schemas.py
infrastructure/postgres/models.py
```

---

## 34. Phase 3 Command Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/commands.py
```

Added command models:

```text
ProviderOverrides
IngestDocumentCommand
ResumeRunCommand
RebuildProjectionCommand
VerifyArtifactCommand
```

Responsibilities:

```text
ProviderOverrides
  Captures optional per-command provider overrides for reasoning, embeddings, vector store, and graph store.

IngestDocumentCommand
  Represents the validated input contract for ingesting one logical document version.

ResumeRunCommand
  Represents the validated input contract for resuming an existing ingestion run.

RebuildProjectionCommand
  Represents the validated input contract for rebuilding a derived projection such as Neo4j or Chroma.

VerifyArtifactCommand
  Represents the validated input contract for verifying an emitted requirement artifact.
```

Important behavior:

```text
Command models are immutable.
Unknown fields are rejected.
project_key is normalized for ingestion.
document_version is validated through shared version-normalization logic.
Provider overrides are optional.
Provider overrides do not replace global configuration.
```

Example ingestion command responsibility:

```text
Raw CLI/API input
  → IngestDocumentCommand
  → application service / future LangGraph workflow
  → parser / chunker / persistence / projection phases
```

---

## 35. Phase 3 Enum Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/enums.py
```

Added enums:

```text
ReplacePolicy
FactType
RequirementType
DerivationType
ValidationStatus
RunStatus
RunStepName
```

Responsibilities:

```text
ReplacePolicy
  Controls behavior when the same logical document version already exists.

FactType
  Classifies extracted facts.

RequirementType
  Classifies final requirements.

DerivationType
  Describes whether a requirement is direct, normalized, inferred, or cross-chunk derived.

ValidationStatus
  Tracks whether a fact or requirement is candidate, validated, rejected, needs review, or conflicting.

RunStatus
  Defines allowed ingestion run lifecycle states.

RunStepName
  Defines stable workflow step names for run-step tracking.
```

Purpose:

```text
Avoid magic strings.
Provide stable JSON-serializable values.
Support deterministic workflow routing.
Support PostgreSQL persistence later.
Support Neo4j projection later.
Support Chroma metadata later.
Support strict validation of generated artifacts.
Support consistent logs and diagnostics.
```

---

## 36. Phase 3 Error Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/errors.py
```

Added errors:

```text
DomainError
IdentifierError
InvalidRunTransitionError
```

Responsibilities:

```text
DomainError
  Base exception for domain-layer failures.

IdentifierError
  Raised when public IDs, temporary keys, versions, or deterministic identifiers are invalid.

InvalidRunTransitionError
  Raised when an ingestion run attempts to move through an illegal state transition.
```

Reason:

```text
Domain failures should have explicit domain-level exception types instead of generic exceptions spread across the codebase.
```

---

## 37. Phase 3 Identifier and Versioning Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/identifiers.py
```

Added helpers:

```text
validate_public_id
validate_temporary_key
normalize_version
sha256_text
make_chunk_id
```

Identifier rules:

```text
Internal database identity uses UUIDs.
Public traceability IDs use stable prefixes such as FACT-* and REQ-*.
LLM outputs use temporary keys such as F1 and R1.
The LLM does not create permanent fact IDs.
The LLM does not create permanent requirement IDs.
Permanent public IDs are allocated only after validation.
Chunk IDs are deterministic.
Document versions are normalized for comparison.
Document versions are not forced into numeric parsing.
```

Validation behavior:

```text
validate_public_id
  Validates public traceability IDs such as FACT-000001 and REQ-000001.

validate_temporary_key
  Validates temporary LLM-local keys such as F1 and R1.

normalize_version
  Normalizes document versions for comparison while preserving support for non-numeric versions.

sha256_text
  Produces deterministic SHA-256 text hashes.

make_chunk_id
  Produces deterministic CHUNK-* identifiers from document_version_id, ordinal, and content_hash.
```

Supported identifier split:

```text
Internal primary key
  UUID

Public traceability ID
  FACT-000001
  REQ-000001
  CHUNK-<deterministic-hash>

Temporary LLM-local key
  F1
  R1
```

This prevents the LLM from controlling permanent system identity.

---

## 38. Phase 3 Document Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/documents.py
```

Added models:

```text
Project
Document
DocumentVersion
ParsedBlock
ParsedDocument
```

Responsibilities:

```text
Project
  Represents one project workspace.

Document
  Represents one logical document inside a project.

DocumentVersion
  Represents one exact version of a logical document.

ParsedBlock
  Represents one parser-produced source text block with page, section, paragraph, and character offset metadata.

ParsedDocument
  Represents the complete parsed output for one document version.
```

Traceability chain:

```text
Project
  → Document
    → DocumentVersion
      → ParsedDocument
        → ParsedBlock
```

Important rules:

```text
DocumentVersion stores supplied_version.
DocumentVersion stores normalized_version.
DocumentVersion stores source_checksum.
DocumentVersion supports supersedes_document_version_id for version lineage.
DocumentVersion supports parser_fingerprint.
DocumentVersion supports chunker_fingerprint.
DocumentVersion supports embedding_fingerprint.
DocumentVersion supports prompt_fingerprint.
ParsedBlock preserves raw_text.
ParsedBlock preserves normalized_text.
ParsedBlock preserves page number where applicable.
ParsedBlock preserves section path where available.
ParsedBlock preserves paragraph number where available.
ParsedBlock preserves character offsets for later evidence validation.
```

The document contracts do not parse files.

They define the valid shape of parsed document outputs that later parsing infrastructure must produce.

---

## 39. Phase 3 Chunk Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/chunks.py
```

Added models:

```text
Chunk
ChunkManifestEntry
ChunkManifest
```

Responsibilities:

```text
Chunk
  Represents one final retrieval-ready, source-traceable text unit.

ChunkManifestEntry
  Represents lightweight metadata for one chunk.

ChunkManifest
  Represents the full chunk manifest for one document version.
```

Important distinction:

```text
chunks.py does not create chunks.
chunks.py defines what a valid chunk must look like after a chunking service creates it.
```

Actual chunk creation belongs to a later parsing/chunking implementation phase.

Chunk validation rules:

```text
chunk_id must not be empty.
document_version_id is required.
ordinal must be greater than or equal to 1.
content_hash must be 64 characters.
normalized_text must not be empty.
original_text must not be empty.
character_start must be greater than or equal to 0.
character_end must be greater than character_start.
page_start and page_end must be greater than or equal to 1 when present.
page_end cannot be less than page_start.
```

Traceability chain:

```text
ParsedDocument
  → Chunk
    → ChunkManifest
      → PostgreSQL canonical chunk records
      → Neo4j chunk projection
      → Chroma vector index
      → EvidenceReference
      → Fact
      → Requirement
      → RequirementArtifact
```

Storage intention:

```text
PostgreSQL
  Stores canonical chunks and manifests later.

Neo4j
  Stores projected Chunk nodes and relationships later.

ChromaDB
  Stores chunk vectors and chunk metadata later.

Generated artifacts
  Refer back to chunks through source_trace later.
```

---

## 40. Phase 3 Evidence Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/evidence.py
```

Added model:

```text
EvidenceReference
```

Responsibilities:

```text
Represents a verified or candidate evidence location.
Links a fact or requirement back to a chunk.
Stores exact quoted source text.
Optionally stores document_version_id.
Optionally stores character_start and character_end offsets.
Rejects invalid character offset ranges.
```

Validation rules:

```text
chunk_id must not be empty.
exact_quote must not be empty.
character_start must be greater than or equal to 0 when present.
character_end must be greater than or equal to 0 when present.
If both offsets are present, character_end must be greater than character_start.
```

Purpose:

```text
EvidenceReference is the source-trace bridge between generated facts/requirements and exact source document text.
```

Traceability role:

```text
Requirement
  → supporting Fact
    → EvidenceReference
      → Chunk
        → DocumentVersion
          → Document
            → Project
```

---

## 41. Phase 3 Fact Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/facts.py
```

Added models:

```text
FactCandidate
Fact
```

Responsibilities:

```text
FactCandidate
  Represents an untrusted candidate fact, usually produced from LLM extraction over chunks.

Fact
  Represents a validated, permanent fact with an internal UUID and public FACT-* traceability ID.
```

Important rules:

```text
FactCandidate uses temporary_fact_key such as F1.
Fact uses permanent fact_id such as FACT-000001.
FactCandidate defaults to ValidationStatus.CANDIDATE.
FactCandidate requires evidence references.
Fact requires evidence references.
Fact IDs are validated through validate_public_id.
Temporary fact keys are validated through validate_temporary_key.
```

Fact lifecycle:

```text
Chunk
  → LLM or extraction service proposes fact
  → FactCandidate
  → evidence verification
  → normalization and deduplication
  → permanent ID allocation
  → Fact
  → PostgreSQL persistence later
  → Neo4j projection later
  → requirement derivation later
```

The file does not extract facts.

It only defines what fact candidates and validated facts must look like.

---

## 42. Phase 3 Requirement Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/requirements.py
```

Added models:

```text
RequirementCandidate
Requirement
DiscoveryBatch
DiscoveryResult
RequirementArtifact
```

Responsibilities:

```text
RequirementCandidate
  Represents an untrusted candidate requirement, usually derived from facts and source chunks.

Requirement
  Represents a validated permanent requirement with a REQ-* public traceability ID.

DiscoveryBatch
  Represents one batch of discovered fact and requirement candidates.

DiscoveryResult
  Represents the normalized result of a requirement discovery operation.

RequirementArtifact
  Represents the final emitted requirement artifact contract.
```

Important rules:

```text
RequirementCandidate uses temporary_requirement_key such as R1.
Requirement uses permanent requirement_id such as REQ-000001.
Requirement must include supporting facts.
Requirement must include source evidence.
RequirementArtifact total_requirements must match the number of requirements.
RequirementArtifact contains traceable validated requirement records.
```

Requirement lifecycle:

```text
FactCandidate / Fact
  → RequirementCandidate
  → validation
  → supporting fact verification
  → source evidence verification
  → permanent requirement ID allocation
  → Requirement
  → RequirementArtifact
```

The file does not call the LLM.

It does not perform requirement discovery.

It only defines valid requirement discovery and artifact contracts.

---

## 43. Phase 3 Run Contracts

Implemented in:

```text
src/multi_agentic_graph_rag/domain/runs.py
```

Added models and functions:

```text
IngestionRun
RunStep
validate_run_transition
```

Responsibilities:

```text
IngestionRun
  Represents the lifecycle of one ingestion run.

RunStep
  Represents one step executed inside an ingestion run.

validate_run_transition
  Validates legal transitions between RunStatus values.
```

Run state machine purpose:

```text
Prevent an ingestion run from being marked completed before required lifecycle states have been reached.
```

Valid transition tested:

```text
requested -> validated
```

Invalid transition tested and rejected:

```text
requested -> completed
```

Observed rejection:

```text
InvalidRunTransitionError: Invalid run transition: requested -> completed
```

Reason:

```text
A run cannot jump directly from requested to completed.
It must pass through the required lifecycle states for validation, registration, parsing, chunking, persistence, projection, indexing, requirement validation, artifact writing, and completion.
```

---

## 44. Phase 3 Domain Public API

Implemented in:

```text
src/multi_agentic_graph_rag/domain/__init__.py
```

Purpose:

```text
Expose a clean public API for commonly used domain contracts.
Keep internal module layout hidden from most callers.
Define the stable import surface of the domain package.
```

Example supported import style:

```python
from multi_agentic_graph_rag.domain import (
    IngestDocumentCommand,
    ReplacePolicy,
    RunStatus,
    validate_run_transition,
)
```

Instead of forcing every caller to import from separate internal files:

```python
from multi_agentic_graph_rag.domain.commands import IngestDocumentCommand
from multi_agentic_graph_rag.domain.enums import ReplacePolicy
from multi_agentic_graph_rag.domain.enums import RunStatus
from multi_agentic_graph_rag.domain.runs import validate_run_transition
```

Important rule:

```text
__init__.py exports selected stable contracts.
It should not become a dumping ground for every internal helper.
```

---

## 45. Phase 3 Schema Generation Script

Implemented in:

```text
scripts/phase3_schema_check.py
```

Purpose:

```text
Verify that selected Phase 3 Pydantic domain models can generate JSON Schema.
```

The script generates schema drafts under:

```text
generated_artifacts/schema_drafts/
```

Models validated by the script:

```text
IngestDocumentCommand
ParsedDocument
ChunkManifest
FactCandidate
RequirementCandidate
DiscoveryBatch
DiscoveryResult
RequirementArtifact
IngestionRun
RunStep
```

Command:

```powershell
uv run python scripts\phase3_schema_check.py
```

Observed output:

```text
PASS IngestDocumentCommand -> generated_artifacts\schema_drafts\IngestDocumentCommand.schema.json
PASS ParsedDocument -> generated_artifacts\schema_drafts\ParsedDocument.schema.json
PASS ChunkManifest -> generated_artifacts\schema_drafts\ChunkManifest.schema.json
PASS FactCandidate -> generated_artifacts\schema_drafts\FactCandidate.schema.json
PASS RequirementCandidate -> generated_artifacts\schema_drafts\RequirementCandidate.schema.json
PASS DiscoveryBatch -> generated_artifacts\schema_drafts\DiscoveryBatch.schema.json
PASS DiscoveryResult -> generated_artifacts\schema_drafts\DiscoveryResult.schema.json
PASS RequirementArtifact -> generated_artifacts\schema_drafts\RequirementArtifact.schema.json
PASS IngestionRun -> generated_artifacts\schema_drafts\IngestionRun.schema.json
PASS RunStep -> generated_artifacts\schema_drafts\RunStep.schema.json
```

Meaning:

```text
All selected Phase 3 Pydantic domain contracts can generate JSON Schema successfully.
```

Why this matters:

```text
The generated JSON Schema files can later support:
- LLM structured output validation.
- Artifact validation.
- API contract documentation.
- Regression tests.
- Contract drift detection.
```

---

## 46. Phase 3 Validation Commands

Phase 3 type checking command:

```powershell
uv run mypy src
```

Observed result:

```text
Success: no issues found in 20 source files
```

Phase 3 schema generation command:

```powershell
uv run python scripts\phase3_schema_check.py
```

Observed result:

```text
All selected schema generation checks passed.
```

Phase 3 Ruff format check:

```powershell
uv run ruff format --check src scripts
```

Observed result:

```text
21 files already formatted
```

Phase 3 Ruff lint check:

```powershell
uv run ruff check src scripts
```

Observed result:

```text
All checks passed!
```

Valid run transition command:

```powershell
uv run python -c "from multi_agentic_graph_rag.domain.enums import RunStatus; from multi_agentic_graph_rag.domain.runs import validate_run_transition; validate_run_transition(RunStatus.REQUESTED, RunStatus.VALIDATED); print('valid transition PASS')"
```

Observed result:

```text
valid transition PASS
```

Invalid run transition command:

```powershell
uv run python -c "from multi_agentic_graph_rag.domain.enums import RunStatus; from multi_agentic_graph_rag.domain.runs import validate_run_transition; validate_run_transition(RunStatus.REQUESTED, RunStatus.COMPLETED)"
```

Observed result:

```text
multi_agentic_graph_rag.domain.errors.InvalidRunTransitionError: Invalid run transition: requested -> completed
```

The invalid-transition failure is expected and correct.

---

## 47. Phase 3 Quality Gate Snapshot

Current Phase 3 validation status:

```text
mypy src
  PASS

phase3_schema_check.py
  PASS

ruff format --check src scripts
  PASS

ruff check src scripts
  PASS

valid run transition check
  PASS

invalid run transition rejection
  PASS
```

Phase 3 acceptance conditions satisfied:

```text
Domain contracts exist.
Domain contracts are immutable where required.
Unknown fields are rejected by Pydantic models.
Identifier helpers exist.
Temporary LLM keys are validated.
Public traceability IDs are validated.
Document versions are normalized.
Chunk IDs are deterministic.
Evidence offsets are validated.
Chunk ranges are validated.
Requirement artifact totals are validated.
Run transitions are validated.
JSON Schema generation succeeds.
Domain layer remains infrastructure-free.
```

---

## 48. Phase 3 Cleanup Note

During Phase 3 staging, accidental duplicate files were detected under:

```text
.src/
```

The real source folder is:

```text
src/
```

The accidental folder must not be committed.

Temporary file also detected:

```text
test.md
```

Cleanup commands:

```powershell
git restore --staged .src
Remove-Item -Recurse -Force .src

git restore --staged test.md
Remove-Item -Force test.md
```

After cleanup, verify:

```powershell
git status
```

The status must not include:

```text
.src/
test.md
```

Only valid Phase 3 paths should remain staged:

```text
src/multi_agentic_graph_rag/domain/
scripts/phase3_schema_check.py
generated_artifacts/schema_drafts/
README.md
```

---

## 49. Phase 3 Commit and Tag Commands

After cleanup and final validation, stage Phase 3 files:

```powershell
git add src\multi_agentic_graph_rag\domain scripts\phase3_schema_check.py generated_artifacts\schema_drafts README.md
```

Commit:

```powershell
git commit -m "feat: add phase 3 domain contracts"
```

Create milestone tag:

```powershell
git tag phase-3-complete
```

Push commit:

```powershell
git push origin main
```

Push tag:

```powershell
git push origin phase-3-complete
```

Verify final state:

```powershell
git status
git tag --list "phase-*"
```

Expected final working tree:

```text
nothing to commit, working tree clean
```

Expected milestone tags include:

```text
phase-2-complete
phase-3-complete
```

---

## 50. Updated Implementation Scope After Phase 3

Implemented through Phase 3:

- Python 3.12 project pinning.
- `uv`-managed installable Python package.
- `src/` package layout.
- Hatchling build backend.
- Version loading through `importlib.metadata`.
- Typer/Rich CLI.
- `marag version`.
- `marag config-check`.
- `marag doctor`.
- Central validated configuration model.
- Config precedence model.
- Provider enum validation.
- Runtime path bootstrap.
- Cache environment redirection.
- Secret masking in diagnostics.
- `.env.example`.
- `config.json`.
- `.gitignore`.
- Ruff, mypy, and pre-commit hooks.
- Local runtime placeholder directories with `.gitkeep`.
- Infrastructure-free domain package.
- Command contracts.
- Document contracts.
- Chunk contracts.
- Evidence contracts.
- Fact contracts.
- Requirement contracts.
- Ingestion run contracts.
- Run step contracts.
- Domain enums.
- Domain exceptions.
- Identifier validation helpers.
- Version normalization helpers.
- Deterministic chunk ID helper.
- Run transition validation.
- JSON Schema generation script.
- Generated schema drafts.

Still not implemented:

- PostgreSQL schema.
- SQLAlchemy ORM models.
- Alembic migration environment.
- PostgreSQL repositories.
- Neo4j client and projections.
- Chroma adapter and embedding indexing.
- LangGraph ingestion workflow.
- Document parsing implementation.
- Chunk creation implementation.
- LLM requirement discovery implementation.
- Evidence quote verification implementation.
- Requirement reconciliation implementation.
- Artifact writer implementation.
- Resume/reconciliation CLI commands.

---

## 51. Updated Current Validation Snapshot

The implementation through Phase 3 has passed:

```text
uv run ruff check src scripts
uv run ruff format --check src scripts
uv run mypy src
uv run python scripts\phase3_schema_check.py
```

Observed successful outputs include:

```text
All checks passed!
21 files already formatted
Success: no issues found in 20 source files
PASS IngestDocumentCommand
PASS ParsedDocument
PASS ChunkManifest
PASS FactCandidate
PASS RequirementCandidate
PASS DiscoveryBatch
PASS DiscoveryResult
PASS RequirementArtifact
PASS IngestionRun
PASS RunStep
```

Run transition validation has also been checked:

```text
requested -> validated
  PASS

requested -> completed
  correctly rejected
```

---

## 52. Updated Current Development Baseline

The repository is now a clean Phase 3 foundation when the following remain true:

```text
Configuration validation passes.
Runtime paths resolve under the project root.
Secrets remain outside Git.
Provider values are explicit enums.
Domain code remains infrastructure-free.
Domain models generate JSON Schema.
Invalid run transitions are rejected.
Ruff lint passes.
Ruff format check passes.
mypy strict passes.
No accidental .src/ duplicate folder exists.
No temporary test.md file is staged.
Phase 3 commit exists.
phase-3-complete tag exists.
```

---

## 53. Next Planned Phase

The next implementation phase is:

```text
Phase 4 — PostgreSQL canonical registry and migrations
```

Primary goal:

```text
Implement the canonical transactional persistence layer for projects, documents, document versions, ingestion runs, run steps, chunks, chunk manifests, facts, requirements, evidence, artifacts, and projection status.
```

Expected Phase 4 work:

```text
Add SQLAlchemy models.
Add Alembic migration environment.
Create initial PostgreSQL schema.
Create canonical repository interfaces and implementations.
Persist projects.
Persist documents.
Persist document versions.
Persist ingestion runs.
Persist run steps.
Persist chunk manifests.
Persist chunks.
Prepare fact tables.
Prepare requirement tables.
Prepare evidence tables.
Prepare artifact registry tables.
Prepare projection status tracking.
Add database health checks.
Add idempotent upsert behavior where needed.
Keep PostgreSQL as the completion-state source of truth.
```

Phase 4 must preserve these rules:

```text
PostgreSQL is canonical.
Neo4j is a rebuildable relationship projection.
ChromaDB is a rebuildable semantic retrieval projection.
Generated artifacts are filesystem outputs registered back in PostgreSQL.
The LLM never writes to databases.
The LLM never creates permanent IDs.
Domain code remains infrastructure-free.
```