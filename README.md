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