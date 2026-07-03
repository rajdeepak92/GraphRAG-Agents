# CLAUDE.md — Multi-Agentic Graph RAG

End-to-end product-quality test-cycle platform for embedded/monitoring-protocol
devices. Each major stage is a **standalone agent** consuming the previous
stage's output schema plus the knowledge stores. Pipeline target:

`BRD/SRS ingest → discover requirements → user stories → test scenarios → test cases → execute → report`

**Implemented today:** ingestion → requirement discovery → user stories → test
scenarios. Test cases, execution, report, and the re-discover-with-feedback
command are **planned, not built**.

## Environment & Commands

- Python 3.12 (`>=3.12,<3.13`), managed by **uv**. Package: `multi_agentic_graph_rag` under `src/`. Windows-first; use `uv run`, not generated launchers.
- Install: `uv sync --dev --extra local-llm` (HF) or add `--extra azure` (Azure OpenAI).
- CLI (`typer`, name `marag`, entry `python -m multi_agentic_graph_rag`):
  `version`, `config-check`, `hf-check [--load-model]`, `doctor`, `db-check`,
  `postgres-reset --yes`, `ingest`, `generate-user-stories`,
  `generate-test-scenarios`, `run status|resume <RUN-ID>`,
  `artifact verify <path>`.
- Ingest: `... ingest --project P --document PATH --version V [--logical-name N] [--replace-version] [--reasoning-provider ...] [--embedding-provider ...] --json-output`
- Checks before commit: `uv run ruff check .`; `uv run python -m unittest discover -s tests`; `uv run mypy src/multi_agentic_graph_rag` (mypy **strict**); `uv run python -m compileall -q src`.
- See `README.md` / `RUNBOOK.md` for setup; `commands.md` for command reference.

## Storage Responsibilities (do not blur these)

- **Neo4j** — graph knowledge base. **Today:** document/chunk graph plus
  validated user-story and test-scenario traceability claim-nodes linked back to
  their evidence chunks. Nodes include `Project`, `Document`, `DocumentVersion`,
  `Chunk`, `Requirement`, `UserStory`, and `TestScenario`; rels include
  `OWNS_DOCUMENT`, `HAS_VERSION`, `HAS_CHUNK`, `COVERS_REQUIREMENT`,
  `VALIDATES_STORY`, `HAS_USER_STORY`, `HAS_TEST_SCENARIO`, and
  `EVIDENCED_BY_CHUNK`. `Neo4jStore.project_artifact` remains a no-op because
  PostgreSQL is the generated-artifact ledger.
- **ChromaDB** — chunk **embeddings + chunk text + chunk/doc metadata only**, for semantic search. Collection `marag_chunks`, persisted at `runtime/databases/chroma`.
- **PostgreSQL** — all **generated AI content + ledger**:
  `requirements_full.json` payload, requirement ledger tables, user-story
  artifacts/rows, test-scenario artifacts/rows, and fallback retrieval when
  local JSON is absent. Managed tables in `db/postgres.py::_MANAGED_TABLES`
  include `test_scenarios`, `test_scenario_artifacts`, `user_stories`,
  `user_story_artifacts`, `canonical_facts`, `fact_occurrences`,
  `requirements`, `requirement_revisions`, `requirement_evidence`,
  `requirement_fact_links`, `requirement_delta_events`,
  `requirement_artifacts`, `document_versions`, and `ingestion_runs`. Schema
  self-validates against `_EXPECTED_COLUMNS`; mismatch → `postgres-reset`.
- Each store has a `local_json` fallback mode (`runtime/staging/*.jsonl`) for dry runs; real ingest requires `postgres`/`neo4j` modes (enforced in `_validate_required_ingest_stack`).

## Orchestration & Ingestion Pipeline

- **LangGraph** is the brain. `workflows/ingestion_graph.py` compiles: `validate_request → run_pipeline → END`. Add new stages as new agents/graphs following this pattern.
- `_run_pipeline` order: db checks → ensure pg schema → build models (reasoning+embedding+reranker) + warmup → read/checksum doc → `parse_document` → `build_manifest` (preliminary) → `chunk_blocks` → final manifest → `assert_version_allowed` → write manifest → `neo4j.project_manifest` (chunks) → `chroma.index_chunks` (embeddings) → `RequirementDiscoveryAgent.run` → load prior revisions from pg → `build_requirement_artifact` → write `requirements_full.json` + compact `requirements.json` → `persist_manifest` + `persist_artifact` (ledger) → `record_run`.
- Output dir (gitignored): `generated/<PROJECT>/req/<RUN_ID>/` containing `requirements.json`, `requirements_full.json`, `run.log`, `run.jsonl`, `chunk_manifest.json`, and `llm_response_*.txt` (saved on parse/trace failure, or always when `LOG_LLM_RESPONSES=true`).

## Models (provider-neutral ports + factory)

- Ports: `llm_models/ports.py` — `ReasoningModel.generate_structured(prompt, schema)`, `EmbeddingModel.embed_documents`, `RerankerModel.rerank`. Built in `llm_models/factory.py`.
- Providers: reasoning + embedding = `huggingface` **or** `azure_openai`; reranker = `huggingface` only. Defaults: Qwen2.5-Coder-7B-Instruct, BGE-M3, bge-reranker-base.
- LLM calls happen at: **embeddings** (Chroma index), **requirement discovery**
  (reasoning), **user-story generation** (reasoning), and **test-scenario
  generation** (reasoning). The reranker is wired into the shared hybrid
  retrieval path used by user-story and test-scenario generation.
- Config precedence: `config.json` → `.env` → defaults, loaded via `config/config_loader.py` into `config/settings.py::AppSettings` (`extra="forbid"`). Provider switches: `REASONING_MODEL_PROVIDER`, `EMBEDDING_MODEL_PROVIDER`, `RERANKER_MODEL_PROVIDER`.

## Requirement Discovery (reference agent to copy)

`agents/requirement_discovery_agent.py`: loops **per chunk**, one strict rule-based prompt per chunk, LLM returns temp IDs (`F1`,`R1`). Validation chain: JSON → Pydantic strict schemas → regex meaningfulness checks → **source-trace verification** (the `quote` must be locatable in the normalized chunk and map back to exact raw offsets). One retry with the validation error fed back; failure persists raw response and raises `ModelOutputError`.

`services/requirement_builder.py` converts validated output into ledger records and assigns **permanent deterministic IDs** (`domain/identifiers.py`, sha256 `stable_token`):
- `chunk_id` remembered per requirement/fact via `source_trace`.
- Fact identity: `canonical_fact_id` (normalized text) + per-occurrence `fact_id`.
- Requirement identity is **two-level for supersede tracking**:
  - `requirement_id` = **lineage** = `stable_token(project, document_id, requirement_key)` — stable across versions.
  - `revision_id` = `stable_token(lineage, normalized_statement)` — changes when wording/value changes.
  - `requirement_key` (`derive_requirement_key`) masks numbers/temperatures/quoted values, so threshold 70→80 keeps the **same lineage** but a **new revision**, and the old revision is marked `superseded` (implements the v1→v2→v3 supersede chain).
- Deltas: `_build_delta_events` compares against `prior_revisions` (pg `load_requirement_revision_snapshot`) → emits `new` / `duplicate` / `changed` / `superseded`; `impacted_artifact_types = [user_story, scenario, test_case]` for non-duplicates.

## Schemas / Contracts (`domain/schemas.py`, all `StrictModel` extra=forbid)

- Ingestion: `IngestionRequest`, `ParsedBlock`, `DocumentChunk`, `DocumentManifest`, `IngestionResult`.
- LLM I/O: `RequirementDiscoveryChunkOutput` (model returns) → normalized to `RequirementDiscoveryOutput`.
- Ledger: `CanonicalFact`, `VerifiedFact`, `VerifiedRequirement`, `RequirementEvidence`, `RequirementDeltaEvent`, `RequirementRevisionSnapshot`.
- **Downstream input contract** = the two written artifacts:
  - `RequirementArtifact` (schema `2.0`) → `requirements_full.json` (audit/traceability, full evidence + delta events).
  - `CompactRequirementArtifact` (schema `3.0-compact`) → `requirements.json`: `requirements: {requirement_id: [CompactRequirementOccurrence{chunk_id, fact_id, requirement_text, requirement_type, priority(High/Medium/Low), status(Active/Superseded), doc_version}]}`. Scoped to **requirement-text retrieval/traceback**; the compact feed the **UserStory agent** reads.
  - **Planned `run.json`** — exact duplicate of the psql ledger rows (statuses/revisions/delta/evidence). Downstream reads `run.json` first, psql as trust anchor. Keep ledger status **out of** `requirements.json`.

## Adding the Next Agent (testcase / execution / report / ...)

1. Define strict input/output Pydantic schemas in `domain/schemas.py` (e.g. `UserStory`, `UserStoryArtifact`). Input = the prior stage's artifact schema.
2. Add deterministic IDs to `domain/identifiers.py`; add pg tables + `_EXPECTED_COLUMNS` entries if persisting.
3. Implement the agent under `agents/`, provider-neutral via `llm_models` ports. Read input from local JSON first, fall back to psql query (performance-first local preference).
4. Add a LangGraph workflow under `workflows/` (mirror `ingestion_graph.py`) and a CLI command in `cli.py` wrapped in `command_session`.
5. Write outputs to `generated/<PROJECT>/<stage>/<RUN_ID>/` + persist to psql for traceability. Add tests under `tests/`.
6. Reuse `RetrievalService` for downstream context: Chroma semantic search +
   Neo4j full-text + Neo4j multi-hop over chunks + reranker fusion.

## Design Decisions (resolved) & Open Items

**Data-layer model (decided):**
- Chunks = **immutable source truth**. Requirements = **validated generated claims**. User stories = **downstream generated claims**. Graph relationships connect claims back to source evidence.
- **Neo4j is NOT only a source-document graph.** It also holds **validated
  generated traceability nodes** for user stories and test scenarios linked back
  to their evidence chunks, so a stage can retrieve graph context for
  **content-quality validation**. Neo4j is **not** the primary generated-artifact
  store.
- **PostgreSQL remains the generated-artifact ledger** (source of truth for content + status).
- **"User stories validated by Neo4j"** means: at generation time, retrieve the relevant Neo4j context for content-quality validation — nothing more. **Coverage/dedup** ("is this requirement already covered by a previous run?") is answered from `requirements.json` / psql (**local `.json` first**, psql fallback), not Neo4j.

> **OPEN ITEM:** test-case generation, execution, reporting, and
> re-discover-with-feedback are still future stages. `Neo4jStore.project_artifact`
> remains a no-op by design; downstream claim-node projections are handled by
> their stage-specific methods.

**Artifact split (decided):** keep `requirements.json` scoped to **requirement-text retrieval/traceback** only. Add a separate **`run.json`** that is an **exact duplicate of the psql ledger rows** (statuses, revisions, delta events, evidence). Reading order for downstream agents: **`run.json` first** (local, exact mirror), psql as the trust anchor / fallback. Do not overload `requirements.json` with ledger status.

**Version continuity (decided):** no code enforcement for now. Convention: users name files `project_brd_v1.pdf`, `project_brd_v2.pdf`, … so `--logical-name`/`document_id` stays stable across versions and supersede tracking fires. (Delta still silently depends on a stable `document_id`; revisit enforcement later.)

**Re-discover-with-feedback (deferred):** planning phase only, do not implement now. The deterministic `canonical_fact_id`/lineage ID scheme already supports the idempotency it will need.

**Consistency:** otherwise the design is internally consistent — standalone agents, strict schema hand-off, deterministic idempotent IDs, clean store separation, LLM output always validated (regex + Pydantic + source-trace). The ledger/revision model correctly implements the v1→v2→v3 supersede requirement.
