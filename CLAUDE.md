# CLAUDE.md - Multi-Agentic Graph RAG

End-to-end test-cycle platform for embedded and monitoring-protocol devices.
Each stage is a standalone agent that consumes the previous stage's output
schema plus the knowledge stores.

Pipeline target:

`BRD/SRS ingest -> discover requirements -> user stories -> test scenarios -> test cases -> execute -> report`

Implemented today: ingestion, requirement discovery, user stories, test
scenarios, optional HFIL review for test scenarios, and reconcile. Test cases,
execution, and reporting are future work.

## Environment & Commands

- Python 3.12 (`>=3.12,<3.13`), managed by `uv`. Package: `multi_agentic_graph_rag` under `src/`. Windows-first; use `uv run`, not generated launchers.
- Install: `uv sync --dev --extra local-llm` for the Hugging Face stack, or add `--extra azure` for Azure OpenAI.
- CLI (`typer`, name `marag`, entry `python -m multi_agentic_graph_rag`): `version`, `config-check`, `hf-check [--load-model]`, `doctor`, `db-check`, `postgres-reset --yes`, `ingest`, `build-knowledge-graph --project <P> --document-version-id <DV>`, `generate-user-stories`, `generate-test-scenarios`, `coverage --project <P> [--document-version-id <DV>]`, `reconcile --project <P> [--document-version <DV>]`, `run status|resume <RUN-ID>`, `artifact verify <path>`, `artifact verify-user-stories <path>`, `artifact verify-test-scenarios <path>`.
- Ingest: `... ingest --project P --document PATH --version V [--logical-name N] [--replace-version] [--reasoning-provider ...] [--embedding-provider ...] --json-output`
- Optional HFIL: `... generate-test-scenarios ... --hfil [--thread-id ID] [--emit-md]`; `--no-hfil` forces the existing non-interactive path.
- Checks before commit: `uv run ruff check .`; `uv run python -m unittest discover -s tests`; `uv run mypy src/multi_agentic_graph_rag` (mypy strict); `uv run python -m compileall -q src`.
- See `README.md`, `huggingface-workflow.md`, and `azure-huggingface-workflow.md` for setup and end-to-end command walkthroughs; `schema.md` for artifact schemas.

## Storage Responsibilities

- **Neo4j** - graph knowledge base. Today it holds the document/chunk graph, validated user-story and test-scenario traceability claim-nodes linked back to evidence chunks, and the optional source-knowledge graph built by `build-knowledge-graph` (`TextUnit`, `Entity`, `EntityMention`, `Assertion`, `AssertionEvidence`). Nodes include `Project`, `Document`, `DocumentVersion`, `Chunk`, `Requirement`, `UserStory`, and `TestScenario`; rels include `OWNS_DOCUMENT`, `HAS_VERSION`, `HAS_CHUNK`, `COVERS_REQUIREMENT`, `VALIDATES_STORY`, `HAS_USER_STORY`, `HAS_TEST_SCENARIO`, and `EVIDENCED_BY_CHUNK`; knowledge-graph rels are `SUBJECT`, `OBJECT`, `SUPPORTED_BY`, `FROM_CHUNK`, `FROM_TEXT_UNIT`, `CONTAINS_TEXT_UNIT`, `NEXT_TEXT_UNIT`, `HAS_MENTION`, `REFERS_TO`, and `HAS_ASSERTION`. `Neo4jStore.project_artifact` remains a no-op because PostgreSQL is the generated-artifact ledger; the knowledge graph is extracted source semantics, not a projection of that ledger.
- **ChromaDB** - chunk embeddings, chunk text, and chunk/document metadata only, for semantic search.
- **PostgreSQL** - all generated AI content and ledger rows: canonical schema-5 requirement artifacts, relational requirement/revision/evidence and identity-resolution audit rows, user-story artifacts/rows, test-scenario artifacts/rows, and fallback retrieval when local JSON is absent or stale. Canonical requirement, revision, and evidence IDs are globally unique primary keys; a partial unique index enforces one active revision per requirement. The retired presentation-alias tables and columns are removed by migration `0005_contract_canonical_ids`. Schema self-validates against `_EXPECTED_COLUMNS`; mismatch is reported explicitly.
- Generated artifacts are committed to PostgreSQL first and then mirrored to local JSON. Valid local JSON is preferred on read only when metadata/payload checks match PostgreSQL; stale/missing JSON is repaired with `marag reconcile`.
- Each store has a `local_json` fallback mode (`runtime/staging/*.jsonl`) for dry runs; real ingest requires `postgres` and `neo4j` modes.

## Orchestration & Ingestion Pipeline

- `LangGraph` is the orchestrator. `workflows/ingestion_graph.py` compiles `validate_request -> run_pipeline -> END`. Add future stages as new agents/graphs following this pattern.
- `_run_pipeline` order: db checks -> ensure pg schema -> build models (reasoning + embedding + reranker) + warmup -> read/checksum doc -> parse document -> build manifest -> chunk blocks -> final manifest -> `assert_version_allowed` -> write manifest -> `neo4j.project_manifest` -> `chroma.index_chunks` -> acquire document identity lock -> load and seed prior active requirement memory -> `RequirementDiscoveryAgent.run` -> semantic identity reconciliation -> persist relational ledger and canonical `requirements.json` (`5.0-requirements`) -> write `identity_resolution.json` -> best-effort semantic KG build when enabled -> record run.
- Output dir (gitignored): `generated/<PROJECT>/req/<RUN_ID>/` containing `requirements.json`, `identity_resolution.json`, `run.log`, `run.jsonl`, `chunk_manifest.json`, and any saved `llm_response_*.txt` files.

## Models

- Ports: `llm_models/ports.py` - `ReasoningModel.generate_structured(prompt, schema)`, `EmbeddingModel.embed_documents`, `RerankerModel.rerank`. Built in `llm_models/factory.py`.
- Providers: reasoning and embedding = `huggingface` or `azure_openai`; reranker = `huggingface` only. Defaults: Qwen2.5-Coder-7B-Instruct, BGE-M3, bge-reranker-base.
- LLM calls happen at embeddings (Chroma index), requirement discovery (reasoning), user-story generation (reasoning), and test-scenario generation (reasoning). The reranker is wired into the shared hybrid retrieval path used by user-story and test-scenario generation.
- Config precedence: `config.json` -> `.env` -> defaults, loaded via `config/config_loader.py` into `config/settings.py::AppSettings` (`extra="forbid"`). Provider switches: `REASONING_MODEL_PROVIDER`, `EMBEDDING_MODEL_PROVIDER`, `RERANKER_MODEL_PROVIDER`.

## HFIL Scope

- HFIL is optional and scoped only to test-scenario review after permanent scenario IDs are assigned and before persistence.
- The HFIL node uses LangGraph `interrupt()`/resume with a checkpointer and no DB/JSON writes inside review turns. Finalize/persist is the only commit boundary.
- Semantic story matching uses calibrated embedding percentages (`hfil_cos_floor`/`hfil_cos_ceil`) because raw BGE cosine scores do not map directly to 0-100% business thresholds.
- Duplicate removal uses canonicalization, embedding recall, optional reranking, and deterministic LLM bidirectional entailment. The user must approve scenario IDs before deletion.
- MCP and broader HITL remain out of scope unless explicitly re-approved.

## Requirement Discovery

`agents/requirement_discovery_agent.py` loops per chunk, one strict rule-based prompt per chunk, and the LLM returns temp IDs (`F1`, `R1`). Validation chain: JSON -> Pydantic strict schemas -> regex meaningfulness checks -> source-trace verification. One retry with the validation error fed back; failure persists the raw response and raises `ModelOutputError`.

`services/requirement_builder.py` converts validated output into ledger records and assigns permanent deterministic IDs (`domain/identifiers.py`, sha256 `stable_token`):

- `chunk_id` remembered per requirement/fact via `source_trace`.
- `source_req_id` is preserved only when a normalized source identifier is present in the verified quote; otherwise it is nulled and `id_generation_type` remains `generated`.
- LLM output must include `confidence`; the builder carries it into `VerifiedRequirement`, PostgreSQL rows, and projected artifacts.
- Fact identity: `canonical_fact_id` (normalized text) + per-occurrence `fact_id`.
- Requirement identity is two-level for supersede tracking:
  - `requirement_id` is an allocated `REQ-<uuid7>` lineage ID, reused from cross-version memory only after semantic reconciliation.
  - `revision_id` is an allocated `REQREV-<uuid7>` ID; unchanged normalized text reuses it, while a mutable-parameter change creates a new revision in the same lineage.
  - `requirement_key` and `source_req_id` are hints/provenance only. Structured signatures preserve discriminators, retrieval scores only recall candidates, strict bidirectional entailment decides paraphrase merges, and ambiguity fails closed to a distinct lineage.
  - Evidence occurrences use `REQEVID-<uuid7>` and are retained when requirements deduplicate.
- Deltas: `_build_delta_events` compares against `prior_revisions` and emits `new`, `duplicate`, `changed`, or `superseded`; `impacted_artifact_types = [user_story, scenario, test_case]` for non-duplicates.

## Knowledge Graph (source semantics)

Optional stage after ingest: `marag build-knowledge-graph --project <P>
--document-version-id <DV>` (`workflows/knowledge_graph.py`).

- `agents/knowledge_extraction_agent.py` runs one strict prompt per chunk (same
  discipline as requirement discovery): entities + subject/predicate/object
  assertions, exact-quote grounding to `SourceTrace`, subject/object must match
  a declared entity, one retry with the validation error fed back, then
  `ModelOutputError` with the raw response persisted.
- `services/entity_resolution.py` resolves deterministically (exact normalized
  name -> alias -> acronym, never across entity types or projects) against the
  project's existing Neo4j entities, so re-runs and new versions reuse
  `entity_id`s. Embedding/reranker/LLM tie-break layers are future work.
- `services/text_unit_segmentation.py` derives atomic `TextUnit`s
  (sentence/bullet) deterministically from the ingested chunks at build time —
  no re-ingest needed; overlapping chunk windows dedupe by source-level span,
  and assertion evidence rows link to their overlapping units
  (`FROM_TEXT_UNIT`).
- `services/assertion_canonicalization.py` builds the project-scoped
  `assertion_key` (subject, predicate, object/literal, modality, polarity,
  condition; explicitness excluded — explicit wins over inferred on merge) and
  merges repeated occurrences into one document-version-scoped `assertion_id`
  with one evidence row per occurrence.
- The predicate is data on the `Assertion` node (normalized UPPER_SNAKE via
  `services/ontology.py`), so new predicates never require schema migrations.
  Structural rels only: `SUBJECT`/`OBJECT`/`SUPPORTED_BY`/`FROM_CHUNK`/
  `FROM_TEXT_UNIT`/`HAS_MENTION`/`REFERS_TO`/`HAS_ASSERTION`. Typed direct rels
  (materialization) are future work.
- Retrieval integration is flag-gated via `KnowledgeGraphSettings`
  (`knowledge_graph` config section / `KNOWLEDGE_GRAPH_ENABLED`,
  `KNOWLEDGE_GRAPH_SHADOW_MODE`, `GRAPH_PRIMARY_STORY`,
  `GRAPH_PRIMARY_SCENARIO`, `KNOWLEDGE_GRAPH_EXPANSION_K`; the checked-in profile enables graph-primary retrieval).
  When enabled, `RetrievalService` expands seed evidence chunks via
  `Neo4jStore.knowledge_related_chunks` (chunk -> assertion -> shared entity ->
  related assertion -> evidence chunk, version-scoped). Shadow mode leaves the
  generated context byte-identical and only records comparison snapshots to
  `generation_context_runs`/`generation_context_items`; the stage's
  graph-primary flag adds the candidates to the fusion pool where the reranker
  and the evidence-chunk guarantee still apply.
- Structured assertion retrieval (shadow): `services/knowledge_context.py`
  (`assemble_semantic_context`) builds a bounded, evidence-backed
  `SemanticContext` of `AssertionContextItem`s — mandatory source anchors
  (`Neo4jStore.fetch_anchor_assertions`), full-text seeds
  (`search_assertions_fulltext`), a degree-bounded one-hop entity expansion
  (`expand_entity_assertions`, predicate-profiled per story/scenario stage),
  scored by predicate priority + confidence + explicitness − hop decay with
  anchors never budget-dropped, then hydrated to exact TextUnits
  (`hydrate_assertion_evidence`). In shadow/primary mode `RetrievalService`
  records this as a second `graph_semantic` snapshot (schema §18 columns on
  `generation_context_items`: `assertion_id`/`text_unit_id`/`entity_id`/
  `predicate`/`hop_count`/`normalized_score`/`mandatory`/`metadata_json`)
  alongside the legacy chunk snapshot.
- Story cutover (graph-primary): with `GRAPH_PRIMARY_STORY` on,
  `RetrievalService.assemble_primary_context` runs the assembler in loop 1 and,
  when the grounding gate passes (mandatory anchor present + at least
  `graph_min_assertions` selected), freezes the selected `AssertionContextItem`s
  into the story `context_map` (`context_mode="graph_primary"`) so the structured
  context is auditable and reproduced identically on resume. The user-story
  prompt then renders an "AUTHORITATIVE FACTS FOR THIS REQUIREMENT" section
  (mandatory/hop-0) separately from a disambiguation-only "RELATED CONTEXT"
  section (hop-1), with chunk text kept as supplementary excerpts. Any gate miss
  or retrieval error falls back to the legacy chunk path, so the cutover is
  independently reversible via the flag. A post-generation claim-grounding
  validator is deferred (unverifiable without live services); the pre-generation
  gate plus the authoritative/related prompt split are the grounding controls.
- Scenario cutover (graph-primary): symmetric to the story cutover behind
  `GRAPH_PRIMARY_SCENARIO`. Loop 1 assembles with the scenario predicate profile
  (`config_for_stage` selects `SCENARIO_PREDICATES` for stage `test_scenario`)
  using the story query (`_story_query_text`: title + persona + acceptance
  criteria + linked requirement) and the linked requirement's evidence chunks as
  anchors, freezes the selection into the scenario `context_map`, and the
  scenario prompt renders "AUTHORITATIVE FACTS FOR THIS STORY" (grounds expected
  results) vs. "RELATED CONSTRAINTS AND CONDITIONS" (test dimensions to probe).
  The assertion-line rendering and authoritative/related split are shared via
  `services/knowledge_context.py` (`render_assertion_lines`,
  `authoritative_related`). Both cutovers stay off by default and independently
  reversible.
- Audit copy: `generated/<PROJECT>/kg/<RUN_ID>/knowledge_graph.json`
  (`KnowledgeGraphArtifact`, schema `1.0-knowledge-graph`); the run is recorded
  in PostgreSQL `ingestion_runs`. Neo4j is the source of truth for this graph.

## Schemas / Contracts

All runtime schemas live in `domain/schemas.py` as `StrictModel` extra=forbid.

- Ingestion: `IngestionRequest`, `ParsedBlock`, `DocumentChunk`, `DocumentManifest`, `IngestionResult`.
- LLM I/O: `RequirementDiscoveryChunkOutput` -> normalized to `RequirementDiscoveryOutput`.
- Ledger: `CanonicalFact`, `VerifiedFact`, `VerifiedRequirement`, `RequirementEvidence`, `RequirementDeltaEvent`, `RequirementRevisionSnapshot`.
- Requirement artifacts:
  - `RequirementArtifact` (schema `2.1`) is an internal build/persistence model and is not emitted publicly.
  - `CanonicalRequirementsArtifact` (`5.0-requirements`) -> the sole public `requirements.json`, one row per `(requirement_id, revision_id)` with nested evidence occurrences.
  - `RequirementIdentityResolutionArtifact` -> `identity_resolution.json`, the auditable resolution decisions for the run.
  - Old occurrence catalogs are accepted only by `requirements repair-identities`; normal generation readers require schema 5.0.
- Public projected artifacts:
  - `UserStoryArtifact` exposes `story_id`, `requirement_id`, flat `acceptance_criteria`, `source_req_id`, `confidence`, and traceability rows.
  - `TestScenarioArtifact` exposes `scenario_id`, `story_id`, `requirement_id`, `source_req_id`, `confidence`, and traceability rows.
- Runtime rows (`VerifiedRequirement`, `UserStoryRecord`, `TestScenarioRecord`) keep canonical IDs, parent lineage/revision, evidence chunks, status/origin, and PostgreSQL/Neo4j compatibility. `TestScenarioRecord.origin` may be `generation` or `feedback`; feedback records are created only by optional stage-4 HFIL before final persistence.
- Display aliases are allocated by `services/id_registry.py`: `REQ-001`, `US-001`, and `TS-001` are stable per project while internal hash IDs remain the system identities.

## Adding the Next Agent

1. Define strict input/output Pydantic schemas in `domain/schemas.py`.
2. Add deterministic IDs to `domain/identifiers.py`; add pg tables and `_EXPECTED_COLUMNS` entries if persisting.
3. Implement the agent under `agents/`, provider-neutral via `llm_models` ports.
4. Add a LangGraph workflow under `workflows/` and a CLI command in `cli.py` wrapped in `command_session`.
5. Write outputs to `generated/<PROJECT>/<stage>/<RUN_ID>/` and persist to PostgreSQL for traceability.
6. Reuse `RetrievalService` for downstream context: Chroma semantic search, Neo4j full-text search, Neo4j multi-hop over chunks, and reranker fusion.

## Design Decisions

- Chunks are immutable source truth. Requirements are validated generated claims. User stories and test scenarios are downstream generated claims linked back to source evidence.
- Neo4j is not the primary generated-artifact store. It holds validated traceability nodes for user stories and test scenarios so later stages can retrieve graph context for content-quality validation.
- PostgreSQL remains the generated-artifact ledger and source of truth for content plus status.
- Coverage, resume, HFIL, dedup, PostgreSQL, Neo4j, and public JSON all use canonical IDs.
- Artifact contract: `requirements.json`, `user_stories.json`, and `test_scenarios.json` are canonical public artifacts; requirement resolution decisions are emitted separately in `identity_resolution.json`.
- Version continuity: users should keep `--logical-name` / `document_id` stable across versions so supersede tracking fires.
- Future stages: test cases, execution, and reporting remain future work. The deterministic `canonical_fact_id` / lineage ID scheme already supports the idempotency those stages will need.
