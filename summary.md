# Change Summary — Coverage, Traceability & Cleanups

This change closes the psql "master-sheet" traceability gap (every user story →
requirement + chunk ids; every test scenario → story + requirement + chunk ids),
adds coverage reporting and run provenance, freezes requirements against
interactive revision, and fixes three defects (checkpoint filename collision,
temp-id prompt scaffolding, and a noisy Neo4j warning).

All changes were validated against a leaner alternative to a ChatGPT-proposed
plan: the generic `coverage_links`, `generated_artifact_fingerprints`, and
`generation_runs` tables were **rejected** as redundant with the existing
single-parent relational columns / deterministic IDs / artifact tables. See
"Design decisions" at the end.

**Gates after the change:** `ruff` clean · `mypy --strict` clean (56 files) ·
**143 unit tests pass** · `compileall` clean.

---

## 1. Chunk-level traceability for stories & scenarios (Problems 1, 3, 4, 5, 6)

### Root cause fixed
Full generation never populated `evidence_chunk_ids` on the persisted records —
chunk provenance existed only for the feedback path. `user_stories` /
`test_scenarios` also had no first-class `chunk_id` linkage in psql (only inside
the JSONB payload).

### What changed
- **`src/multi_agentic_graph_rag/services/user_story_builder.py`** — `_to_record`
  now sets `evidence_chunk_ids=list(requirement.evidence_chunk_ids)`. *Why:* a
  story now inherits its parent requirement's source chunks, so the chunk →
  requirement → story chain is complete on the normal generation path (not just
  feedback).
- **`src/multi_agentic_graph_rag/services/test_scenario_builder.py`** — `_to_record`
  now sets `evidence_chunk_ids=list(story.evidence_chunk_ids)`. *Why:* a scenario
  inherits its parent story's chunks, completing chunk → requirement → story →
  scenario lineage.
- **`src/multi_agentic_graph_rag/db/postgres.py`** — two new managed tables
  mirroring the existing `requirement_evidence` pattern:
  - `user_story_evidence` (`evidence_id` PK, `story_id`, `requirement_id`,
    `document_version_id`, `chunk_id`, `run_id`, `created_at`) → **Problem 6**.
  - `test_scenario_evidence` (`evidence_id` PK, `scenario_id`, `story_id`,
    `requirement_id`, `document_version_id`, `chunk_id`, `run_id`, `created_at`)
    → **Problem 5** (full lineage denormalized into one row for single-query
    traceback).
  - Registered in `_MANAGED_TABLES` (child-first drop order), `_EXPECTED_COLUMNS`
    (schema self-validation), `ensure_schema` DDL, and new indexes on
    `(story_id)`/`(scenario_id)`, `(chunk_id)`, `(document_version_id)`.
  - `_persist_user_stories_postgres` / `_persist_test_scenarios_postgres` (and the
    `local_json` twins) now insert one evidence row per `evidence_chunk_id` with
    `on conflict (evidence_id) do nothing`. *Why:* first-class, queryable chunk
    linkage; idempotent re-persist. Feedback additions flow through the same
    persist path, so **Problem 2 (incremental additions)** gets chunk linkage for
    free.
- **`src/multi_agentic_graph_rag/domain/identifiers.py`** — added
  `user_story_evidence_id(...)` and `test_scenario_evidence_id(...)` (deterministic
  `stable_token` over ids + document_version + chunk), prefixes `USEVID-` /
  `SCEVID-`. *Why:* deterministic evidence ids make re-persist a clean no-op.
- **`common_defs.py`** — added `IdentifierPrefix.USEVID` / `.SCEVID`.
- **`src/multi_agentic_graph_rag/services/requirement_source.py`** — *(verified,
  unchanged)* already fills `RequirementInput.evidence_chunk_ids` from each compact
  occurrence's `chunk_id` (and from source-trace chunks in the full-payload path),
  so the real `requirements.json` load path feeds the chain end to end.

---

## 2. Run provenance + `origin` (Problem 3 — psql as master sheet)

- **`src/multi_agentic_graph_rag/db/postgres.py`** — added `run_id` and `origin`
  columns to `user_stories` and `test_scenarios` (DDL + `_EXPECTED_COLUMNS`), and
  carry them into the row INSERTs (`run_id` was already passed into the persist
  methods; `origin` comes from the record, `generation` vs `feedback`).
  *Why:* answer "which run created this?" and "was this generated or added via
  feedback?" without a separate `generation_runs` table.
  > **Note:** `origin` is an addition beyond the plan's `run_id`-only scope —
  > cheap, tested, and useful for auditing feedback additions.

---

## 3. Coverage report query + CLI (Problem 1 — keep coverage)

- **`src/multi_agentic_graph_rag/db/postgres.py`** — `load_coverage_status(*, project,
  document_version_id=None)` + module-level pure helper `_compute_coverage(...)`.
  Computes per-requirement status `no_story` / `story_covered` /
  `scenario_covered` from existing relational columns (no coverage table). Works
  in both `postgres` and `local_json` modes.
- **`src/multi_agentic_graph_rag/cli.py`** — new `marag coverage --project <P>
  [--document-version-id <DV>] [--json-output]` command (read-only, wrapped in
  `command_session`), renders a Rich table.
- **`common_defs.py`** — added `RuntimeCommand.COVERAGE`.

---

## 4. Requirements stay immutable (Problem 7)

- Confirmed & locked at the schema level: `FeedbackRequest.stage` is
  `Literal["user_story", "test_scenario"]` in
  `src/multi_agentic_graph_rag/domain/schemas.py`, so feedback/incremental flows
  can **never** target requirements. Ingestion-time supersede (v1→v2→v3 for a new
  document version) is intentionally left untouched — that is document-version
  evolution, not interactive revision.
- Guarded by a test (`RequirementsFreezeTests`).

---

## 5. Checkpoint filename collision fix (Problem 8)

Both stages wrote `context_map.json` (+ `generation_progress.json` /
`generation_errors.jsonl`) into the **same** output directory during a full
pipeline, so the scenario stage tripped a `CheckpointError` on the story stage's
file.

- **`src/multi_agentic_graph_rag/services/generation_checkpoint.py`** — replaced
  the shared filename constants with **stage-derived** filenames:
  - `context_story.json` / `context_scenario.json`
  - `progress_story.json` / `progress_scenario.json`
  - `errors_story.jsonl` / `errors_scenario.jsonl`
  - via `context_map_filename(stage)` / `generation_progress_filename(stage)` /
    `generation_errors_filename(stage)` (backed by `_stage_suffix`).
  - `append_generation_error` now takes an explicit `stage` argument.
  *Why:* the two extra files collided identically and would have been the next
  bug; fixing all three removes the class of defect.
- **`workflows/user_story_graph.py`** & **`workflows/test_scenario_graph.py`** —
  updated imports, log-path lines, and `append_generation_error(out_dir, stage, ...)`
  calls.

---

## 6. Remove temp-id scaffolding from prompts & schemas (Problem 9)

The LLM was asked to emit throwaway ids (`F1`/`R1`, `US1`/`AC1`/`BR1`/`TS1`,
`SC1`) that Python immediately discarded and replaced with permanent ids.
Relationships are already expressed by list containment, and the existing schemas
are `list[...]` with `min_length=1`, so **one-to-many was already supported** —
the change is pure deletion, no count-branching logic added.

- **`src/multi_agentic_graph_rag/common_prompt_defs.py`** — trimmed the temp-id
  fields and instructions from all three prompts (requirement discovery, user
  story, test scenario); replaced with "Do not return any id fields … Python
  assigns all permanent ids from list position."
- **`src/multi_agentic_graph_rag/domain/schemas.py`** — removed the LLM-facing id
  fields: `LLMDiscoveredFact.fact_id`, `LLMDiscoveredRequirement.req_id`,
  `UserStoryModel.story_id`, `TestScenarioModel.scenario_id`. The nested
  `AcceptanceCriterion.id` / `BusinessRule.id` / `TestScenario.id` fields are
  **kept** (the builder assigns permanent `AC-001`/`BR-001`/`TS-001`).
- **`src/multi_agentic_graph_rag/agents/requirement_discovery_agent.py`** — since
  the model no longer returns ids, internal temp labels (`F{n}` / `R{n}`, used
  only in validation error messages) are synthesized by ordinal in
  `_fact_to_candidate` / `_source_trace_from_quote`.
  *Why:* smaller prompts, fewer LLM failure modes (no malformed/duplicate temp
  ids), same permanent-id guarantees.
- **Tests updated** to drop the removed keys from stub LLM outputs
  (`test_requirement_discovery_agent.py`, `test_user_story_agent.py`,
  `test_test_scenario_agent.py`, `test_user_story_workflow.py`,
  `test_test_scenario_workflow.py`, `test_generation_two_loop.py`,
  `test_user_story_builder.py`, `test_test_scenario_builder.py`).

---

## 7. Silence the Neo4j PDF `section` warning (Problem 10)

PDF chunks never get a `section` property (only Markdown/DOCX headings set it), so
the multi-hop `neighbor_chunks` query referencing `c.section` emitted a spurious
`UnknownPropertyKey` notification.

- **`src/multi_agentic_graph_rag/db/neo4j_store.py`** — `_driver()` now passes
  `notifications_disabled_classifications=[NotificationDisabledClassification.UNRECOGNIZED]`
  (verified against the installed `neo4j==6.2.0`). *Why:* surgically suppresses
  only the unknown-property notification; genuine schema/performance notifications
  still surface. We deliberately did **not** write `section=""` for PDFs (an empty
  string is not null and would break neighbor-matching semantics).
  > Mechanism-verified (driver accepts the kwarg; `UNRECOGNIZED` is the correct
  > classification). Effect ("no warning in run.log") should be confirmed on the
  > live PDF pipeline run.

---

## 8. Tests added

- **`tests/test_story_scenario_evidence.py`** (new, 9 tests):
  - builder threads `evidence_chunk_ids` for stories and scenarios;
  - the real `requirements.json` loader → builder → persist path produces
    non-empty `user_story_evidence` rows (guards the linchpin);
  - evidence rows persist with full lineage and are idempotent on re-persist;
  - deterministic evidence ids (`USEVID-`/`SCEVID-`);
  - coverage status `no_story` / `story_covered` / `scenario_covered`;
  - P7 requirements-freeze (feedback stage cannot be `requirement`).
- **`tests/test_generation_checkpoint.py`** — stage-specific filename usage;
  repurposed the old stage-mismatch test into a stage-filename-separation test
  plus an identity-mismatch (different project) guard test.
- Pre-existing `E501` long lines in touched/adjacent test files were wrapped to
  keep the suite ruff-clean.

---

## 9. Docs kept accurate

- **`CLAUDE.md`** — added `user_story_evidence` / `test_scenario_evidence` to the
  managed-tables list and the `coverage` command to the CLI list.

---

## Design decisions (validation of the ChatGPT plan)

| Proposal | Decision | Reason |
|---|---|---|
| `coverage_links` table | **Rejected** | Model is strictly single-parent + add-only. `user_stories.requirement_id` and `test_scenarios.story_id`+`requirement_id` already are the edges. A separate edge table = second source of truth. Coverage status is a query, not a table. |
| `generated_artifact_fingerprints` | **Rejected/deferred** | No semantic-dedup requirement; a normalized-title hash isn't semantic dedup (that needs embeddings via existing Chroma). Feedback already dedupes via deterministic `feedback_id` + the gate's duplicate-decline. |
| `generation_runs` table | **Downgraded** | `ingestion_runs` + per-run dirs + `*_artifacts` already give run provenance. Added a cheap `run_id` column instead of a new table. |
| **The real gap** | **Adopted** | First-class chunk linkage for stories/scenarios (evidence tables) + populating `evidence_chunk_ids` on the full-generation path. |

---

## Migration & verification

1. New tables/columns trip schema self-validation → run once against your
   disposable/local DB:
   ```powershell
   uv run marag postgres-reset --yes
   uv run marag db-check
   ```
   (wipes generated data; re-ingest regenerates it.)
2. Full pipeline on a **PDF** proves Problem 8 (no checkpoint collision) and
   Problem 10 (no `section` warning in `run.log`).
3. Master-sheet checks (Problems 3–6):
   ```sql
   select story_id, requirement_id, run_id, origin from user_stories;
   select story_id, chunk_id, requirement_id from user_story_evidence;      -- non-empty
   select scenario_id, story_id, requirement_id, chunk_id from test_scenario_evidence; -- non-empty
   ```
4. `uv run marag coverage --project <P>` → per-requirement rollup.
5. Gates: `uv run ruff check .` · `uv run mypy src/multi_agentic_graph_rag` ·
   `uv run python -m unittest discover -s tests` · `uv run python -m compileall -q src`.
