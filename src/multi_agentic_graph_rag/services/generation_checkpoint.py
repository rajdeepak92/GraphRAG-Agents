"""Local crash-resume checkpoints for the two-loop generation stages.

The user-story and test-scenario stages run as two loops:

* **Loop 1** retrieves context *identifiers* per input and checkpoints them to
  ``context_map.json`` (ids + retrieval metadata only, never chunk text).
* **Loop 2** hydrates full chunk text, feeds the LLM one item at a time, and
  records per-item progress to ``generation_progress.json`` after each success.

Checkpoints are local files under the existing generated run directory. None of
this is persisted to PostgreSQL, and no new SQL table is introduced. All JSON is
written atomically (temp file + ``os.replace``) so a crash never leaves a
partially written canonical checkpoint.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field, ValidationError

from multi_agentic_graph_rag.domain.errors import CheckpointError
from multi_agentic_graph_rag.domain.schemas import (
    AssertionContextItem,
    GenerationTrace,
    SemanticContext,
    StrictModel,
)
from multi_agentic_graph_rag.observability.logging import RunLogger
from multi_agentic_graph_rag.services.retrieval import (
    ChunkProvenance,
    RetrievedChunk,
    RetrievedContext,
)

# Context-map ``context_mode`` values: legacy chunk retrieval vs. the frozen
# graph-primary structured assertion selection (plan Increment 3).
CONTEXT_MODE_CHUNKS = "chunks"
CONTEXT_MODE_GRAPH_PRIMARY = "graph_primary"

STAGE_USER_STORY = "user_story"
STAGE_TEST_SCENARIO = "test_scenario"

# Stage-scoped checkpoint filenames. The user-story and test-scenario stages can
# share one output directory (full pipeline), so every checkpoint file is named per
# stage to prevent cross-stage collisions (e.g. context_story.json vs
# context_scenario.json).
_STAGE_FILE_SUFFIX = {
    STAGE_USER_STORY: "story",
    STAGE_TEST_SCENARIO: "scenario",
}


def _stage_suffix(stage: str) -> str:
    try:
        return _STAGE_FILE_SUFFIX[stage]
    except KeyError as exc:
        raise CheckpointError(f"unknown checkpoint stage {stage!r}") from exc


def context_map_filename(stage: str) -> str:
    return f"context_{_stage_suffix(stage)}.json"


def generation_progress_filename(stage: str) -> str:
    return f"progress_{_stage_suffix(stage)}.json"


def generation_errors_filename(stage: str) -> str:
    return f"errors_{_stage_suffix(stage)}.jsonl"


_EVIDENCE_SOURCE = "evidence"


class ChunkFetcher(Protocol):
    """Minimal read surface used to hydrate chunk ids back into full text."""

    def fetch_chunks(self, chunk_ids: list[str]) -> list[tuple[str, str]]: ...


class RetrievalMetadataEntry(StrictModel):
    chunk_id: str
    source: str
    rank: int
    score: float | None = None


class ContextMapEntry(StrictModel):
    """One retrieved-context record.

    ``story_id`` is populated only for the test-scenario stage; it is omitted
    from the serialized user-story context map (see :func:`_dump_entry`).
    """

    story_id: str | None = None
    requirement_id: str
    document_version_id: str
    chunk_ids: list[str] = Field(default_factory=list)
    retrieval_metadata: list[RetrievalMetadataEntry] = Field(default_factory=list)
    context_mode: str = CONTEXT_MODE_CHUNKS
    assertions: list[AssertionContextItem] = Field(default_factory=list)
    generation_context_run_id: str = ""


class ContextMapCheckpoint(StrictModel):
    run_id: str
    stage: str
    project: str
    document_version_id: str
    entries: list[ContextMapEntry] = Field(default_factory=list)


class GenerationProgressItem(StrictModel):
    """A single completed/failed input.

    ``requirement_id`` is populated only for the test-scenario stage; it is
    omitted from the serialized user-story progress (see :func:`_dump_item`).
    """

    input_id: str
    requirement_id: str | None = None
    status: str
    completed_at: str
    payload: dict[str, Any] = Field(default_factory=dict)


class GenerationProgress(StrictModel):
    run_id: str
    stage: str
    project: str
    document_version_id: str
    completed: dict[str, GenerationProgressItem] = Field(default_factory=dict)
    failed: dict[str, GenerationProgressItem] = Field(default_factory=dict)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as deterministic, readable JSON via a temp-file swap.

    The temp file is created in the destination directory and swapped into place
    with :func:`os.replace`, which is atomic on the same filesystem, so the
    canonical checkpoint file is never observed half-written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.stem + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
            handle.write("\n")
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            with contextlib.suppress(OSError):
                temp_path.unlink()
        raise


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_context_map_entry(
    *,
    requirement_id: str,
    document_version_id: str,
    provenance: Sequence[ChunkProvenance],
    story_id: str | None = None,
    semantic: SemanticContext | None = None,
    generation_context_run_id: str = "",
) -> ContextMapEntry:
    """Build one loop-1 context-map entry from retrieval provenance (no text).

    When ``semantic`` is supplied (graph-primary), the selected assertions are
    frozen into the entry alongside the chunk provenance, so the structured
    context that feeds generation is recorded and reproduced identically on
    resume — mirroring how chunk selection is frozen. ``generation_context_run_id``
    is the single id shared with the recorded semantic snapshot and later
    persisted on the generated record.
    """
    metadata = [
        RetrievalMetadataEntry(
            chunk_id=item.chunk_id,
            source=item.source,
            rank=item.rank,
            score=item.score,
        )
        for item in provenance
    ]
    return ContextMapEntry(
        story_id=story_id,
        requirement_id=requirement_id,
        document_version_id=document_version_id,
        chunk_ids=[item.chunk_id for item in provenance],
        retrieval_metadata=metadata,
        context_mode=(CONTEXT_MODE_GRAPH_PRIMARY if semantic is not None else CONTEXT_MODE_CHUNKS),
        assertions=list(semantic.items) if semantic is not None else [],
        generation_context_run_id=generation_context_run_id,
    )


def trace_from_entry(entry: ContextMapEntry) -> GenerationTrace:
    """Derive the generation grounding trace from a frozen context-map entry.

    Because it is derived from the checkpointed entry, the trace is reproduced
    identically on resume — the persisted story/scenario records the exact context
    that fed the LLM, not a re-retrieved approximation.
    """
    return GenerationTrace(
        generation_context_run_id=entry.generation_context_run_id,
        retrieved_assertion_ids=[item.assertion_id for item in entry.assertions],
        retrieved_chunk_ids=list(entry.chunk_ids),
        context_mode=entry.context_mode,
    )


def validate_context_map(
    entries: Sequence[ContextMapEntry],
    *,
    expected_document_version_id: str,
    require_story_id: bool,
    evidence_by_key: Mapping[str, list[str]] | None = None,
    logger: RunLogger | None = None,
) -> list[ContextMapEntry]:
    """Normalize and validate the in-memory context map (local only).

    * Deduplicates chunk ids per entry while preserving deterministic rank order
      and re-numbering ranks contiguously.
    * Rebuilds ``chunk_ids`` from the deduped metadata so every metadata record
      points to a chunk in ``chunk_ids`` and vice versa.
    * Ensures each entry's chunk ids belong to ``expected_document_version_id``.
    * Re-adds any available evidence chunk ids that are missing.
    * Allows empty retrieval (the generation agents fall back to input text).
    """
    validated: list[ContextMapEntry] = []
    for entry in entries:
        key = _entry_key(entry, require_story_id=require_story_id)
        if entry.document_version_id != expected_document_version_id:
            raise CheckpointError(
                "context map entry "
                f"{key} has document_version_id {entry.document_version_id!r} "
                f"but expected {expected_document_version_id!r}"
            )
        ordered = sorted(entry.retrieval_metadata, key=lambda meta: meta.rank)
        deduped: list[RetrievalMetadataEntry] = []
        seen: set[str] = set()
        for meta in ordered:
            if meta.chunk_id in seen:
                continue
            seen.add(meta.chunk_id)
            deduped.append(meta)

        evidence_ids = list(evidence_by_key.get(key, [])) if evidence_by_key else []
        for evidence_id in evidence_ids:
            if evidence_id not in seen:
                seen.add(evidence_id)
                deduped.append(
                    RetrievalMetadataEntry(
                        chunk_id=evidence_id,
                        source=_EVIDENCE_SOURCE,
                        rank=len(deduped) + 1,
                    )
                )

        renumbered = [
            meta.model_copy(update={"rank": rank}) for rank, meta in enumerate(deduped, start=1)
        ]
        if not renumbered and logger is not None:
            logger.warning(
                "Context map entry has no retrieved chunks; generation falls back to input text",
                step="context_map.validate",
                input_id=key,
                document_version_id=expected_document_version_id,
                status="degraded",
            )
        validated.append(
            entry.model_copy(
                update={
                    "chunk_ids": [meta.chunk_id for meta in renumbered],
                    "retrieval_metadata": renumbered,
                }
            )
        )
    return validated


def hydrate_context_map_entry(
    entry: ContextMapEntry,
    *,
    neo4j: ChunkFetcher,
    logger: RunLogger | None = None,
) -> RetrievedContext:
    """Fetch full chunk text for an entry's chunk ids, preserving rank order.

    The context map stores identifiers only; this rehydrates them into a
    prompt-ready :class:`RetrievedContext` right before prompt construction so
    the LLM always receives actual chunk text, never bare ids.
    """
    if not entry.chunk_ids:
        source = entry.context_mode if entry.assertions else "requirement_text_fallback"
        return RetrievedContext(chunks=[], source=source, assertions=list(entry.assertions))
    text_by_id = dict(neo4j.fetch_chunks(list(entry.chunk_ids)))
    chunks: list[RetrievedChunk] = []
    missing: list[str] = []
    for chunk_id in entry.chunk_ids:
        text = text_by_id.get(chunk_id)
        if text is None:
            missing.append(chunk_id)
            continue
        chunks.append(RetrievedChunk(chunk_id=chunk_id, text=text))
    if missing and logger is not None:
        logger.warning(
            "Some context-map chunk ids could not be hydrated; continuing with the rest",
            step="context_map.hydrate",
            input_id=_entry_key(entry, require_story_id=entry.story_id is not None),
            missing_chunk_ids=missing,
            status="degraded",
        )
    if not chunks:
        # All requested chunk ids were missing from the store. When the checkpoint
        # froze a graph-primary assertion selection, that structured context is the
        # actual grounding for generation and must survive missing chunk text — do
        # not downgrade to the requirement-text fallback and silently drop it.
        if entry.assertions:
            return RetrievedContext(
                chunks=[], source=entry.context_mode, assertions=list(entry.assertions)
            )
        return RetrievedContext(chunks=[], source="requirement_text_fallback")
    if entry.assertions:
        return RetrievedContext(
            chunks=chunks, source=entry.context_mode, assertions=list(entry.assertions)
        )
    non_evidence = any(meta.source != _EVIDENCE_SOURCE for meta in entry.retrieval_metadata)
    return RetrievedContext(chunks=chunks, source="hybrid" if non_evidence else "evidence")


def write_context_map_checkpoint(
    out_dir: Path,
    *,
    run_id: str,
    stage: str,
    project: str,
    document_version_id: str,
    entries: Sequence[ContextMapEntry],
) -> Path:
    payload = {
        "run_id": run_id,
        "stage": stage,
        "project": project,
        "document_version_id": document_version_id,
        "entries": [_dump_entry(entry) for entry in entries],
    }
    path = out_dir / context_map_filename(stage)
    atomic_write_json(path, payload)
    return path


def load_context_map_checkpoint(
    out_dir: Path,
    *,
    stage: str,
    project: str,
    document_version_id: str,
    run_id: str,
    logger: RunLogger | None = None,
) -> list[ContextMapEntry] | None:
    """Load and validate a prior context map; ``None`` if none exists.

    Raises :class:`CheckpointError` if the file is corrupted or belongs to a
    different stage/project/document version so checkpoint data is never mixed
    silently across runs.
    """
    path = out_dir / context_map_filename(stage)
    if not path.exists():
        return None
    checkpoint = _load_model(path, ContextMapCheckpoint)
    _guard_identity(
        path,
        found_stage=checkpoint.stage,
        found_project=checkpoint.project,
        found_document_version_id=checkpoint.document_version_id,
        stage=stage,
        project=project,
        document_version_id=document_version_id,
    )
    _log_run_id_reuse(checkpoint.run_id, run_id, path, logger)
    return checkpoint.entries


def load_generation_progress(
    out_dir: Path,
    *,
    stage: str,
    project: str,
    document_version_id: str,
    run_id: str,
    logger: RunLogger | None = None,
) -> GenerationProgress:
    """Load prior per-item progress, or a fresh empty tracker if none exists."""
    path = out_dir / generation_progress_filename(stage)
    if not path.exists():
        return GenerationProgress(
            run_id=run_id,
            stage=stage,
            project=project,
            document_version_id=document_version_id,
        )
    progress = _load_model(path, GenerationProgress)
    _guard_identity(
        path,
        found_stage=progress.stage,
        found_project=progress.project,
        found_document_version_id=progress.document_version_id,
        stage=stage,
        project=project,
        document_version_id=document_version_id,
    )
    _log_run_id_reuse(progress.run_id, run_id, path, logger)
    return progress


def write_generation_progress(out_dir: Path, progress: GenerationProgress) -> Path:
    payload = {
        "run_id": progress.run_id,
        "stage": progress.stage,
        "project": progress.project,
        "document_version_id": progress.document_version_id,
        "completed": {key: _dump_item(item) for key, item in progress.completed.items()},
        "failed": {key: _dump_item(item) for key, item in progress.failed.items()},
    }
    path = out_dir / generation_progress_filename(progress.stage)
    atomic_write_json(path, payload)
    return path


def record_completion(
    progress: GenerationProgress,
    *,
    input_id: str,
    payload: dict[str, Any],
    requirement_id: str | None = None,
) -> None:
    progress.failed.pop(input_id, None)
    progress.completed[input_id] = GenerationProgressItem(
        input_id=input_id,
        requirement_id=requirement_id,
        status="completed",
        completed_at=_now_iso(),
        payload=payload,
    )


def record_failure(
    progress: GenerationProgress,
    *,
    input_id: str,
    error: str,
    requirement_id: str | None = None,
) -> None:
    progress.failed[input_id] = GenerationProgressItem(
        input_id=input_id,
        requirement_id=requirement_id,
        status="failed",
        completed_at=_now_iso(),
        payload={"error": error},
    )


def append_generation_error(out_dir: Path, stage: str, record: dict[str, Any]) -> None:
    """Append one per-item failure to the local debug log (best effort)."""
    path = out_dir / generation_errors_filename(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"logged_at": _now_iso(), **record}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _entry_key(entry: ContextMapEntry, *, require_story_id: bool) -> str:
    if require_story_id:
        if not entry.story_id:
            raise CheckpointError(
                f"context map entry for requirement {entry.requirement_id} is missing story_id"
            )
        return entry.story_id
    return entry.requirement_id


def _dump_entry(entry: ContextMapEntry) -> dict[str, Any]:
    exclude = {"story_id"} if entry.story_id is None else set()
    if not entry.assertions:
        # Legacy chunk-only entries stay byte-identical: omit graph-primary fields.
        exclude = exclude | {"context_mode", "assertions"}
    if not entry.generation_context_run_id:
        exclude = exclude | {"generation_context_run_id"}
    return entry.model_dump(mode="json", exclude=exclude)


def _dump_item(item: GenerationProgressItem) -> dict[str, Any]:
    exclude = {"requirement_id"} if item.requirement_id is None else set()
    return item.model_dump(mode="json", exclude=exclude)


def _load_model[ModelT: StrictModel](path: Path, model: type[ModelT]) -> ModelT:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(data)
    except (json.JSONDecodeError, ValidationError, OSError, ValueError) as exc:
        raise CheckpointError(
            f"checkpoint {path} is corrupted or incompatible ({exc}); "
            "delete it to restart this stage from the beginning"
        ) from exc


def _guard_identity(
    path: Path,
    *,
    found_stage: str,
    found_project: str,
    found_document_version_id: str,
    stage: str,
    project: str,
    document_version_id: str,
) -> None:
    for label, found, expected in (
        ("stage", found_stage, stage),
        ("project", found_project, project),
        ("document_version_id", found_document_version_id, document_version_id),
    ):
        if found != expected:
            raise CheckpointError(
                f"checkpoint {path} belongs to a different {label} "
                f"({found!r} != {expected!r}); refusing to mix checkpoint data"
            )


def _log_run_id_reuse(
    found_run_id: str,
    run_id: str,
    path: Path,
    logger: RunLogger | None,
) -> None:
    if found_run_id == run_id or logger is None:
        return
    logger.info(
        "Resuming from checkpoint written by an earlier run",
        step="checkpoint.resume",
        checkpoint_path=str(path),
        checkpoint_run_id=found_run_id,
        run_id=run_id,
        status="resumed",
    )
