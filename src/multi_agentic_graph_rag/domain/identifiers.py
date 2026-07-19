"""Stable project/item identities for the simplified workflow."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from uuid import uuid4

from uuid_utils import uuid7

_UNSAFE = re.compile(r"[^a-z0-9]+")


def stable_token(*parts: object, length: int = 20) -> str:
    """Return an uppercase SHA-256 prefix for deterministic identities."""
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length].upper()


def normalize_project(value: str) -> str:
    """Normalize a project only for database/collection/path compatibility."""
    normalized = _UNSAFE.sub("-", value.strip().lower()).strip("-")
    return normalized or "project"


def new_run_id(project: str) -> str:
    """Allocate an operational run ID that is never used as artifact identity."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"RUN-{timestamp}-{stable_token(project, uuid4().hex, length=6)}"


def make_chunk_id(
    *,
    chunk_text: str,
    start_char: int,
    end_char: int,
    source_location: str | None,
) -> str:
    """Build a stable chunk ID independent of project, run, and document identity."""
    return "CHK-" + stable_token(
        chunk_text,
        start_char,
        end_char,
        source_location or "",
        length=20,
    )


def new_requirement_id() -> str:
    """Allocate a permanent requirement UUIDv7."""
    return f"REQ-{str(uuid7()).upper()}"


def new_story_id() -> str:
    """Allocate a permanent user-story UUIDv7."""
    return f"US-{str(uuid7()).upper()}"


def new_scenario_id() -> str:
    """Allocate a permanent scenario UUIDv7."""
    return f"TS-{str(uuid7()).upper()}"


def make_criterion_id(story_id: str, ordinal: int, content: str) -> str:
    """Build a permanent criterion ID under its canonical story."""
    return f"AC-{stable_token(story_id, ordinal, content, length=18)}"


def make_evidence_id(parent_id: str, chunk_id: str, start_char: int, end_char: int) -> str:
    """Build a deterministic evidence ID."""
    return f"EVD-{stable_token(parent_id, chunk_id, start_char, end_char, length=20)}"


def make_entity_id(project: str, normalized_name: str, entity_type: str) -> str:
    """Build a project-scoped canonical entity ID."""
    token = stable_token(normalize_project(project), normalized_name, entity_type, length=20)
    return f"ENT-{token}"


def make_relationship_id(
    *,
    project: str,
    chunk_id: str,
    requirement_text_hash: str,
    source_entity_id: str,
    relationship_type: str,
    target_entity_id: str,
    evidence_hash: str,
) -> str:
    """Build a deterministic semantic projection ID from complete grounded provenance."""
    return "REL-" + stable_token(
        normalize_project(project),
        chunk_id,
        requirement_text_hash,
        source_entity_id,
        relationship_type,
        target_entity_id,
        evidence_hash,
        length=24,
    )


def make_checkpoint_thread_id(project: str, run_id: str, stage: str) -> str:
    """Derive the internal LangGraph thread identity."""
    return f"{normalize_project(project)}:{run_id}:{stage}"


# --- Stage 4: test-code generation identities -------------------------------


def new_codegen_run_id(project: str) -> str:
    """Allocate an operational Stage-4 codegen run ID (never artifact identity)."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"CGR-{timestamp}-{stable_token(project, uuid4().hex, length=6)}"


def new_test_case_id() -> str:
    """Allocate a permanent test-case UUIDv7."""
    return f"TC-{str(uuid7()).upper()}"


def make_framework_snapshot_id(
    *,
    repository_id: str,
    filesystem_checksum: str,
    extractor_version: str,
    extractor_config_hash: str,
) -> str:
    """Build a deterministic filesystem-derived framework snapshot identity.

    Stage 4 deliberately excludes Git metadata.  The normalized filesystem
    checksum is the complete code-content identity used for replay and KG
    publication.
    """
    return "FWS-" + stable_token(
        repository_id,
        filesystem_checksum,
        extractor_version,
        extractor_config_hash,
        length=24,
    )


def make_test_data_snapshot_id(
    *,
    project: str,
    workbook_checksum: str,
    normalized_checksum: str,
    schema_version: str,
    decision_revision: str,
) -> str:
    """Build a deterministic, project-scoped test-data snapshot identity."""
    return "TDS-" + stable_token(
        normalize_project(project),
        workbook_checksum,
        normalized_checksum,
        schema_version,
        decision_revision,
        length=24,
    )


def make_scenario_data_binding_id(
    *,
    scenario_id: str,
    execution_profile_id: str,
    test_data_snapshot_id: str,
) -> str:
    """Build a deterministic scenario-data binding identity."""
    return "BND-" + stable_token(
        scenario_id,
        execution_profile_id,
        test_data_snapshot_id,
        length=22,
    )


def make_resolved_bundle_id(
    *,
    binding_id: str,
    test_data_snapshot_id: str,
    parameterization_key: str = "",
) -> str:
    """Build a deterministic resolved test-data bundle identity."""
    return "TDB-" + stable_token(
        binding_id,
        test_data_snapshot_id,
        parameterization_key,
        length=22,
    )


def make_context_manifest_id(
    *,
    codegen_run_id: str,
    scenario_id: str,
    sequence: int,
) -> str:
    """Build a deterministic context-manifest identity for one model invocation."""
    return "CTX-" + stable_token(
        codegen_run_id,
        scenario_id,
        sequence,
        length=22,
    )


def make_capability_binding_id(
    *,
    scenario_id: str,
    capability_id: str,
) -> str:
    """Build a deterministic scenario-action to capability binding identity."""
    return "CAP-" + stable_token(scenario_id, capability_id, length=22)


def new_blocker_id() -> str:
    """Allocate a permanent Stage-4 blocker UUIDv7."""
    return f"BLK-{str(uuid7()).upper()}"


def make_code_symbol_id(
    *,
    snapshot_id: str,
    fqn: str,
    kind: str,
    start_byte: int,
    end_byte: int,
) -> str:
    """Build a snapshot-scoped code-symbol ID (never merged by short name, §6.6)."""
    return "SYM-" + stable_token(snapshot_id, fqn, kind, start_byte, end_byte, length=24)


def make_code_edge_id(
    *,
    snapshot_id: str,
    source_symbol_id: str,
    relation: str,
    target_symbol_id: str,
    source_location: str,
) -> str:
    """Build a deterministic code-relationship ID for a directed edge-labelled multigraph."""
    return "EDG-" + stable_token(
        snapshot_id,
        source_symbol_id,
        relation,
        target_symbol_id,
        source_location,
        length=24,
    )


def make_test_data_record_id(
    *,
    snapshot_id: str,
    record_type: str,
    natural_key: str,
) -> str:
    """Build a snapshot-scoped, typed test-data record ID from its natural key."""
    return "TDR-" + stable_token(snapshot_id, record_type, natural_key, length=22)


# --- Stage 4: permanent six-digit TC identity (plan §8) ----------------------

TC_ID_MIN = 100001
TC_ID_MAX = 999999
_TITLE_WORD = re.compile(r"[A-Za-z0-9]+")


def make_logical_key_hash(
    *,
    project_name: str,
    scenario_id: str,
    execution_profile_id: str,
    variant_id: str,
) -> str:
    """Hash the canonical logical test-case key (project+scenario+profile+variant).

    ``variant_id`` is deterministic Stage-3 input, never LLM prose. This hash
    backs the second uniqueness constraint on the reservation table (plan §8.1).
    """
    return "LKH-" + stable_token(
        normalize_project(project_name),
        scenario_id,
        execution_profile_id,
        variant_id or "default",
        length=32,
    )


def pascal_title(title: str) -> str:
    """Convert a free-text scenario title to a PascalCase identifier fragment.

    ``"Validate Temperature Sensor Threshold"`` -> ``"ValidateTemperatureSensorThreshold"``.
    Guaranteed to start with a letter so it is a legal Python identifier suffix.
    """
    words = _TITLE_WORD.findall(title)
    pascal = "".join(word[:1].upper() + word[1:] for word in words)
    if not pascal:
        pascal = "TestCase"
    if pascal[0].isdigit():
        pascal = "N" + pascal
    return pascal


def tc_stem(tc_id: int, title: str) -> str:
    """Build the exact ``Tc<6-digit-id><PascalTitle>`` stem shared by module + class.

    The Python filename stem, the class name, and the Robot library import must
    all match this exact string (plan §8.3, §12).
    """
    if not TC_ID_MIN <= tc_id <= TC_ID_MAX:
        raise ValueError(f"tc_id {tc_id} outside {TC_ID_MIN}..{TC_ID_MAX}")
    return f"Tc{tc_id}{pascal_title(title)}"


def make_provider_fingerprint_hash(
    *,
    provider: str,
    model: str,
    model_revision: str | None,
    generation_params_checksum: str,
    prompt_revision: str,
) -> str:
    """Hash the pinned provider fingerprint; every resume must match it (plan §9.3)."""
    return "PFP-" + stable_token(
        provider,
        model,
        model_revision or "",
        generation_params_checksum,
        prompt_revision,
        length=28,
    )


__all__ = [
    "TC_ID_MAX",
    "TC_ID_MIN",
    "make_capability_binding_id",
    "make_checkpoint_thread_id",
    "make_chunk_id",
    "make_code_edge_id",
    "make_code_symbol_id",
    "make_context_manifest_id",
    "make_criterion_id",
    "make_entity_id",
    "make_evidence_id",
    "make_framework_snapshot_id",
    "make_logical_key_hash",
    "make_provider_fingerprint_hash",
    "make_relationship_id",
    "make_resolved_bundle_id",
    "make_scenario_data_binding_id",
    "make_test_data_record_id",
    "make_test_data_snapshot_id",
    "new_blocker_id",
    "new_codegen_run_id",
    "new_requirement_id",
    "new_run_id",
    "new_scenario_id",
    "new_story_id",
    "new_test_case_id",
    "normalize_project",
    "pascal_title",
    "stable_token",
    "tc_stem",
]
