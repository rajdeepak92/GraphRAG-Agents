"""Stable domain enumerations."""

from __future__ import annotations

from enum import StrEnum


class ReplacePolicy(StrEnum):
    """How same document/version conflicts are handled."""

    REJECT = "reject"
    REPLACE_VERSION = "replace_version"
    RESUME_EXISTING = "resume_existing"


class FactType(StrEnum):
    """Canonical fact classification."""

    BUSINESS_RULE = "business_rule"
    FUNCTIONAL_BEHAVIOR = "functional_behavior"
    SECURITY_CONSTRAINT = "security_constraint"
    DATA_CONSTRAINT = "data_constraint"
    PERFORMANCE_CONSTRAINT = "performance_constraint"
    INTEGRATION_CONSTRAINT = "integration_constraint"
    OPERATIONAL_CONSTRAINT = "operational_constraint"
    ASSUMPTION = "assumption"
    UNKNOWN = "unknown"


class RequirementType(StrEnum):
    """Canonical requirement category."""

    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DATA = "data"
    INTEGRATION = "integration"
    OPERATIONAL = "operational"


class DerivationType(StrEnum):
    """How a requirement was derived from facts."""

    DIRECT = "direct"
    NORMALIZED = "normalized"
    INFERRED = "inferred"
    CROSS_CHUNK = "cross_chunk"


class ValidationStatus(StrEnum):
    """Validation state of a candidate or persisted requirement."""

    CANDIDATE = "candidate"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    CONFLICTING = "conflicting"


class RunStatus(StrEnum):
    """Ingestion run state machine statuses."""

    REQUESTED = "requested"
    VALIDATED = "validated"
    SOURCE_REGISTERED = "source_registered"
    PARSED = "parsed"
    CHUNKED = "chunked"
    CHUNKS_PERSISTED = "chunks_persisted"
    GRAPH_PROJECTED = "graph_projected"
    VECTORS_INDEXED = "vectors_indexed"
    DISCOVERY_RUNNING = "discovery_running"
    REQUIREMENTS_VALIDATED = "requirements_validated"
    REQUIREMENTS_PERSISTED = "requirements_persisted"
    ARTIFACT_WRITTEN = "artifact_written"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIALLY_COMPLETED = "partially_completed"
    CANCELLED = "cancelled"


class RunStepName(StrEnum):
    """Executable ingestion workflow step names."""

    VALIDATE_COMMAND = "validate_command"
    BOOTSTRAP_RUNTIME = "bootstrap_runtime"
    CHECK_DEPENDENCIES = "check_dependencies"
    REGISTER_RUN = "register_run"
    RESOLVE_VERSION_LINEAGE = "resolve_version_lineage"
    PARSE_DOCUMENT = "parse_document"
    CHUNK_DOCUMENT = "chunk_document"
    PERSIST_MANIFEST = "persist_manifest"
    PERSIST_SOURCE_RECORDS = "persist_source_records"
    PROJECT_CHUNK_GRAPH = "project_chunk_graph"
    INDEX_CHUNK_VECTORS = "index_chunk_vectors"
    CREATE_DISCOVERY_RUN = "create_discovery_run"
    DISCOVER_REQUIREMENTS = "discover_requirements"
    RECONCILE_REQUIREMENTS = "reconcile_requirements"
    PERSIST_REQUIREMENTS = "persist_requirements"
    PROJECT_REQUIREMENT_GRAPH = "project_requirement_graph"
    WRITE_ARTIFACT = "write_artifact"
    VERIFY_OUTPUTS = "verify_outputs"
    COMPLETE_RUN = "complete_run"
