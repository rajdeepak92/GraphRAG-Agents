"""Domain enums."""

from enum import StrEnum


class RunStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentVersionStatus(StrEnum):
    NEW = "new"
    IDEMPOTENT = "idempotent"
    REPLACED = "replaced"
