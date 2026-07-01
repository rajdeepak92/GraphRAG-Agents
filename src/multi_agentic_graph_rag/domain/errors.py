"""Domain exceptions."""


class MaragError(Exception):
    """Base application error."""


class ConfigurationError(MaragError):
    """Configuration is invalid."""


class IngestionError(MaragError):
    """Ingestion failed."""


class VersionConflictError(IngestionError):
    """Same logical version has different source bytes."""


class StoreUnavailableError(IngestionError):
    """A required backing store is unavailable."""


class ModelOutputError(IngestionError):
    """A model returned output that could not be validated."""


class SchemaMismatchError(StoreUnavailableError):
    """A backing store schema does not match the current application contract."""


class TraceValidationError(IngestionError):
    """LLM trace data does not match source text."""


class UserStoryValidationError(MaragError):
    """Generated user-story content failed meaningfulness validation."""
