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


class TraceValidationError(IngestionError):
    """LLM trace data does not match source text."""
