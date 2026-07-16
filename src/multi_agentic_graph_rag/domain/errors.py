"""Application exception hierarchy."""


class MaragError(Exception):
    """Base application error."""


class ConfigurationError(MaragError):
    """Operational configuration is invalid."""


class ModelOutputError(MaragError):
    """A reasoning provider did not produce valid structured output."""


class StoreUnavailableError(MaragError):
    """A required external store is unavailable."""


class TraceValidationError(MaragError):
    """Supplied evidence or traceability is invalid."""


__all__ = [
    "ConfigurationError",
    "MaragError",
    "ModelOutputError",
    "StoreUnavailableError",
    "TraceValidationError",
]
