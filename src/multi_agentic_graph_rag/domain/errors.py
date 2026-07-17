"""Application exception hierarchy."""


class MaragError(Exception):
    """Base application error."""


class ConfigurationError(MaragError):
    """Operational configuration is invalid."""


class ModelOutputError(MaragError):
    """A reasoning provider did not produce valid structured output."""


class ProviderRefusalError(ModelOutputError):
    """A reasoning provider explicitly refused the request."""


class ContentFilterError(ModelOutputError):
    """A provider content filter blocked the request or response."""


class MalformedModelOutputError(ModelOutputError):
    """A provider response was not valid JSON for the requested schema."""


class SemanticValidationError(ModelOutputError):
    """A schema-valid provider response violated deterministic domain rules."""


class StoreUnavailableError(MaragError):
    """A required external store is unavailable."""


class TraceValidationError(MaragError):
    """Supplied evidence or traceability is invalid."""


__all__ = [
    "ConfigurationError",
    "ContentFilterError",
    "MalformedModelOutputError",
    "MaragError",
    "ModelOutputError",
    "ProviderRefusalError",
    "SemanticValidationError",
    "StoreUnavailableError",
    "TraceValidationError",
]
