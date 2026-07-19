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


# --- Stage 4: autonomous test-code generation (plan §18) --------------------


class Stage4Error(MaragError):
    """Base for Stage-4 test-code generation failures."""


class TcIdCapacityExhausted(Stage4Error):
    """The six-digit TC sequence reached ``999999`` (plan §8.2, §18)."""


class RevisionRequired(Stage4Error):
    """An accepted logical key changed inputs; overwriting is forbidden (plan §8.4)."""


class InputManifestChanged(Stage4Error):
    """The frozen run manifest differs on resume (plan §5, §18)."""


class ExistingBehaviorChangeForbidden(Stage4Error):
    """A patch would alter an existing production/test_lib symbol (plan §7)."""


class Stage4PersistenceConflict(Stage4Error):
    """An immutable Stage-4 durable record conflicts with persisted state."""


class ProviderModelFailure(Stage4Error):
    """A selected reasoning provider failed after its single same-provider retry.

    Concrete subclasses carry the sanitized, provider-specific error code so
    Stage 4 never leaks credentials and never falls back to the other provider.
    """

    code: str = "PROVIDER_MODEL_FAILURE"


class AzureOpenAIModelFailure(ProviderModelFailure):
    """Azure OpenAI failed twice; correct Azure and retry the run (plan §9.1)."""

    code = "AZURE_OPENAI_MODEL_FAILURE"


class HuggingFaceModelFailure(ProviderModelFailure):
    """The private Hugging Face model failed twice (plan §9.2)."""

    code = "HUGGINGFACE_MODEL_FAILURE"


__all__ = [
    "AzureOpenAIModelFailure",
    "ConfigurationError",
    "ContentFilterError",
    "ExistingBehaviorChangeForbidden",
    "HuggingFaceModelFailure",
    "InputManifestChanged",
    "MalformedModelOutputError",
    "MaragError",
    "ModelOutputError",
    "ProviderModelFailure",
    "ProviderRefusalError",
    "RevisionRequired",
    "SemanticValidationError",
    "Stage4Error",
    "Stage4PersistenceConflict",
    "StoreUnavailableError",
    "TcIdCapacityExhausted",
    "TraceValidationError",
]
