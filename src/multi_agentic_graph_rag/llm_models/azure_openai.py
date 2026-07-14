"""Azure OpenAI adapters."""

from __future__ import annotations

import re
from datetime import date
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.common_prompt_defs import PromptSharedFragments
from multi_agentic_graph_rag.config.settings import AzureOpenAISettings
from multi_agentic_graph_rag.domain.errors import ConfigurationError, ModelOutputError
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary

T = TypeVar("T", bound=BaseModel)
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_SCHEMA_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_EMBEDDING_MAX_INPUTS = 2_048
_EMBEDDING_MAX_INPUT_TOKENS = 8_192
_EMBEDDING_MAX_REQUEST_TOKENS = 300_000
# Transient-transport retries are orthogonal to schema-repair attempts and must
# not be derived from ``max_attempts``; deriving one from the other multiplies the
# worst-case call count (transport attempts x schema attempts). The SDK retries
# only transient HTTP failures (timeouts, 429, 5xx), never a well-formed response
# that violates the Python schema — that is the schema-repair loop's job.
_TRANSPORT_MAX_RETRIES = 2
_AZURE_UNSUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "default",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minimum",
        "maximum",
        "multipleOf",
        "patternProperties",
        "unevaluatedProperties",
        "propertyNames",
        "minProperties",
        "maxProperties",
        "unevaluatedItems",
        "contains",
        "minContains",
        "maxContains",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


class AzureOpenAIReasoningModel:
    """Coordinate azure open aireasoning model behavior within the llm_models boundary."""

    provider_name = "azure_openai"

    def __init__(
        self,
        settings: AzureOpenAISettings,
        *,
        discovery_batch_size: int = 1,
        log_llm_responses: bool = False,
        logger: Any | None = None,
        run_dir: Path | None = None,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (AzureOpenAISettings): Validated settings that control this operation.
            discovery_batch_size (int): Bounded requirement-discovery batch size.
            log_llm_responses (bool): Log llm responses required by the operation's typed contract.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
            run_dir (Path | None): Filesystem location authorized for this operation.
        """
        self.settings = settings
        self.discovery_batch_size = discovery_batch_size
        self.log_llm_responses = log_llm_responses
        self.logger = logger
        self.run_dir = run_dir
        self.last_response_path: Path | None = None
        self._last_completion: str | None = None
        self._last_prompt: str | None = None
        self._last_parse_attempt = 0
        self._last_operation = "structured_generation"
        self._last_request_id = "request"
        self._last_call_index = 0
        self._structured_call_count = 0
        self._response_context: dict[str, Any] = {}

    def warmup(self) -> None:
        """Fail fast when the configured Azure API cannot enforce strict outputs."""
        if not _supports_structured_outputs_api(self.settings.api_version):
            raise ConfigurationError(_structured_outputs_capability_message(self.settings))

    def set_response_context(
        self,
        *,
        batch_index: int,
        attempt: int,
        chunk_ids: list[str],
    ) -> None:
        """Execute the set response context operation within its declared architectural boundary.

        Args:
            batch_index (int): Batch index required by the operation's typed contract.
            attempt (int): Bounded attempt used for deterministic processing.
            chunk_ids (list[str]): Chunk ids required by the operation's typed contract.
        """
        self._response_context = {
            "batch_index": batch_index,
            "validation_attempt": attempt,
            "chunk_ids": chunk_ids,
        }
        self.last_response_path = None

    def clear_response_context(self) -> None:
        """Clear safe response anchors after a provider call finishes."""
        self._response_context = {}

    def persist_last_response(self, *, filename: str | None = None) -> Path | None:
        """Persist last response through the owning storage boundary.

        Args:
            filename (str | None): Filename required by the operation's typed contract.

        Returns:
            Path | None: The typed result produced by the operation.

        Side Effects:
            May create or atomically replace files in the configured artifact boundary.
        """
        if self.run_dir is None or self._last_completion is None:
            return None
        response_name = filename
        if response_name is None:
            response_name = _response_filename(
                self._last_prompt or "",
                self._last_parse_attempt,
                self._response_context,
                operation=self._last_operation,
                request_id=self._last_request_id,
                call_index=self._last_call_index,
            )
        response_path = self.run_dir / Path(response_name).name
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(self._last_completion, encoding="utf-8")
        self.last_response_path = response_path
        return response_path

    def generate_structured(
        self,
        *,
        prompt: str,
        schema: type[T],
        system_message: str,
        operation: str,
        request_id: str,
        max_attempts: int = 2,
    ) -> T:
        """Generate structured.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            schema (type[T]): Schema required by the operation's typed contract.
            system_message (str): Operation-scoped system instruction retained across retries.
            operation (str): Safe operation label used for diagnostics.
            request_id (str): Safe request anchor used for diagnostics.
            max_attempts (int): Hard ceiling for provider transport attempts.

        Returns:
            T: The typed result produced by the operation.

        Raises:
            ModelOutputError: If validated inputs or required dependencies cannot satisfy the
            contract.
        """
        self.last_response_path = None
        self.warmup()
        self._structured_call_count += 1
        call_index = self._structured_call_count
        attempt_limit = max(1, min(max_attempts, 2))
        # Symmetric with the Hugging Face adapter: Azure strict Structured Outputs
        # cannot enforce every Pydantic constraint (numeric bounds, non-empty
        # strings, and cross-field rules are stripped or inexpressible), so a
        # well-formed response can still violate the Python schema. Feed the
        # validation error back and regenerate up to ``attempt_limit`` before
        # failing, persisting the raw response on every failed attempt.
        current_prompt = prompt
        for attempt in range(1, attempt_limit + 1):
            completion = self._request_completion(
                current_prompt,
                schema=schema,
                system_message=system_message,
            )
            try:
                parsed = schema.model_validate_json(completion)
            except (ValueError, ValidationError) as error:
                self._record_response(
                    prompt=current_prompt,
                    completion=completion,
                    attempt=attempt,
                    parse_failed=True,
                    error=error,
                    operation=operation,
                    request_id=request_id,
                    max_attempts=attempt_limit,
                    call_index=call_index,
                )
                if attempt >= attempt_limit:
                    raise ModelOutputError(
                        "Azure OpenAI structured output violated the Python schema: "
                        f"{sanitized_exception_summary(error)}"
                    ) from error
                current_prompt = _build_repair_prompt(prompt, error)
                continue
            self._record_response(
                prompt=current_prompt,
                completion=completion,
                attempt=attempt,
                parse_failed=False,
                error=None,
                operation=operation,
                request_id=request_id,
                max_attempts=attempt_limit,
                call_index=call_index,
            )
            return parsed
        raise ModelOutputError("Azure OpenAI structured output could not be produced")

    def _request_completion(
        self,
        prompt: str,
        *,
        schema: type[BaseModel],
        system_message: str,
    ) -> str:
        """Request one natively schema-constrained completion string from Azure.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            schema (type[BaseModel]): Strict Pydantic response contract supplied to Azure.
            system_message (str): Operation-scoped system instruction for this request.

        Returns:
            str: Raw JSON content retained for validation and diagnostics.

        Raises:
            ModelOutputError: If Azure refuses or omits the structured response.
            ConfigurationError: If the deployment/API cannot enforce strict outputs.
            RuntimeError: If the Azure provider is not configured.
        """
        if not all(
            [
                self.settings.endpoint,
                self.settings.api_key,
                self.settings.reasoning_deployment,
            ]
        ):
            raise RuntimeError("Azure OpenAI reasoning is not configured")

        openai = cast(Any, import_module("openai"))
        AzureOpenAI = openai.AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=self.settings.endpoint,
            api_key=self.settings.api_key,
            api_version=self.settings.api_version,
            max_retries=_TRANSPORT_MAX_RETRIES,
        )
        try:
            response = client.chat.completions.create(
                model=self.settings.reasoning_deployment,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format=_strict_response_format(schema),
            )
        except Exception as error:
            if _is_content_filter_error(error):
                raise ModelOutputError(
                    "Azure OpenAI content filtering blocked the request"
                ) from error
            if _is_azure_configuration_error(error, openai):
                raise ConfigurationError(
                    _structured_outputs_capability_message(self.settings)
                ) from error
            raise
        if not response.choices:
            raise ModelOutputError("Azure OpenAI returned no structured response choice")
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "content_filter":
            raise ModelOutputError("Azure OpenAI content filtering blocked the response")
        message = choice.message
        if getattr(message, "refusal", None):
            raise ModelOutputError("Azure OpenAI refused the structured request")
        content = getattr(message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise ModelOutputError("Azure OpenAI returned no parsed structured response content")
        return content

    def _record_response(
        self,
        *,
        prompt: str,
        completion: str,
        attempt: int,
        parse_failed: bool,
        error: Exception | None,
        operation: str,
        request_id: str,
        max_attempts: int,
        call_index: int,
    ) -> None:
        """Record response through the owning storage boundary.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            completion (str): Completion required by the operation's typed contract.
            attempt (int): Bounded attempt used for deterministic processing.
            parse_failed (bool): Parse failed required by the operation's typed contract.
            error (Exception | None): Failure being classified or converted without exposing its
                                      message.
            operation (str): Safe operation label used for diagnostics.
            request_id (str): Safe request anchor used for diagnostics.
            max_attempts (int): Schema-repair attempt ceiling used to classify retry status.
            call_index (int): Unique adapter invocation number for diagnostic file isolation.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        self._last_completion = completion
        self._last_prompt = prompt
        self._last_parse_attempt = attempt
        self._last_operation = operation
        self._last_request_id = request_id
        self._last_call_index = call_index
        self.last_response_path = None

        if not parse_failed and not self.log_llm_responses:
            return

        response_path: Path | None = None
        if self.run_dir is not None:
            response_path = self.persist_last_response()

        if self.logger is None:
            return

        if parse_failed:
            self.logger.warning(
                "Azure OpenAI response failed structured parsing",
                step=f"{operation}.llm_response",
                operation=f"{operation}.parse_structured",
                request_id=request_id,
                attempt=attempt,
                max_attempts=max_attempts,
                retry_delay_seconds=0.0,
                response_path=str(response_path) if response_path else None,
                exception_type=error.__class__.__name__ if error else None,
                error_summary=sanitized_exception_summary(error) if error else None,
                raw_response=completion,
                status="retrying" if attempt < max_attempts else "failed",
            )
        else:
            self.logger.info(
                "Azure OpenAI response diagnostic captured",
                step=f"{operation}.llm_response",
                operation=operation,
                request_id=request_id,
                attempt=attempt,
                response_path=str(response_path) if response_path else None,
                status="captured",
            )

        raw_block = getattr(self.logger, "raw_block", None)
        if callable(raw_block):
            raw_block(
                title=f"Azure OpenAI raw response attempt={attempt}",
                body=completion,
                also_stderr=self.log_llm_responses,
            )


class AzureOpenAIEmbeddingModel:
    """Coordinate azure open aiembedding model behavior within the llm_models boundary."""

    provider_name = "azure_openai"

    def __init__(self, settings: AzureOpenAISettings, *, logger: Any | None = None) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (AzureOpenAISettings): Validated settings that control this operation.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
        """
        self.settings = settings
        self.embedding_fingerprint = f"azure:{settings.embedding_deployment}"
        self.logger = logger
        if not all([settings.endpoint, settings.api_key, settings.embedding_deployment]):
            raise RuntimeError("Azure OpenAI embeddings are not configured")
        openai = import_module("openai")
        self._client = cast(Any, openai).AzureOpenAI(
            azure_endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
        )
        tiktoken = import_module("tiktoken")
        self._tokenizer = cast(Any, tiktoken).get_encoding("cl100k_base")
        self._embedding_dimension: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Execute the embed documents operation within its declared architectural boundary.

        Args:
            texts (list[str]): Texts required by the operation's typed contract.

        Returns:
            list[list[float]]: The typed result produced by the operation.

        Raises:
            RuntimeError: If validated inputs or required dependencies cannot satisfy the contract.
        """
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Azure embedding inputs must all be text")

        token_counts: list[int] = []
        for index, text in enumerate(texts):
            try:
                token_count = len(self._tokenizer.encode(text, disallowed_special=()))
            except Exception:
                raise ValueError(
                    f"Azure embedding input at index {index} could not be tokenized"
                ) from None
            if token_count > _EMBEDDING_MAX_INPUT_TOKENS:
                raise ValueError(
                    f"Azure embedding input at index {index} exceeds the per-input token limit"
                )
            token_counts.append(token_count)

        batches = _embedding_batches(texts, token_counts)
        ordered_vectors: list[list[float]] = []
        dimension = self._embedding_dimension
        for batch in batches:
            response = self._client.embeddings.create(
                model=self.settings.embedding_deployment,
                input=batch,
            )
            vectors, dimension = _validated_embedding_response(
                response,
                expected_count=len(batch),
                expected_dimension=dimension,
            )
            ordered_vectors.extend(vectors)

        self._embedding_dimension = dimension
        debug = getattr(self.logger, "debug", None)
        if callable(debug):
            debug(
                "Azure embedding request completed",
                step="embed_documents",
                operation="azure_openai.embed_documents",
                input_count=len(texts),
                batch_count=len(batches),
                provider_call_count=len(batches),
                status="completed",
            )
        return ordered_vectors


def _embedding_batches(texts: list[str], token_counts: list[int]) -> list[list[str]]:
    """Build stable Azure embedding batches within item and aggregate token limits."""
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_tokens = 0
    for text, token_count in zip(texts, token_counts, strict=True):
        if batch and (
            len(batch) >= _EMBEDDING_MAX_INPUTS
            or batch_tokens + token_count > _EMBEDDING_MAX_REQUEST_TOKENS
        ):
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(text)
        batch_tokens += token_count
    if batch:
        batches.append(batch)
    return batches


def _validated_embedding_response(
    response: Any,
    *,
    expected_count: int,
    expected_dimension: int | None,
) -> tuple[list[list[float]], int]:
    """Validate and reorder one Azure embedding response by its explicit indices."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != expected_count:
        raise ModelOutputError("Azure OpenAI embedding response count did not match the request")

    ordered: list[list[float] | None] = [None] * expected_count
    dimension = expected_dimension
    for item in data:
        index = getattr(item, "index", None)
        if isinstance(index, bool) or not isinstance(index, int):
            raise ModelOutputError("Azure OpenAI embedding response contained an invalid index")
        if index < 0 or index >= expected_count:
            raise ModelOutputError(
                "Azure OpenAI embedding response contained an out-of-range index"
            )
        if ordered[index] is not None:
            raise ModelOutputError("Azure OpenAI embedding response contained a duplicate index")
        vector = getattr(item, "embedding", None)
        if not isinstance(vector, list) or not vector:
            raise ModelOutputError("Azure OpenAI embedding response contained an empty vector")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ModelOutputError(
                "Azure OpenAI embedding response contained inconsistent vector dimensions"
            )
        ordered[index] = vector

    if dimension is None or any(vector is None for vector in ordered):
        raise ModelOutputError("Azure OpenAI embedding response omitted one or more indices")
    return [vector for vector in ordered if vector is not None], dimension


def _build_repair_prompt(prompt: str, error: Exception) -> str:
    """Rebuild the request prompt with the Python-schema validation error fed back.

    The full validation error is passed to the model (never logged) so it can
    return a corrected object. The wording matches the shared repair fragments
    used by the agents and the Hugging Face adapter, keeping both providers'
    schema-repair behavior symmetric.

    Args:
        prompt (str): Original prompt text sent to the provider and never logged.
        error (Exception): Pydantic validation failure to feed back verbatim to the model.

    Returns:
        str: The corrected-output prompt for the next attempt.
    """
    return (
        f"{prompt}\n\n"
        f"{PromptSharedFragments.CORRECTED_JSON_ONLY.value}\n"
        f"{PromptSharedFragments.VALIDATION_ERROR_PREFIX.value}{error}"
    )


def _strict_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    """Build Azure's supported strict JSON Schema envelope for a Pydantic model.

    Pydantic constraints unsupported by Azure remain enforced by the second,
    local ``model_validate_json`` boundary after generation.

    Args:
        schema (type[BaseModel]): Pydantic response contract for the operation.

    Returns:
        dict[str, Any]: Native ``json_schema`` response-format payload.
    """
    json_schema = schema.model_json_schema()
    _normalize_azure_schema(json_schema)
    schema_name = _SAFE_SCHEMA_NAME.sub("_", schema.__name__).strip("_") or "response"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": json_schema,
        },
    }


def _supports_structured_outputs_api(api_version: str) -> bool:
    """Return whether an Azure API version can enforce strict Structured Outputs."""
    normalized = api_version.strip().lower()
    if normalized == "v1":
        return True
    match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:-preview)?", normalized)
    if match is None:
        return False
    try:
        configured = date.fromisoformat(match.group(1))
    except ValueError:
        return False
    return configured >= date(2024, 8, 1)


def _structured_outputs_capability_message(settings: AzureOpenAISettings) -> str:
    """Build the non-secret strict-output readiness failure message."""
    return (
        "Azure strict Structured Outputs readiness failed for deployment "
        f"'{settings.reasoning_deployment}' with API version '{settings.api_version}'. "
        "The deployment model/version and API must support "
        "response_format=json_schema with strict=true; use API 2024-08-01-preview or newer "
        "(or GA v1) and a supported Azure model deployment."
    )


def _is_azure_configuration_error(error: Exception, openai: Any) -> bool:
    """Classify provider errors that indicate an unsupported deployment/API contract."""
    error_types = tuple(
        error_type
        for name in ("BadRequestError", "NotFoundError")
        if isinstance((error_type := getattr(openai, name, None)), type)
    )
    return bool(error_types) and isinstance(error, error_types)


def _is_content_filter_error(error: Exception) -> bool:
    """Detect Azure content-filter provider failures without logging their payload."""
    code = getattr(error, "code", None)
    if isinstance(code, str) and code.lower() in {"content_filter", "contentfilter"}:
        return True
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    nested = body.get("error")
    values = [body.get("code")]
    if isinstance(nested, dict):
        values.append(nested.get("code"))
    return any(
        isinstance(value, str) and value.lower() in {"content_filter", "contentfilter"}
        for value in values
    )


def _normalize_azure_schema(node: Any) -> None:
    """Normalize a Pydantic schema in place to Azure's strict supported subset.

    Args:
        node (Any): Current JSON Schema node being normalized recursively.
    """
    if isinstance(node, list):
        for item in node:
            _normalize_azure_schema(item)
        return
    if not isinstance(node, dict):
        return
    for key in list(node):
        if key in _AZURE_UNSUPPORTED_SCHEMA_KEYWORDS:
            node.pop(key)
            continue
        _normalize_azure_schema(node[key])
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
        node["additionalProperties"] = False


def _response_filename(
    prompt: str,
    attempt: int,
    context: dict[str, Any] | None = None,
    *,
    operation: str = "structured_generation",
    request_id: str = "request",
    call_index: int = 0,
) -> str:
    """Execute the response filename operation within its declared architectural boundary.

    Args:
        prompt (str): Prompt text passed only to the selected model provider and never logged.
        attempt (int): Bounded attempt used for deterministic processing.
        context (dict[str, Any] | None): Context required by the operation's typed contract.
        operation (str): Safe operation label used for diagnostics.
        request_id (str): Safe request anchor used for diagnostics.
        call_index (int): Unique adapter invocation number for diagnostic file isolation.

    Returns:
        str: The typed result produced by the operation.
    """
    if operation != "structured_generation" or request_id != "request":
        safe_operation = _SAFE_RESPONSE_NAME.sub("_", operation).strip("._") or "operation"
        safe_request_id = _SAFE_RESPONSE_NAME.sub("_", request_id).strip("._") or "request"
        return (
            f"llm_response_{safe_operation}_{safe_request_id}_"
            f"call-{call_index:06d}_attempt-{attempt}.txt"
        )

    if context:
        batch_index = context.get("batch_index")
        validation_attempt = context.get("validation_attempt")
        if isinstance(batch_index, int) and isinstance(validation_attempt, int):
            suffix = f"{batch_index}_{validation_attempt}"
            if attempt != 1:
                suffix = f"{suffix}_parse{attempt}"
            return f"llm_response_{suffix}.txt"

    return f"llm_response_structured_generation_request_{attempt}.txt"
