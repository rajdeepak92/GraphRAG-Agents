"""Google Gemini (Developer API) adapters."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.common_prompt_defs import PromptSharedFragments
from multi_agentic_graph_rag.config.settings import GeminiSettings
from multi_agentic_graph_rag.domain.errors import (
    ConfigurationError,
    ContentFilterError,
    MalformedModelOutputError,
    ModelOutputError,
)
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary

T = TypeVar("T", bound=BaseModel)
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
# Conservative per-request input cap for the Developer API embeddings endpoint.
_EMBEDDING_MAX_INPUTS = 100


def _load_genai() -> tuple[Any, Any]:
    """Import the google-genai SDK lazily and return (genai, types)."""
    genai = cast(Any, import_module("google.genai"))
    types = cast(Any, import_module("google.genai.types"))
    return genai, types


class GeminiReasoningModel:
    """Coordinate Gemini reasoning model behavior within the llm_models boundary."""

    provider_name = "gemini"

    def __init__(
        self,
        settings: GeminiSettings,
        *,
        discovery_batch_size: int = 1,
        log_llm_responses: bool = False,
        logger: Any | None = None,
        run_dir: Path | None = None,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (GeminiSettings): Validated settings that control this operation.
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
        self._client: Any | None = None

    def warmup(self) -> None:
        """Fail fast when the Gemini provider is not configured."""
        if not self.settings.api_key:
            raise ConfigurationError("Gemini reasoning requires GEMINI_API_KEY")
        if not self.settings.reasoning_model:
            raise ConfigurationError("Gemini reasoning requires a reasoning model")

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
        # Symmetric with the Azure adapter: Gemini's response schema cannot enforce
        # every Pydantic constraint (numeric bounds, non-empty strings, and
        # cross-field rules are stripped or inexpressible), so a well-formed
        # response can still violate the Python schema. Feed the validation error
        # back and regenerate up to ``attempt_limit`` before failing, persisting the
        # raw response on every failed attempt.
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
                    raise MalformedModelOutputError(
                        "Gemini structured output violated the Python schema: "
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
        raise ModelOutputError("Gemini structured output could not be produced")

    def _request_completion(
        self,
        prompt: str,
        *,
        schema: type[BaseModel],
        system_message: str,
    ) -> str:
        """Request one schema-constrained JSON completion string from Gemini.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            schema (type[BaseModel]): Pydantic response contract supplied to Gemini.
            system_message (str): Operation-scoped system instruction for this request.

        Returns:
            str: Raw JSON content retained for validation and diagnostics.

        Raises:
            ModelOutputError: If Gemini refuses or omits the structured response.
            ConfigurationError: If the Gemini provider is not configured.
            ContentFilterError: If Gemini safety filtering blocks the request or response.
        """
        if not self.settings.api_key or not self.settings.reasoning_model:
            raise RuntimeError("Gemini reasoning is not configured")

        genai, types = _load_genai()
        client = self._ensure_client(genai)
        config = types.GenerateContentConfig(
            system_instruction=system_message,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            temperature=0.0,
        )
        try:
            response = client.models.generate_content(
                model=self.settings.reasoning_model,
                contents=prompt,
                config=config,
            )
        except Exception as error:
            if _is_quota_error(error):
                raise ConfigurationError(
                    f"Gemini quota is exhausted for model '{self.settings.reasoning_model}' "
                    f"({_provider_error_summary(error)}); enable billing or select a model "
                    "with available quota via GEMINI_REASONING_MODEL"
                ) from error
            if _is_unavailable_error(error):
                raise ConfigurationError(
                    f"Gemini model '{self.settings.reasoning_model}' is unavailable "
                    f"({_provider_error_summary(error)}); verify GEMINI_REASONING_MODEL names a "
                    "valid, available model (for example gemini-2.5-flash) and retry"
                ) from error
            if _is_configuration_error(error):
                raise ConfigurationError(
                    f"Gemini rejected the request ({_provider_error_summary(error)}); verify "
                    f"GEMINI_API_KEY and the model name '{self.settings.reasoning_model}'"
                ) from error
            raise

        block_reason = _prompt_block_reason(response)
        if block_reason:
            raise ContentFilterError("Gemini safety filtering blocked the request")
        candidate = _first_candidate(response)
        if candidate is not None and _candidate_blocked(candidate):
            raise ContentFilterError("Gemini safety filtering blocked the response")
        content = getattr(response, "text", None)
        if not isinstance(content, str) or not content.strip():
            raise ModelOutputError("Gemini returned no parsed structured response content")
        return content

    def _ensure_client(self, genai: Any) -> Any:
        """Build (and memoize) the Gemini Developer API client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.api_key)
        return self._client

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
                "Gemini response failed structured parsing",
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
                "Gemini response diagnostic captured",
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
                title=f"Gemini raw response attempt={attempt}",
                body=completion,
                also_stderr=self.log_llm_responses,
            )


class GeminiEmbeddingModel:
    """Coordinate Gemini embedding model behavior within the llm_models boundary."""

    provider_name = "gemini"

    def __init__(self, settings: GeminiSettings, *, logger: Any | None = None) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (GeminiSettings): Validated settings that control this operation.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
        """
        self.settings = settings
        self.embedding_fingerprint = f"gemini:{settings.embedding_model}"
        self.logger = logger
        if not all([settings.api_key, settings.embedding_model]):
            raise RuntimeError("Gemini embeddings are not configured")
        genai, _ = _load_genai()
        self._client = genai.Client(api_key=settings.api_key)
        self._embedding_dimension: int | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Execute the embed documents operation within its declared architectural boundary.

        Args:
            texts (list[str]): Texts required by the operation's typed contract.

        Returns:
            list[list[float]]: The typed result produced by the operation.

        Raises:
            ValueError: If validated inputs cannot satisfy the contract.
        """
        if not texts:
            return []
        if any(not isinstance(text, str) for text in texts):
            raise ValueError("Gemini embedding inputs must all be text")

        _, types = _load_genai()
        config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        batches = _embedding_batches(texts)
        ordered_vectors: list[list[float]] = []
        dimension = self._embedding_dimension
        for batch in batches:
            response = self._client.models.embed_content(
                model=self.settings.embedding_model,
                contents=batch,
                config=config,
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
                "Gemini embedding request completed",
                step="embed_documents",
                operation="gemini.embed_documents",
                input_count=len(texts),
                batch_count=len(batches),
                provider_call_count=len(batches),
                status="completed",
            )
        return ordered_vectors


def _embedding_batches(texts: list[str]) -> list[list[str]]:
    """Split embedding inputs into request batches within the per-request input cap."""
    return [
        texts[start : start + _EMBEDDING_MAX_INPUTS]
        for start in range(0, len(texts), _EMBEDDING_MAX_INPUTS)
    ]


def _validated_embedding_response(
    response: Any,
    *,
    expected_count: int,
    expected_dimension: int | None,
) -> tuple[list[list[float]], int]:
    """Validate one Gemini embedding response and return vectors in input order."""
    embeddings = getattr(response, "embeddings", None)
    if not isinstance(embeddings, list) or len(embeddings) != expected_count:
        raise ModelOutputError("Gemini embedding response count did not match the request")

    ordered: list[list[float]] = []
    dimension = expected_dimension
    for item in embeddings:
        vector = getattr(item, "values", None)
        if not isinstance(vector, list) or not vector:
            raise ModelOutputError("Gemini embedding response contained an empty vector")
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            raise ModelOutputError(
                "Gemini embedding response contained inconsistent vector dimensions"
            )
        ordered.append([float(value) for value in vector])

    if dimension is None:
        raise ModelOutputError("Gemini embedding response omitted all vectors")
    return ordered, dimension


def _build_repair_prompt(prompt: str, error: Exception) -> str:
    """Rebuild the request prompt with the Python-schema validation error fed back.

    The full validation error is passed to the model (never logged) so it can
    return a corrected object. The wording matches the shared repair fragments used
    by the agents and the Azure adapter, keeping both providers' schema-repair
    behavior symmetric.

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


def _first_candidate(response: Any) -> Any:
    """Return the first candidate of a Gemini response, or None."""
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        return candidates[0]
    return None


def _prompt_block_reason(response: Any) -> Any:
    """Return the prompt-feedback block reason if the request was blocked."""
    feedback = getattr(response, "prompt_feedback", None)
    return getattr(feedback, "block_reason", None) if feedback is not None else None


def _candidate_blocked(candidate: Any) -> bool:
    """Detect whether a candidate was terminated by Gemini safety filtering."""
    finish_reason = getattr(candidate, "finish_reason", None)
    name = getattr(finish_reason, "name", finish_reason)
    return isinstance(name, str) and name.upper() in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"}


def _is_configuration_error(error: Exception) -> bool:
    """Classify Gemini SDK errors that indicate an unusable key/model contract."""
    code = getattr(error, "code", None)
    if isinstance(code, int) and code in {400, 401, 403, 404}:
        return True
    status = getattr(error, "status", None)
    return isinstance(status, str) and status.upper() in {
        "INVALID_ARGUMENT",
        "PERMISSION_DENIED",
        "UNAUTHENTICATED",
        "NOT_FOUND",
    }


def _is_quota_error(error: Exception) -> bool:
    """Classify Gemini SDK errors caused by exhausted project/model quota."""
    code = getattr(error, "code", None)
    if isinstance(code, int) and code == 429:
        return True
    status = getattr(error, "status", None)
    return isinstance(status, str) and status.upper() == "RESOURCE_EXHAUSTED"


def _is_unavailable_error(error: Exception) -> bool:
    """Classify Gemini errors that indicate the model is unavailable or overloaded.

    A persistent 503 is also what the API returns for a model name that is not
    available to the caller, so the raised guidance points at the model name.
    """
    code = getattr(error, "code", None)
    if isinstance(code, int) and code == 503:
        return True
    status = getattr(error, "status", None)
    return isinstance(status, str) and status.upper() == "UNAVAILABLE"


def _provider_error_summary(error: Exception) -> str:
    """Return a short, credential-free summary of a Gemini API error for diagnostics."""
    status = getattr(error, "status", None)
    message = getattr(error, "message", None) or str(error)
    first_line = str(message).splitlines()[0].strip()[:200]
    parts = [part for part in (str(status).strip() if status else "", first_line) if part]
    return " - ".join(parts) or "no provider detail"


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
