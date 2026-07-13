"""Azure OpenAI adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any, NotRequired, TypedDict, TypeVar, cast

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.config.settings import AzureOpenAISettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.llm_models.json_output import (
    parse_json_object,
    sanitized_output_preview,
)
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary

T = TypeVar("T", bound=BaseModel)
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_SYSTEM_MESSAGE = PromptRequirementDiscovery.SYS_PROMPT_REQUIREMENT_DISCOVERY.value


class _ChatCompletionRequest(TypedDict):
    """Describe the chat completion request state exchanged between typed workflow nodes."""

    model: str
    messages: list[dict[str, str]]
    temperature: NotRequired[float]


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
        self._response_context: dict[str, Any] = {}
        self._system_message = _DEFAULT_SYSTEM_MESSAGE
        # Once a deployment rejects an explicit temperature, remember it so every
        # later call omits the parameter up front instead of paying a failing
        # round-trip per chunk. Reset only by constructing a new model.
        self._omit_temperature = False

    def set_system_message(self, text: str) -> None:
        """Execute the set system message operation within its declared architectural boundary.

        Args:
            text (str): Input text processed in memory and excluded from diagnostic logs.
        """
        self._system_message = text

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
            )
        response_path = self.run_dir / Path(response_name).name
        response_path.parent.mkdir(parents=True, exist_ok=True)
        response_path.write_text(self._last_completion, encoding="utf-8")
        self.last_response_path = response_path
        return response_path

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        """Generate structured.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            schema (type[T]): Schema required by the operation's typed contract.

        Returns:
            T: The typed result produced by the operation.

        Raises:
            ModelOutputError: If validated inputs or required dependencies cannot satisfy the
            contract.
        """
        self.last_response_path = None
        first_completion = self._generate_completion(prompt)
        try:
            parsed = self._parse_completion(first_completion, schema)
        except (ValueError, ValidationError) as first_error:
            self._record_response(
                prompt=prompt,
                completion=first_completion,
                attempt=1,
                parse_failed=True,
                error=first_error,
            )
            retry_prompt = _build_retry_prompt(prompt, first_error)
        else:
            self._record_response(
                prompt=prompt,
                completion=first_completion,
                attempt=1,
                parse_failed=False,
                error=None,
            )
            return parsed

        retry_completion = self._generate_completion(retry_prompt)
        try:
            parsed = self._parse_completion(retry_completion, schema)
        except (ValueError, ValidationError) as retry_error:
            self._record_response(
                prompt=prompt,
                completion=retry_completion,
                attempt=2,
                parse_failed=True,
                error=retry_error,
            )
            raise ModelOutputError(
                "Azure OpenAI model output could not be validated after retry: "
                f"{sanitized_exception_summary(retry_error)}; "
                f"diagnostic={sanitized_output_preview(retry_completion)}"
            ) from retry_error

        self._record_response(
            prompt=prompt,
            completion=retry_completion,
            attempt=2,
            parse_failed=False,
            error=None,
        )
        return parsed

    def _generate_completion(self, prompt: str) -> str:
        """Generate completion.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.

        Returns:
            str: The typed result produced by the operation.

        Raises:
            RuntimeError: If validated inputs or required dependencies cannot satisfy the contract.
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
        )
        request = _chat_completion_request(
            model=self.settings.reasoning_deployment,
            messages=[
                {
                    "role": "system",
                    "content": self._system_message,
                },
                {"role": "user", "content": prompt},
            ],
            temperature=None if self._omit_temperature else self.settings.reasoning_temperature,
        )
        try:
            response = client.chat.completions.create(**request)
        except openai.BadRequestError as error:
            if "temperature" not in request or not _is_unsupported_temperature_error(error):
                raise
            # Azure reasoning deployments may reject any explicit temperature. Retry
            # exactly once with only that optional parameter removed; all other 400s
            # and any second failure propagate unchanged. Remember the rejection so
            # subsequent calls skip the doomed round-trip entirely.
            request.pop("temperature")
            self._omit_temperature = True
            response = client.chat.completions.create(**request)
        return cast(str, response.choices[0].message.content or "{}")

    def _parse_completion(self, completion: str, schema: type[T]) -> T:
        """Parse completion.

        Args:
            completion (str): Completion required by the operation's typed contract.
            schema (type[T]): Schema required by the operation's typed contract.

        Returns:
            T: The typed result produced by the operation.
        """
        return schema.model_validate(parse_json_object(completion))

    def _record_response(
        self,
        *,
        prompt: str,
        completion: str,
        attempt: int,
        parse_failed: bool,
        error: Exception | None,
    ) -> None:
        """Record response through the owning storage boundary.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            completion (str): Completion required by the operation's typed contract.
            attempt (int): Bounded attempt used for deterministic processing.
            parse_failed (bool): Parse failed required by the operation's typed contract.
            error (Exception | None): Failure being classified or converted without exposing its
                                      message.

        Side Effects:
            Emits sanitized run-scoped diagnostics when a logger is available.
        """
        self._last_completion = completion
        self._last_prompt = prompt
        self._last_parse_attempt = attempt
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
                step="discover_requirements.llm_response",
                operation="azure_openai.parse_structured",
                attempt=attempt,
                max_attempts=2,
                retry_delay_seconds=0.0,
                response_path=str(response_path) if response_path else None,
                exception_type=error.__class__.__name__ if error else None,
                error_summary=sanitized_exception_summary(error) if error else None,
                raw_response=completion,
                status="retrying" if attempt < 2 else "failed",
            )
        else:
            self.logger.info(
                "Azure OpenAI response diagnostic captured",
                step="discover_requirements.llm_response",
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

    def __init__(self, settings: AzureOpenAISettings) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (AzureOpenAISettings): Validated settings that control this operation.
        """
        self.settings = settings
        self.embedding_fingerprint = f"azure:{settings.embedding_deployment}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Execute the embed documents operation within its declared architectural boundary.

        Args:
            texts (list[str]): Texts required by the operation's typed contract.

        Returns:
            list[list[float]]: The typed result produced by the operation.

        Raises:
            RuntimeError: If validated inputs or required dependencies cannot satisfy the contract.
        """
        if not all(
            [
                self.settings.endpoint,
                self.settings.api_key,
                self.settings.embedding_deployment,
            ]
        ):
            raise RuntimeError("Azure OpenAI embeddings are not configured")

        AzureOpenAI = cast(Any, import_module("openai")).AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=self.settings.endpoint,
            api_key=self.settings.api_key,
            api_version=self.settings.api_version,
        )
        response = client.embeddings.create(model=self.settings.embedding_deployment, input=texts)
        return [item.embedding for item in response.data]


def _chat_completion_request(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float | None,
) -> _ChatCompletionRequest:
    """Execute the chat completion request operation within its declared architectural boundary.

    Args:
        model (str): Provider-neutral model adapter used by the operation.
        messages (list[dict[str, str]]): Messages required by the operation's typed contract.
        temperature (float | None): Temperature required by the operation's typed contract.

    Returns:
        _ChatCompletionRequest: The typed result produced by the operation.
    """
    request: _ChatCompletionRequest = {"model": model, "messages": messages}
    if temperature is not None:
        request["temperature"] = temperature
    return request


def _is_unsupported_temperature_error(error: Exception) -> bool:
    """Return whether unsupported temperature error.

    Args:
        error (Exception): Failure being classified or converted without exposing its message.

    Returns:
        bool: The typed result produced by the operation.
    """
    param = getattr(error, "param", None)
    code = getattr(error, "code", None)
    body = getattr(error, "body", None)
    if isinstance(body, Mapping):
        payload: Mapping[str, Any] = body
        nested = body.get("error")
        if isinstance(nested, Mapping):
            payload = nested
        param = param or payload.get("param")
        code = code or payload.get("code")
    return param == "temperature" and code == "unsupported_value"


def _build_retry_prompt(prompt: str, error: Exception) -> str:
    """Build retry prompt.

    Args:
        prompt (str): Prompt text passed only to the selected model provider and never logged.
        error (Exception): Failure being classified or converted without exposing its message.

    Returns:
        str: The typed result produced by the operation.
    """
    return (
        f"{prompt}\n\n"
        "The previous response failed validation. "
        "Return one corrected JSON object only.\n"
        f"Validation error: {error}"
    )


def _extract_chunks(prompt: str) -> list[tuple[str, str]]:
    """Extract chunks.

    Args:
        prompt (str): Prompt text passed only to the selected model provider and never logged.

    Returns:
        list[tuple[str, str]]: The typed result produced by the operation.
    """
    matches = re.findall(r"<chunk id=\"([^\"]+)\">\n(.*?)\n</chunk>", prompt, flags=re.S)
    return [(chunk_id, text.strip()) for chunk_id, text in matches]


def _response_filename(
    prompt: str,
    attempt: int,
    context: dict[str, Any] | None = None,
) -> str:
    """Execute the response filename operation within its declared architectural boundary.

    Args:
        prompt (str): Prompt text passed only to the selected model provider and never logged.
        attempt (int): Bounded attempt used for deterministic processing.
        context (dict[str, Any] | None): Context required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    if context:
        batch_index = context.get("batch_index")
        validation_attempt = context.get("validation_attempt")
        if isinstance(batch_index, int) and isinstance(validation_attempt, int):
            suffix = f"{batch_index}_{validation_attempt}"
            if attempt != 1:
                suffix = f"{suffix}_parse{attempt}"
            return f"llm_response_{suffix}.txt"

    chunks = _extract_chunks(prompt)
    if len(chunks) == 1:
        chunk_id = chunks[0][0]
    elif chunks:
        chunk_id = f"batch_{chunks[0][0]}_{chunks[-1][0]}"
    else:
        chunk_id = "unknown"
    safe_chunk_id = _SAFE_RESPONSE_NAME.sub("_", chunk_id).strip("._") or "unknown"
    return f"llm_response_{safe_chunk_id}_{attempt}.txt"
