"""Hugging Face model adapters."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.config.settings import HuggingFaceSettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.llm_models.json_output import (
    parse_json_object,
    sanitized_output_preview,
)
from multi_agentic_graph_rag.observability.logging import sanitized_exception_summary

T = TypeVar("T", bound=BaseModel)
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class HuggingFaceReasoningModel:
    """Coordinate hugging face reasoning model behavior within the llm_models boundary."""

    provider_name = "huggingface"

    def __init__(
        self,
        settings: HuggingFaceSettings,
        *,
        logger: Any | None = None,
        run_dir: Path | None = None,
    ) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (HuggingFaceSettings): Validated settings that control this operation.
            logger (Any | None): Optional run-scoped logger used only for sanitized diagnostics.
            run_dir (Path | None): Filesystem location authorized for this operation.
        """
        self.settings = settings
        self.discovery_batch_size = settings.discovery_batch_size
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
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def warmup(self) -> None:
        """Warm up warmup."""
        self._load_components()

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
            max_attempts (int): Hard ceiling for structured generation attempts.

        Returns:
            T: The typed result produced by the operation.

        Raises:
            RuntimeError: If validated inputs or required dependencies cannot satisfy the contract.
            ModelOutputError: If validated inputs or required dependencies cannot satisfy the
            contract.
        """
        self.last_response_path = None
        if not self.settings.reasoning_model:
            raise RuntimeError("Hugging Face reasoning model is not configured")
        self._structured_call_count += 1
        call_index = self._structured_call_count
        attempt_limit = max(1, min(max_attempts, 2))
        first_completion = self._generate_completion(prompt, system_message=system_message)
        try:
            parsed = self._parse_completion(first_completion, schema)
        except (ValueError, ValidationError) as first_error:
            self._record_response(
                prompt=prompt,
                completion=first_completion,
                attempt=1,
                parse_failed=True,
                error=first_error,
                operation=operation,
                request_id=request_id,
                max_attempts=attempt_limit,
                call_index=call_index,
            )
            if attempt_limit == 1:
                raise ModelOutputError(
                    "Hugging Face model output could not be validated within the attempt limit: "
                    f"{sanitized_exception_summary(first_error)}; "
                    f"diagnostic={sanitized_output_preview(first_completion)}"
                ) from first_error
            retry_prompt = _build_retry_prompt(prompt, first_error)
        else:
            self._record_response(
                prompt=prompt,
                completion=first_completion,
                attempt=1,
                parse_failed=False,
                error=None,
                operation=operation,
                request_id=request_id,
                max_attempts=attempt_limit,
                call_index=call_index,
            )
            return parsed
        retry_completion = self._generate_completion(
            retry_prompt,
            system_message=system_message,
        )
        try:
            parsed = self._parse_completion(retry_completion, schema)
        except (ValueError, ValidationError) as retry_error:
            self._record_response(
                prompt=prompt,
                completion=retry_completion,
                attempt=2,
                parse_failed=True,
                error=retry_error,
                operation=operation,
                request_id=request_id,
                max_attempts=attempt_limit,
                call_index=call_index,
            )
            raise ModelOutputError(
                "Hugging Face model output could not be validated after retry: "
                f"{sanitized_exception_summary(retry_error)}; "
                f"diagnostic={sanitized_output_preview(retry_completion)}"
            ) from retry_error
        self._record_response(
            prompt=prompt,
            completion=retry_completion,
            attempt=2,
            parse_failed=False,
            error=None,
            operation=operation,
            request_id=request_id,
            max_attempts=attempt_limit,
            call_index=call_index,
        )
        return parsed

    def _generate_completion(self, prompt: str, *, system_message: str) -> str:
        """Generate completion.

        Args:
            prompt (str): Prompt text passed only to the selected model provider and never logged.
            system_message (str): Operation-scoped system instruction for this request.

        Returns:
            str: The typed result produced by the operation.
        """
        tokenizer, model = self._load_components()
        rendered_prompt = _build_chat_prompt(tokenizer, prompt, system_message)
        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        inputs = _move_inputs_to_model(inputs, model)
        generate_kwargs: dict[str, Any] = {
            **dict(inputs),
            "max_new_tokens": self.settings.max_new_tokens,
            "do_sample": False,
        }
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generate_kwargs["pad_token_id"] = pad_token_id
        output_ids = model.generate(**generate_kwargs)
        prompt_token_count = _prompt_token_count(inputs["input_ids"])
        completion_ids = output_ids[0][prompt_token_count:]
        return cast(str, tokenizer.decode(completion_ids, skip_special_tokens=True))

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
            max_attempts (int): Hard ceiling for structured generation attempts.
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

        if not parse_failed and not self.settings.log_llm_responses:
            return

        response_path: Path | None = None
        if self.run_dir is not None:
            response_path = self.persist_last_response()

        if self.logger is None:
            return

        if parse_failed:
            self.logger.warning(
                "Hugging Face response failed structured parsing",
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
                "Hugging Face response diagnostic captured",
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
                title=f"Hugging Face raw response attempt={attempt}",
                body=completion,
                also_stderr=self.settings.log_llm_responses,
            )

    def _load_components(self) -> tuple[Any, Any]:
        """Load components within the authorized project and version scope.

        Returns:
            tuple[Any, Any]: The typed result produced by the operation.
        """
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        transformers = import_module("transformers")
        kwargs = self._from_pretrained_kwargs()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.settings.reasoning_model,
            **kwargs,
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.settings.reasoning_model,
            **kwargs,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def _from_pretrained_kwargs(self) -> dict[str, Any]:
        """Execute the from pretrained kwargs operation within its declared architectural boundary.

        Returns:
            dict[str, Any]: The typed result produced by the operation.
        """
        kwargs: dict[str, Any] = {"local_files_only": self.settings.offline}
        if self.settings.token:
            kwargs["token"] = self.settings.token
        return kwargs


class HuggingFaceEmbeddingModel:
    """Coordinate hugging face embedding model behavior within the llm_models boundary."""

    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (HuggingFaceSettings): Validated settings that control this operation.
        """
        self.settings = settings
        self.embedding_fingerprint = f"hf:{settings.embedding_model}"
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"local_files_only": settings.offline}
        if settings.token:
            kwargs["token"] = settings.token
        self.model = SentenceTransformer(settings.embedding_model, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Execute the embed documents operation within its declared architectural boundary.

        Args:
            texts (list[str]): Texts required by the operation's typed contract.

        Returns:
            list[list[float]]: The typed result produced by the operation.
        """
        encoded = self.model.encode(texts, normalize_embeddings=True)
        return cast(list[list[float]], encoded.tolist())


class HuggingFaceRerankerModel:
    """Coordinate hugging face reranker model behavior within the llm_models boundary."""

    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        """Execute the init operation within its declared architectural boundary.

        Args:
            settings (HuggingFaceSettings): Validated settings that control this operation.
        """
        self.settings = settings
        self.reranker_fingerprint = f"hf:{settings.reranker_model}"
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {"local_files_only": settings.offline}
        if settings.token:
            kwargs["token"] = settings.token
        self.model = CrossEncoder(settings.reranker_model, **kwargs)

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        """Execute the rerank operation within its declared architectural boundary.

        Args:
            query (str): Input text processed in memory and excluded from diagnostic logs.
            documents (list[str]): Documents required by the operation's typed contract.

        Returns:
            list[int]: The typed result produced by the operation.
        """
        if not documents:
            return []
        scores = self.model.predict([(query, document) for document in documents])
        score_values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return sorted(
            range(len(documents)),
            key=lambda index: float(score_values[index]),
            reverse=True,
        )


def _build_chat_prompt(tokenizer: Any, prompt: str, system_message: str) -> str:
    """Build chat prompt.

    Args:
        tokenizer (Any): Tokenizer required by the operation's typed contract.
        prompt (str): Prompt text passed only to the selected model provider and never logged.
        system_message (str): System message required by the operation's typed contract.

    Returns:
        str: The typed result produced by the operation.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]
    chat_template = getattr(tokenizer, "chat_template", None)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if chat_template and callable(apply_chat_template):
        return cast(
            str,
            apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
    return f"System: {system_message}\n\nUser: {prompt}\n\nAssistant:"


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


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    """Execute the move inputs to model operation within its declared architectural boundary.

    Args:
        inputs (Any): Inputs required by the operation's typed contract.
        model (Any): Provider-neutral model adapter used by the operation.

    Returns:
        Any: The typed result produced by the operation.
    """
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _prompt_token_count(input_ids: Any) -> int:
    """Execute the prompt token count operation within its declared architectural boundary.

    Args:
        input_ids (Any): Input ids required by the operation's typed contract.

    Returns:
        int: The typed result produced by the operation.
    """
    return len(input_ids[0])


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
