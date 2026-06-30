"""Azure OpenAI adapters."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.config.settings import AzureOpenAISettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.llm_models.json_output import (
    parse_json_object,
    sanitized_output_preview,
)

T = TypeVar("T", bound=BaseModel)
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class AzureOpenAIReasoningModel:
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

    def set_response_context(
        self,
        *,
        batch_index: int,
        attempt: int,
        chunk_ids: list[str],
    ) -> None:
        self._response_context = {
            "batch_index": batch_index,
            "validation_attempt": attempt,
            "chunk_ids": chunk_ids,
        }
        self.last_response_path = None

    def clear_response_context(self) -> None:
        self._response_context = {}

    def persist_last_response(self, *, filename: str | None = None) -> Path | None:
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
                f"{retry_error}; preview={sanitized_output_preview(retry_completion)}"
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
        if not all(
            [
                self.settings.endpoint,
                self.settings.api_key,
                self.settings.reasoning_deployment,
            ]
        ):
            raise RuntimeError("Azure OpenAI reasoning is not configured")

        AzureOpenAI = cast(Any, import_module("openai")).AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=self.settings.endpoint,
            api_key=self.settings.api_key,
            api_version=self.settings.api_version,
        )
        response = client.chat.completions.create(
            model=self.settings.reasoning_deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a requirement discovery engine. Return only valid JSON "
                        "for the requested chunk-local schema."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return cast(str, response.choices[0].message.content or "{}")

    def _parse_completion(self, completion: str, schema: type[T]) -> T:
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
            self.logger.info(
                "Raw Azure OpenAI response failed structured parsing",
                step="discover_requirements.llm_response",
                attempt=attempt,
                response_path=str(response_path) if response_path else None,
                error=str(error) if error else None,
                raw_response=completion,
                status="failed",
            )
        else:
            self.logger.info(
                "Raw Azure OpenAI response captured",
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
    provider_name = "azure_openai"

    def __init__(self, settings: AzureOpenAISettings) -> None:
        self.settings = settings
        self.embedding_fingerprint = f"azure:{settings.embedding_deployment}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
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


def _build_retry_prompt(prompt: str, error: Exception) -> str:
    return (
        f"{prompt}\n\n"
        "The previous response failed validation. "
        "Return one corrected JSON object only.\n"
        f"Validation error: {error}"
    )


def _extract_chunks(prompt: str) -> list[tuple[str, str]]:
    matches = re.findall(r"<chunk id=\"([^\"]+)\">\n(.*?)\n</chunk>", prompt, flags=re.S)
    return [(chunk_id, text.strip()) for chunk_id, text in matches]


def _response_filename(
    prompt: str,
    attempt: int,
    context: dict[str, Any] | None = None,
) -> str:
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
