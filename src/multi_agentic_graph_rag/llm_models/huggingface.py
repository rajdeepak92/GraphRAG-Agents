"""Hugging Face model adapters."""

from __future__ import annotations

import re
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementDiscovery
from multi_agentic_graph_rag.config.settings import HuggingFaceSettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.llm_models.json_output import (
    parse_json_object,
    sanitized_output_preview,
)

T = TypeVar("T", bound=BaseModel)
_SAFE_RESPONSE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_DEFAULT_SYSTEM_MESSAGE = PromptRequirementDiscovery.SYS_PROMPT_REQUIREMENT_DISCOVERY.value


class HuggingFaceReasoningModel:
    provider_name = "huggingface"

    def __init__(
        self,
        settings: HuggingFaceSettings,
        *,
        logger: Any | None = None,
        run_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.discovery_batch_size = settings.discovery_batch_size
        self.logger = logger
        self.run_dir = run_dir
        self.last_response_path: Path | None = None
        self._last_completion: str | None = None
        self._last_prompt: str | None = None
        self._last_parse_attempt = 0
        self._response_context: dict[str, Any] = {}
        self._system_message = _DEFAULT_SYSTEM_MESSAGE
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def warmup(self) -> None:
        self._load_components()

    def set_system_message(self, text: str) -> None:
        self._system_message = text

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
        if not self.settings.reasoning_model:
            raise RuntimeError("Hugging Face reasoning model is not configured")
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
                "Hugging Face model output could not be validated after retry: "
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
        tokenizer, model = self._load_components()
        rendered_prompt = _build_chat_prompt(tokenizer, prompt, self._system_message)
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

        if not parse_failed and not self.settings.log_llm_responses:
            return

        response_path: Path | None = None
        if self.run_dir is not None:
            response_path = self.persist_last_response()

        if self.logger is None:
            return

        if parse_failed:
            self.logger.info(
                "Raw Hugging Face response failed structured parsing",
                step="discover_requirements.llm_response",
                attempt=attempt,
                response_path=str(response_path) if response_path else None,
                error=str(error) if error else None,
                raw_response=completion,
                status="failed",
            )
        else:
            self.logger.info(
                "Raw Hugging Face response captured",
                step="discover_requirements.llm_response",
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
        kwargs: dict[str, Any] = {"local_files_only": self.settings.offline}
        if self.settings.token:
            kwargs["token"] = self.settings.token
        return kwargs


class HuggingFaceEmbeddingModel:
    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        self.settings = settings
        self.embedding_fingerprint = f"hf:{settings.embedding_model}"
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {"local_files_only": settings.offline}
        if settings.token:
            kwargs["token"] = settings.token
        self.model = SentenceTransformer(settings.embedding_model, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        encoded = self.model.encode(texts, normalize_embeddings=True)
        return cast(list[list[float]], encoded.tolist())


class HuggingFaceRerankerModel:
    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        self.settings = settings
        self.reranker_fingerprint = f"hf:{settings.reranker_model}"
        from sentence_transformers import CrossEncoder

        kwargs: dict[str, Any] = {"local_files_only": settings.offline}
        if settings.token:
            kwargs["token"] = settings.token
        self.model = CrossEncoder(settings.reranker_model, **kwargs)

    def rerank(self, query: str, documents: list[str]) -> list[int]:
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
    return (
        f"{prompt}\n\n"
        "The previous response failed validation. "
        "Return one corrected JSON object only.\n"
        f"Validation error: {error}"
    )


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def _prompt_token_count(input_ids: Any) -> int:
    return len(input_ids[0])


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
