"""Hugging Face adapters and deterministic local fallbacks."""

from __future__ import annotations

import hashlib
import json
import re
from importlib import import_module
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from multi_agentic_graph_rag.config.settings import HuggingFaceSettings
from multi_agentic_graph_rag.domain.schemas import (
    LLMFactCandidate,
    LLMRequirementCandidate,
    RequirementDiscoveryOutput,
    SourceTrace,
)

T = TypeVar("T", bound=BaseModel)


class HuggingFaceReasoningModel:
    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        self.settings = settings

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        if not self.settings.reasoning_model:
            raise RuntimeError("Hugging Face reasoning model is not configured")
        pipeline = cast(Any, import_module("transformers")).pipeline

        generator = pipeline("text-generation", model=self.settings.reasoning_model)
        result = generator(prompt, max_new_tokens=1024, do_sample=False)[0]["generated_text"]
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        return schema.model_validate(json.loads(result[json_start:json_end]))


class HuggingFaceEmbeddingModel:
    provider_name = "huggingface"

    def __init__(self, settings: HuggingFaceSettings) -> None:
        self.settings = settings
        self.embedding_fingerprint = f"hf:{settings.embedding_model}"
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        encoded = self.model.encode(texts, normalize_embeddings=True)
        return cast(list[list[float]], encoded.tolist())


class HashEmbeddingModel:
    provider_name = "local_hash"
    embedding_fingerprint = "hash-384-v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_hash_vector(text) for text in texts]


class NoopRerankerModel:
    provider_name = "none"

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        return list(range(len(documents)))


class LocalHeuristicReasoningModel:
    provider_name = "local_heuristic"

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
        if schema is not RequirementDiscoveryOutput:
            raise RuntimeError("local_heuristic only supports requirement discovery")
        chunks = _extract_chunks(prompt)
        facts: list[LLMFactCandidate] = []
        requirements: list[LLMRequirementCandidate] = []
        for index, (chunk_id, text) in enumerate(chunks[:20], start=1):
            sentence = _first_meaningful_sentence(text)
            if not sentence:
                continue
            trace = SourceTrace(
                chunk_id=chunk_id,
                quote=sentence,
                start_char=text.find(sentence),
                end_char=text.find(sentence) + len(sentence),
            )
            fact_temp = f"F{index}"
            facts.append(LLMFactCandidate(temp_id=fact_temp, text=sentence, source_trace=trace))
            statement = _to_requirement(sentence)
            requirements.append(
                LLMRequirementCandidate(
                    temp_id=f"R{index}",
                    statement=statement,
                    fact_temp_ids=[fact_temp],
                    source_trace=trace,
                )
            )
        return schema.model_validate(
            RequirementDiscoveryOutput(facts=facts, requirements=requirements).model_dump()
        )


def _hash_vector(text: str, dimensions: int = 384) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimensions:
        for byte in digest:
            values.append((byte - 128) / 128.0)
            if len(values) == dimensions:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def _extract_chunks(prompt: str) -> list[tuple[str, str]]:
    matches = re.findall(r"<chunk id=\"([^\"]+)\">\n(.*?)\n</chunk>", prompt, flags=re.S)
    return [(chunk_id, text.strip()) for chunk_id, text in matches]


def _first_meaningful_sentence(text: str) -> str:
    for part in re.split(r"(?<=[.!?])\s+", text.strip()):
        sentence = part.strip()
        if len(sentence) >= 12:
            return sentence[:500]
    return text.strip()[:500]


def _to_requirement(sentence: str) -> str:
    lowered = sentence.strip()
    if lowered.lower().startswith(("the system shall", "system shall")):
        return lowered
    return f"The system shall satisfy the documented behavior: {lowered}"
