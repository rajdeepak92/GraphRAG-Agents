"""Azure OpenAI adapters."""

from __future__ import annotations

import json
from importlib import import_module
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from multi_agentic_graph_rag.config.settings import AzureOpenAISettings

T = TypeVar("T", bound=BaseModel)


class AzureOpenAIReasoningModel:
    provider_name = "azure_openai"

    def __init__(self, settings: AzureOpenAISettings) -> None:
        self.settings = settings

    def generate_structured(self, *, prompt: str, schema: type[T]) -> T:
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
                {"role": "system", "content": "Return only valid JSON for the requested schema."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        return schema.model_validate(json.loads(content))


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
