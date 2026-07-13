from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agentic_graph_rag.common_prompt_defs import PromptRequirementIdentity
from multi_agentic_graph_rag.config.settings import HuggingFaceSettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import RequirementDiscoveryChunkOutput
from multi_agentic_graph_rag.llm_models.huggingface import (
    HuggingFaceEmbeddingModel,
    HuggingFaceReasoningModel,
    HuggingFaceRerankerModel,
)


class HuggingFaceModelTests(unittest.TestCase):
    def test_sentence_transformer_receives_token_and_offline_flags(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        module = types.ModuleType("sentence_transformers")

        class FakeSentenceTransformer:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                calls.append((model_name, kwargs))

            def encode(self, texts: list[str], normalize_embeddings: bool) -> object:
                return _Encoded([[0.0] for _ in texts])

        module.SentenceTransformer = FakeSentenceTransformer
        settings = HuggingFaceSettings(
            token="token-value",
            embedding_model="embedding-model",
            offline=True,
        )

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            HuggingFaceEmbeddingModel(settings)

        self.assertEqual(
            calls, [("embedding-model", {"local_files_only": True, "token": "token-value"})]
        )

    def test_reasoning_model_receives_token_offline_and_generation_settings(self) -> None:
        tokenizer_calls: list[tuple[str, dict[str, object]]] = []
        model_calls: list[tuple[str, dict[str, object]]] = []
        generate_calls: list[dict[str, object]] = []
        module = types.ModuleType("transformers")

        class FakeTokenizer:
            chat_template = "template"
            pad_token_id = 0
            eos_token_id = 1

            @classmethod
            def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeTokenizer:
                tokenizer_calls.append((model_name, kwargs))
                return cls()

            def apply_chat_template(
                self,
                messages: list[dict[str, str]],
                *,
                tokenize: bool,
                add_generation_prompt: bool,
            ) -> str:
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return "rendered prompt"

            def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, list[list[int]]]:
                self.prompt = prompt
                self.return_tensors = return_tensors
                return {"input_ids": [[10, 11]]}

            def decode(self, ids: list[int], *, skip_special_tokens: bool) -> str:
                self.decoded_ids = ids
                self.skip_special_tokens = skip_special_tokens
                return '{"facts": []}'

        class FakeModel:
            device = None

            @classmethod
            def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeModel:
                model_calls.append((model_name, kwargs))
                return cls()

            def eval(self) -> None:
                self.evaluated = True

            def generate(self, **kwargs: object) -> list[list[int]]:
                generate_calls.append(kwargs)
                return [[10, 11, 12]]

        module.AutoTokenizer = FakeTokenizer
        module.AutoModelForCausalLM = FakeModel
        settings = HuggingFaceSettings(
            token="token-value",
            reasoning_model="reasoning-model",
            offline=True,
            max_new_tokens=13,
        )

        with patch.dict(sys.modules, {"transformers": module}):
            result = HuggingFaceReasoningModel(settings).generate_structured(
                prompt="extract requirements",
                schema=RequirementDiscoveryChunkOutput,
                system_message="discovery system",
                operation="requirement_discovery.chunk",
                request_id="CHUNK-1",
            )

        self.assertEqual(result.facts, [])
        expected_kwargs = {"local_files_only": True, "token": "token-value"}
        self.assertEqual(tokenizer_calls, [("reasoning-model", expected_kwargs)])
        self.assertEqual(model_calls, [("reasoning-model", expected_kwargs)])
        self.assertEqual(generate_calls[0]["max_new_tokens"], 13)
        self.assertIs(generate_calls[0]["do_sample"], False)

    def test_reranker_receives_token_and_offline_flags(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []
        module = types.ModuleType("sentence_transformers")

        class FakeCrossEncoder:
            def __init__(self, model_name: str, **kwargs: object) -> None:
                calls.append((model_name, kwargs))

            def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
                self.pairs = pairs
                return [0.2, 0.7]

        module.CrossEncoder = FakeCrossEncoder
        settings = HuggingFaceSettings(
            token="token-value",
            reranker_model="reranker-model",
            offline=True,
        )

        with patch.dict(sys.modules, {"sentence_transformers": module}):
            reranker = HuggingFaceRerankerModel(settings)
            order = reranker.rerank("query", ["first", "second"])

        self.assertEqual(
            calls, [("reranker-model", {"local_files_only": True, "token": "token-value"})]
        )
        self.assertEqual(order, [1, 0])

    def test_identity_retry_preserves_system_prompt_and_persists_each_attempt(self) -> None:
        settings = HuggingFaceSettings(reasoning_model="reasoning-model")
        prompt = '<chunk id="CHUNK-1">\nThe system shall import files.\n</chunk>'

        with tempfile.TemporaryDirectory() as temp_dir:
            model = _FailingReasoningModel(settings, run_dir=Path(temp_dir))

            with self.assertRaises(ModelOutputError):
                model.generate_structured(
                    prompt=prompt,
                    schema=RequirementDiscoveryChunkOutput,
                    system_message=(
                        PromptRequirementIdentity.SYS_PROMPT_REQUIREMENT_IDENTITY.value
                    ),
                    operation="requirement_identity.entailment",
                    request_id="pair-1",
                )

            first = Path(temp_dir) / (
                "llm_response_requirement_identity.entailment_pair-1_call-000001_attempt-1.txt"
            )
            second = Path(temp_dir) / (
                "llm_response_requirement_identity.entailment_pair-1_call-000001_attempt-2.txt"
            )
            self.assertEqual(first.read_text(encoding="utf-8"), '{"chunks": [')
            self.assertEqual(second.read_text(encoding="utf-8"), '{"chunks": [')
            self.assertEqual(
                model.system_messages,
                [PromptRequirementIdentity.SYS_PROMPT_REQUIREMENT_IDENTITY.value] * 2,
            )

    def test_structured_attempt_limit_can_disable_schema_repair_retry(self) -> None:
        settings = HuggingFaceSettings(reasoning_model="reasoning-model")

        with tempfile.TemporaryDirectory() as temp_dir:
            model = _FailingReasoningModel(settings, run_dir=Path(temp_dir))

            with self.assertRaises(ModelOutputError):
                model.generate_structured(
                    prompt="prompt",
                    schema=RequirementDiscoveryChunkOutput,
                    system_message="discovery system",
                    operation="requirement_discovery.chunk",
                    request_id="CHUNK-1",
                    max_attempts=1,
                )

            first = Path(temp_dir) / (
                "llm_response_requirement_discovery.chunk_CHUNK-1_call-000001_attempt-1.txt"
            )
            second = Path(temp_dir) / (
                "llm_response_requirement_discovery.chunk_CHUNK-1_call-000001_attempt-2.txt"
            )
            self.assertTrue(first.exists())
            self.assertFalse(second.exists())
            self.assertEqual(model.system_messages, ["discovery system"])


class _Encoded:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


class _FailingReasoningModel(HuggingFaceReasoningModel):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.system_messages: list[str] = []

    def _generate_completion(self, prompt: str, *, system_message: str) -> str:
        self.system_messages.append(system_message)
        return '{"chunks": ['


if __name__ == "__main__":
    unittest.main()
