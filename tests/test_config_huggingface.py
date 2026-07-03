from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from multi_agentic_graph_rag.config.config_loader import load_config
from multi_agentic_graph_rag.config.huggingface_env import HF_TOKEN_ALIASES
from multi_agentic_graph_rag.domain.errors import ConfigurationError
from multi_agentic_graph_rag.llm_models.factory import (
    create_embedding_model,
    create_reasoning_model,
    create_reranker_model,
)


class HuggingFaceConfigTests(unittest.TestCase):
    def test_hf_token_alias_is_preferred_and_exported(self) -> None:
        settings, env = self._load(
            "\n".join(
                [
                    "HF_TOKEN=primary-token",
                    "HUGGINGFACE_TOKEN=legacy-token",
                    "HUGGINGFACE_OFFLINE=true",
                    "HUGGINGFACE_MAX_NEW_TOKENS=77",
                    "HUGGINGFACE_DISCOVERY_BATCH_SIZE=3",
                ]
            )
        )

        self.assertEqual(settings.huggingface.token, "primary-token")
        self.assertTrue(settings.huggingface.offline)
        self.assertEqual(settings.huggingface.max_new_tokens, 77)
        self.assertEqual(settings.discovery.batch_size, 3)
        self.assertEqual(settings.huggingface.discovery_batch_size, 3)
        for alias in HF_TOKEN_ALIASES:
            self.assertEqual(env[alias], "primary-token")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_shared_discovery_batch_size_overrides_legacy_huggingface_alias(self) -> None:
        settings, _ = self._load(
            "\n".join(
                [
                    "DISCOVERY_BATCH_SIZE=4",
                    "HUGGINGFACE_DISCOVERY_BATCH_SIZE=3",
                    "LOG_LLM_RESPONSES=true",
                ]
            )
        )

        self.assertEqual(settings.discovery.batch_size, 4)
        self.assertEqual(settings.huggingface.discovery_batch_size, 4)
        self.assertTrue(settings.discovery.log_llm_responses)
        self.assertTrue(settings.huggingface.log_llm_responses)

    def test_test_scenario_settings_are_loaded_from_env(self) -> None:
        settings, _ = self._load(
            "\n".join(
                [
                    "TEST_SCENARIO_TOP_K=5",
                    "TEST_SCENARIO_DENSE_K=9",
                    "TEST_SCENARIO_SPARSE_K=7",
                    "TEST_SCENARIO_NEIGHBOR_WINDOW=2",
                    "TEST_SCENARIO_MAX_NEW_TOKENS=1024",
                ]
            )
        )

        self.assertEqual(settings.test_scenario.top_k, 5)
        self.assertEqual(settings.test_scenario.dense_k, 9)
        self.assertEqual(settings.test_scenario.sparse_k, 7)
        self.assertEqual(settings.test_scenario.neighbor_window, 2)
        self.assertEqual(settings.test_scenario.max_new_tokens, 1024)

    def test_legacy_huggingface_token_alias_is_recognized(self) -> None:
        settings, env = self._load("HUGGINGFACE_TOKEN=legacy-token")

        self.assertEqual(settings.huggingface.token, "legacy-token")
        for alias in HF_TOKEN_ALIASES:
            self.assertEqual(env[alias], "legacy-token")

    def test_default_stack_is_huggingface_postgres_neo4j(self) -> None:
        settings, _ = self._load("")

        self.assertEqual(settings.reasoning_model.provider, "huggingface")
        self.assertEqual(settings.embedding_model.provider, "huggingface")
        self.assertEqual(settings.reranker_model.provider, "huggingface")
        self.assertEqual(settings.huggingface.reasoning_model, "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertEqual(settings.huggingface.embedding_model, "BAAI/bge-m3")
        self.assertEqual(settings.huggingface.reranker_model, "BAAI/bge-reranker-base")
        self.assertEqual(settings.huggingface.max_new_tokens, 4096)
        self.assertEqual(settings.discovery.batch_size, 1)
        self.assertEqual(settings.huggingface.discovery_batch_size, 1)
        self.assertEqual(settings.postgres.mode, "postgres")
        self.assertEqual(settings.neo4j.mode, "neo4j")

    def test_local_providers_are_rejected_by_factories(self) -> None:
        settings, _ = self._load(
            "\n".join(
                [
                    "REASONING_MODEL_PROVIDER=local_heuristic",
                    "EMBEDDING_MODEL_PROVIDER=local_hash",
                    "RERANKER_MODEL_PROVIDER=none",
                ]
            )
        )

        with self.assertRaises(ConfigurationError):
            create_reasoning_model(settings)
        with self.assertRaises(ConfigurationError):
            create_embedding_model(settings)
        with self.assertRaises(ConfigurationError):
            create_reranker_model(settings)

    def _load(self, dotenv: str) -> tuple[object, dict[str, str]]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env").write_text(dotenv, encoding="utf-8")
            with patch.dict(os.environ, {"PROJECT_ROOT": str(root)}, clear=True):
                settings = load_config()
                return settings, dict(os.environ)


if __name__ == "__main__":
    unittest.main()
