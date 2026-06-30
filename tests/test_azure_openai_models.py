from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from multi_agentic_graph_rag.config.settings import AzureOpenAISettings
from multi_agentic_graph_rag.domain.errors import ModelOutputError
from multi_agentic_graph_rag.domain.schemas import RequirementDiscoveryChunkOutput
from multi_agentic_graph_rag.llm_models.azure_openai import AzureOpenAIReasoningModel


class AzureOpenAIModelTests(unittest.TestCase):
    def test_failed_reasoning_output_is_persisted_per_parse_attempt(self) -> None:
        settings = AzureOpenAISettings(
            endpoint="https://example.openai.azure.com",
            api_key="key",
            reasoning_deployment="deployment",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model = _FailingAzureReasoningModel(
                settings,
                discovery_batch_size=1,
                run_dir=Path(temp_dir),
            )
            model.set_response_context(batch_index=3, attempt=1, chunk_ids=["CHUNK-3"])

            with self.assertRaises(ModelOutputError):
                model.generate_structured(
                    prompt="extract requirements",
                    schema=RequirementDiscoveryChunkOutput,
                )

            first = Path(temp_dir) / "llm_response_3_1.txt"
            second = Path(temp_dir) / "llm_response_3_1_parse2.txt"
            self.assertEqual(first.read_text(encoding="utf-8"), '{"facts": [')
            self.assertEqual(second.read_text(encoding="utf-8"), '{"facts": [')


class _FailingAzureReasoningModel(AzureOpenAIReasoningModel):
    def _generate_completion(self, prompt: str) -> str:
        return '{"facts": ['


if __name__ == "__main__":
    unittest.main()
