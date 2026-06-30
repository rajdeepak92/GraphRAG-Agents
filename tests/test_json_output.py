from __future__ import annotations

import unittest

from multi_agentic_graph_rag.llm_models.json_output import parse_json_object


class JsonOutputTests(unittest.TestCase):
    def test_clean_json(self) -> None:
        self.assertEqual(parse_json_object('{"chunks": []}'), {"chunks": []})

    def test_fenced_json(self) -> None:
        raw = '```json\n{"chunks": []}\n```'

        self.assertEqual(parse_json_object(raw), {"chunks": []})

    def test_prompt_echo_uses_final_valid_object(self) -> None:
        raw = (
            'Sample output: {"chunks": [{"chunk_id": "EXAMPLE", "facts": []}]} '
            'Actual output: {"chunks": []}'
        )

        self.assertEqual(parse_json_object(raw), {"chunks": []})

    def test_think_text_and_invalid_leading_braces_are_ignored(self) -> None:
        raw = '<think>{"not": "the answer"}</think>\n{not valid}\n{"chunks": []}'

        self.assertEqual(parse_json_object(raw), {"chunks": []})

    def test_invalid_output_raises_clean_value_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON object"):
            parse_json_object("no json here")

    def test_incomplete_json_is_rejected_with_specific_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete_json"):
            parse_json_object('{"chunks": [')


if __name__ == "__main__":
    unittest.main()
