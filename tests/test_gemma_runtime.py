import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trinity.gemma_runtime import build_messages, build_training_example  # noqa: E402


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        assert enable_thinking is False
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages) + " | assistant:"

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(ch) % 17 for ch in text]}


class GemmaRuntimeTests(unittest.TestCase):
    def test_build_messages_includes_optional_system_prompt(self) -> None:
        messages = build_messages("hello", "be concise")
        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
        )

    def test_build_training_example_masks_prompt_positions(self) -> None:
        tokenizer = FakeTokenizer()
        prompt_ids, input_ids, labels = build_training_example(
            tokenizer,
            prompt="hello",
            response="world",
            system_prompt="be concise",
            max_response_tokens=None,
        )

        self.assertGreater(len(prompt_ids), 0)
        self.assertEqual(len(input_ids), len(labels))
        self.assertTrue(all(value == -100 for value in labels[: len(prompt_ids) - 1]))
        self.assertEqual(labels[-1], 99)

    def test_build_training_example_applies_response_cap(self) -> None:
        tokenizer = FakeTokenizer()
        _, _, labels = build_training_example(
            tokenizer,
            prompt="hello",
            response="world",
            system_prompt=None,
            max_response_tokens=2,
        )

        non_ignored = [value for value in labels if value != -100]
        self.assertEqual(len(non_ignored), 3)


if __name__ == "__main__":
    unittest.main()
