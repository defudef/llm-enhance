import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_enhance.gemma_runtime import (  # noqa: E402
    GemmaSoftPromptRuntime,
    build_messages,
    build_training_example,
    prepend_per_layer_inputs,
    sample_next_token,
)


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

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        assert add_special_tokens is False
        input_ids = [ord(ch) % 17 for ch in text]
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(input_ids), dtype=torch.long),
            }
        return {"input_ids": input_ids}

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class FakeOutput:
    def __init__(self, *, last_hidden_state, past_key_values):
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values


class FakeLanguageModel:
    def __init__(self):
        self.calls = []

    def get_per_layer_inputs(self, input_ids, _):
        return torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            1,
            3,
            dtype=torch.float32,
        )

    def __call__(
        self,
        *,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        use_cache=None,
        return_dict=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "input_ids_shape": None if input_ids is None else tuple(input_ids.shape),
                "inputs_embeds_shape": (
                    None if inputs_embeds is None else tuple(inputs_embeds.shape)
                ),
                "attention_mask_shape": (
                    None if attention_mask is None else tuple(attention_mask.shape)
                ),
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "return_dict": return_dict,
            }
        )
        if input_ids is not None:
            seq_len = input_ids.shape[1]
        else:
            seq_len = inputs_embeds.shape[1]
        call_index = len(self.calls) - 1
        hidden = torch.full((1, seq_len, 3), float(call_index))
        return FakeOutput(last_hidden_state=hidden, past_key_values=f"cache-{call_index}")


class FakeEmbedding(torch.nn.Module):
    def forward(self, input_ids):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 3)


class FakeLmHead(torch.nn.Module):
    def forward(self, hidden_states):
        call_index = int(hidden_states[0, -1, 0].item())
        logits = torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 100)
        logits[:, :, 2 if call_index == 0 else 99] = 1.0
        return logits


class FakeLastTokenController(torch.nn.Module):
    def forward(self, last_hidden_state):
        return torch.ones_like(last_hidden_state)


class FakeTextConfig:
    final_logit_softcapping = None


class FakeConfig:
    def get_text_config(self):
        return FakeTextConfig()


class FakeGemmaModel:
    def __init__(self):
        self.model = type("ModelContainer", (), {"language_model": FakeLanguageModel()})()
        self.lm_head = FakeLmHead()
        self.config = FakeConfig()
        self.embedding = FakeEmbedding()

    def get_input_embeddings(self):
        return self.embedding


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

    def test_prepend_per_layer_inputs_adds_zero_prefix(self) -> None:
        per_layer_inputs = torch.arange(1 * 3 * 2 * 4, dtype=torch.float16).view(1, 3, 2, 4)

        combined = prepend_per_layer_inputs(
            per_layer_inputs=per_layer_inputs,
            prefix_length=2,
        )

        self.assertEqual(tuple(combined.shape), (1, 5, 2, 4))
        self.assertTrue(torch.equal(combined[:, :2], torch.zeros(1, 2, 2, 4, dtype=torch.float16)))
        self.assertTrue(torch.equal(combined[:, 2:], per_layer_inputs))

    def test_sample_next_token_greedy_returns_argmax(self) -> None:
        logits = torch.tensor([[1.0, 3.0, 2.0]])

        token = sample_next_token(logits, temperature=0.0, top_k=0)

        self.assertEqual(int(token.item()), 1)

    def test_generate_reuses_kv_cache_after_prefill(self) -> None:
        runtime = object.__new__(GemmaSoftPromptRuntime)
        runtime.device = torch.device("cpu")
        runtime.tokenizer = FakeTokenizer()
        runtime.model = FakeGemmaModel()

        response = runtime.generate(
            prompt="hello",
            system_prompt=None,
            controller=None,
            controller_strength=0.0,
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
        )

        language_model = runtime.model.model.language_model
        self.assertEqual(response, "2")
        self.assertEqual(len(language_model.calls), 2)
        self.assertEqual(language_model.calls[0]["input_ids_shape"][1], len("user:hello | assistant:"))
        self.assertIsNone(language_model.calls[0]["past_key_values"])
        self.assertEqual(language_model.calls[1]["input_ids_shape"], (1, 1))
        self.assertEqual(language_model.calls[1]["past_key_values"], "cache-0")
        self.assertTrue(all(call["use_cache"] for call in language_model.calls))

    def test_generate_can_steer_first_logits_with_last_token_controller(self) -> None:
        runtime = object.__new__(GemmaSoftPromptRuntime)
        runtime.device = torch.device("cpu")
        runtime.tokenizer = FakeTokenizer()
        runtime.model = FakeGemmaModel()

        response = runtime.generate(
            prompt="hello",
            system_prompt=None,
            controller=None,
            controller_strength=1.0,
            last_token_controller=FakeLastTokenController(),
            max_new_tokens=4,
            temperature=0.0,
            top_k=0,
        )

        language_model = runtime.model.model.language_model
        self.assertEqual(response, "")
        self.assertEqual(len(language_model.calls), 1)
        self.assertEqual(language_model.calls[0]["input_ids_shape"][1], len("user:hello | assistant:"))


if __name__ == "__main__":
    unittest.main()
