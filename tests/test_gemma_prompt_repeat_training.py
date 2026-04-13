import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gemma_train_prompt_repeat_controller import (  # noqa: E402
    build_last_hidden_pair,
    build_hidden_state_pairs,
    prompt_repeat_distillation_loss,
    repeat_prompt,
    source_baseline_loss,
)
from llm_enhance.last_token_controller import (  # noqa: E402
    LastTokenHiddenStateController,
    LastTokenHiddenStateControllerConfig,
)


class FakeRuntime:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def final_prompt_hidden_state(self, *, prompt: str, system_prompt: str | None):
        del system_prompt
        self.calls.append(prompt)
        value = float(len(prompt.split()))
        return torch.tensor([[value, value + 1.0, value + 2.0]], dtype=torch.float32)


class IdentityController(torch.nn.Module):
    def forward(self, last_hidden_state):
        return last_hidden_state


class GemmaPromptRepeatTrainingTests(unittest.TestCase):
    def test_repeat_prompt_uses_blank_line_separator(self) -> None:
        self.assertEqual(repeat_prompt("hello", 1), "hello")
        self.assertEqual(repeat_prompt("hello", 2), "hello\n\nhello")

    def test_build_last_hidden_pair_uses_original_and_repeated_prompt(self) -> None:
        runtime = FakeRuntime()
        source, target = build_last_hidden_pair(
            runtime,
            {"prompt": "one two"},
            prompt_repeats=2,
        )

        self.assertTrue(torch.equal(source, torch.tensor([[2.0, 3.0, 4.0]])))
        self.assertTrue(torch.equal(target, torch.tensor([[4.0, 5.0, 6.0]])))
        self.assertEqual(runtime.calls, ["one two", "one two\n\none two"])

    def test_build_hidden_state_pairs_caches_vectors_once(self) -> None:
        runtime = FakeRuntime()
        pairs = build_hidden_state_pairs(
            runtime,
            [
                {"prompt": "one two", "category": "a"},
                {"prompt": "three", "category": "b"},
            ],
            prompt_repeats=2,
        )

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs.categories, ["a", "b"])
        self.assertEqual(tuple(pairs.sources.shape), (2, 3))
        self.assertEqual(tuple(pairs.targets.shape), (2, 3))
        self.assertEqual(len(runtime.calls), 4)

    def test_source_baseline_loss_is_finite(self) -> None:
        pairs = build_hidden_state_pairs(
            FakeRuntime(),
            [{"prompt": "one two"}],
            prompt_repeats=2,
        )

        loss = source_baseline_loss(pairs, cosine_weight=0.1)

        self.assertIsNotNone(loss)
        self.assertGreaterEqual(loss, 0.0)

    def test_prompt_repeat_distillation_loss_is_finite(self) -> None:
        loss = prompt_repeat_distillation_loss(
            IdentityController(),
            source_last_hidden=torch.tensor([[1.0, 2.0, 3.0]]),
            target_last_hidden=torch.tensor([[1.5, 2.5, 3.5]]),
            cosine_weight=0.1,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)

    def test_last_token_controller_preserves_hidden_shape(self) -> None:
        controller = LastTokenHiddenStateController(
            LastTokenHiddenStateControllerConfig(
                hidden_size=3,
                controller_dim=2,
                hidden_dim=4,
            )
        )
        hidden = torch.randn(2, 3)

        transformed = controller(hidden)

        self.assertEqual(tuple(transformed.shape), (2, 3))

    def test_last_token_controller_zero_init_starts_as_identity(self) -> None:
        controller = LastTokenHiddenStateController(
            LastTokenHiddenStateControllerConfig(
                hidden_size=3,
                controller_dim=2,
                hidden_dim=4,
                zero_init_output=True,
            )
        )
        hidden = torch.randn(2, 3)

        transformed = controller(hidden)

        self.assertTrue(torch.equal(transformed, hidden))


if __name__ == "__main__":
    unittest.main()
