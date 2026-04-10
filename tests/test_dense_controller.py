import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trinity.dense_controller import (  # noqa: E402
    PromptPoolingSoftPromptController,
    PromptPoolingSoftPromptControllerConfig,
    prepend_soft_prompt,
)


class DenseControllerTests(unittest.TestCase):
    def test_soft_prompt_controller_returns_expected_shape(self) -> None:
        controller = PromptPoolingSoftPromptController(
            PromptPoolingSoftPromptControllerConfig(
                hidden_size=16,
                num_virtual_tokens=4,
                controller_dim=8,
                hidden_dim=32,
            )
        )
        prompt_embeds = torch.randn(2, 6, 16)

        soft_prompt = controller(prompt_embeds)

        self.assertEqual(tuple(soft_prompt.shape), (2, 4, 16))

    def test_soft_prompt_controller_respects_attention_mask_shape(self) -> None:
        controller = PromptPoolingSoftPromptController(
            PromptPoolingSoftPromptControllerConfig(
                hidden_size=8,
                num_virtual_tokens=3,
                controller_dim=4,
                hidden_dim=16,
            )
        )
        prompt_embeds = torch.randn(2, 5, 8)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )

        soft_prompt = controller(prompt_embeds, attention_mask)

        self.assertEqual(tuple(soft_prompt.shape), (2, 3, 8))

    def test_prepend_soft_prompt_extends_attention_mask(self) -> None:
        inputs_embeds = torch.randn(2, 5, 8)
        soft_prompt_embeds = torch.randn(2, 3, 8)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )

        combined_embeds, combined_mask = prepend_soft_prompt(
            inputs_embeds=inputs_embeds,
            soft_prompt_embeds=soft_prompt_embeds,
            attention_mask=attention_mask,
        )

        self.assertEqual(tuple(combined_embeds.shape), (2, 8, 8))
        self.assertIsNotNone(combined_mask)
        assert combined_mask is not None
        self.assertEqual(tuple(combined_mask.shape), (2, 8))
        self.assertTrue(torch.equal(combined_mask[:, :3], torch.ones(2, 3, dtype=torch.long)))

    def test_prepend_soft_prompt_without_mask_returns_none_mask(self) -> None:
        inputs_embeds = torch.randn(1, 5, 8)
        soft_prompt_embeds = torch.randn(1, 2, 8)

        combined_embeds, combined_mask = prepend_soft_prompt(
            inputs_embeds=inputs_embeds,
            soft_prompt_embeds=soft_prompt_embeds,
            attention_mask=None,
        )

        self.assertEqual(tuple(combined_embeds.shape), (1, 7, 8))
        self.assertIsNone(combined_mask)


if __name__ == "__main__":
    unittest.main()
