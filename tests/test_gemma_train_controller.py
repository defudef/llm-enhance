import sys
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gemma_train_controller import set_optimizer_lr, split_train_val  # noqa: E402


class GemmaTrainControllerTests(unittest.TestCase):
    def test_split_train_val_preserves_each_category(self) -> None:
        records = [
            {"prompt": f"a{i}", "response": "x", "category": "alpha"}
            for i in range(6)
        ] + [
            {"prompt": f"b{i}", "response": "y", "category": "beta"}
            for i in range(6)
        ]

        train_records, val_records = split_train_val(
            records,
            val_split=0.25,
            seed=7,
        )

        self.assertTrue(any(row["category"] == "alpha" for row in train_records))
        self.assertTrue(any(row["category"] == "alpha" for row in val_records))
        self.assertTrue(any(row["category"] == "beta" for row in train_records))
        self.assertTrue(any(row["category"] == "beta" for row in val_records))

    def test_set_optimizer_lr_warmup_then_decays(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = AdamW([param], lr=1.0)

        lr0 = set_optimizer_lr(
            optimizer,
            base_lr=1e-3,
            global_step=0,
            warmup_steps=2,
            total_steps=6,
            min_lr_ratio=0.1,
        )
        lr1 = set_optimizer_lr(
            optimizer,
            base_lr=1e-3,
            global_step=1,
            warmup_steps=2,
            total_steps=6,
            min_lr_ratio=0.1,
        )
        lr5 = set_optimizer_lr(
            optimizer,
            base_lr=1e-3,
            global_step=5,
            warmup_steps=2,
            total_steps=6,
            min_lr_ratio=0.1,
        )

        self.assertAlmostEqual(lr0, 5e-4)
        self.assertAlmostEqual(lr1, 1e-3)
        self.assertLess(lr5, lr1)
        self.assertGreaterEqual(lr5, 1e-4)


if __name__ == "__main__":
    unittest.main()
