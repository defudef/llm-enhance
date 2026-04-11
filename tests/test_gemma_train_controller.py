import sys
import unittest
from pathlib import Path

import torch
from torch.optim import AdamW

sys.path.append(str(Path(__file__).resolve().parents[1]))

from gemma_train_controller import (  # noqa: E402
    build_mlflow_params,
    category_counts,
    normalize_mlflow_artifact_location,
    resolve_warmup_steps,
    set_optimizer_lr,
    split_train_val,
)


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

    def test_build_mlflow_params_includes_dataset_split_context(self) -> None:
        train_records = [
            {"prompt": "a", "response": "x", "category": "alpha"},
            {"prompt": "b", "response": "y", "category": "beta"},
        ]
        val_records = [{"prompt": "c", "response": "z", "category": "alpha"}]
        dataset = [*train_records, *val_records]

        params = build_mlflow_params(
            dataset_path=Path("data/sample.jsonl"),
            dataset=dataset,
            train_dataset=train_records,
            val_dataset=val_records,
            model_id="google/gemma-4-E2B-it",
            revision=None,
            cache_dir=None,
            local_dir=None,
            offline=True,
            requested_device="auto",
            requested_dtype="auto",
            actual_device=torch.device("cpu"),
            param_dtype=torch.float32,
            hidden_size=1536,
            output_path=Path("artifacts/latest.pt"),
            best_output_path=None,
            num_virtual_tokens=8,
            controller_dim=128,
            controller_hidden_dim=512,
            controller_dropout=0.05,
            epochs=3,
            learning_rate=2e-4,
            warmup_steps=24,
            effective_warmup_steps=24,
            warmup_ratio=0.03,
            min_lr_ratio=0.1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            val_split=0.2,
            seed=42,
            max_examples=None,
            max_response_tokens=64,
            patience=10,
            min_improvement=1e-3,
        )

        self.assertEqual(params["dataset.examples"], 3)
        self.assertEqual(params["dataset.train_examples"], 2)
        self.assertEqual(params["dataset.val_examples"], 1)
        self.assertEqual(params["dataset.category.alpha.examples"], 2)
        self.assertEqual(params["dataset.train_category.beta.examples"], 1)
        self.assertEqual(params["dataset.val_category.alpha.examples"], 1)
        self.assertEqual(params["model.revision"], "none")
        self.assertEqual(params["model.param_dtype"], "float32")
        self.assertEqual(params["train.effective_warmup_steps"], 24)

    def test_category_counts_uses_default_for_missing_category(self) -> None:
        self.assertEqual(
            category_counts(
                [
                    {"prompt": "a", "response": "x"},
                    {"prompt": "b", "response": "y", "category": "beta"},
                ]
            ),
            {"beta": 1, "default": 1},
        )

    def test_normalize_mlflow_artifact_location_keeps_remote_uri(self) -> None:
        self.assertEqual(
            normalize_mlflow_artifact_location("s3://bucket/mlflow"),
            "s3://bucket/mlflow",
        )

    def test_resolve_warmup_steps_uses_ratio_when_omitted(self) -> None:
        self.assertEqual(
            resolve_warmup_steps(
                warmup_steps=None,
                warmup_ratio=0.03,
                total_steps=1000,
            ),
            30,
        )
        self.assertEqual(
            resolve_warmup_steps(
                warmup_steps=24,
                warmup_ratio=0.03,
                total_steps=1000,
            ),
            24,
        )


if __name__ == "__main__":
    unittest.main()
