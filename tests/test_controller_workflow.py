import sys
import unittest
from pathlib import Path

from typer.testing import CliRunner

sys.path.append(str(Path(__file__).resolve().parents[1]))

import infer_controller
from train_controller import select_best_metric


class ControllerWorkflowTests(unittest.TestCase):
    def test_select_best_metric_falls_back_to_train_loss(self) -> None:
        self.assertEqual(select_best_metric(train_loss=1.25, val_loss=None), 1.25)

    def test_select_best_metric_prefers_validation_loss(self) -> None:
        self.assertEqual(select_best_metric(train_loss=1.25, val_loss=0.75), 0.75)

    def test_infer_requires_existing_controller_checkpoint(self) -> None:
        missing_checkpoint = Path(self.id()).with_suffix(".pt")

        result = CliRunner().invoke(
            infer_controller.app,
            [
                "prompt",
                "--controller-checkpoint",
                str(missing_checkpoint),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "Controller checkpoint not found at",
            result.output,
        )
        self.assertIn(
            str(missing_checkpoint),
            result.output,
        )


if __name__ == "__main__":
    unittest.main()
