import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trinity.gemma_ifeval import app  # noqa: E402
from typer.testing import CliRunner  # noqa: E402


class GemmaIFEvalTests(unittest.TestCase):
    def test_save_every_must_be_positive(self) -> None:
        result = CliRunner().invoke(
            app,
            [
                "--save-every",
                "0",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--save-every must be greater than 0", result.output)

    def test_missing_controller_checkpoint_fails_fast(self) -> None:
        missing = Path("artifacts/does-not-exist.pt")

        result = CliRunner().invoke(
            app,
            [
                "--controller-checkpoint",
                str(missing),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Controller checkpoint not found", result.output)


if __name__ == "__main__":
    unittest.main()
