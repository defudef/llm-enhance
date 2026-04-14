import re
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_enhance.gemma_ifeval import app, build_response_record, repeat_prompt  # noqa: E402
from typer.testing import CliRunner  # noqa: E402


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class GemmaIFEvalTests(unittest.TestCase):
    def test_repeat_prompt_leaves_single_prompt_unchanged(self) -> None:
        self.assertEqual(repeat_prompt("follow this", 1), "follow this")

    def test_repeat_prompt_duplicates_prompt_with_blank_line_separator(self) -> None:
        self.assertEqual(
            repeat_prompt("follow this", 2),
            "follow this\n\nfollow this",
        )

    def test_response_record_keeps_generation_prompt_when_repeated(self) -> None:
        self.assertEqual(
            build_response_record(
                prompt="follow this",
                generation_prompt="follow this\n\nfollow this",
                response="done",
            ),
            {
                "prompt": "follow this",
                "response": "done",
                "generation_prompt": "follow this\n\nfollow this",
            },
        )

    def test_save_every_must_be_positive(self) -> None:
        result = CliRunner().invoke(
            app,
            [
                "--save-every",
                "0",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--save-every must be greater than 0", strip_ansi(result.output))

    def test_prompt_repeats_must_be_positive(self) -> None:
        result = CliRunner().invoke(
            app,
            [
                "--prompt-repeats",
                "0",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--prompt-repeats must be greater than 0", strip_ansi(result.output))

    def test_controller_only_cannot_be_combined_with_base_only(self) -> None:
        result = CliRunner().invoke(
            app,
            [
                "--controller-only",
                "--base-only",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "--controller-only cannot be combined with --base-only",
            strip_ansi(result.output),
        )

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
