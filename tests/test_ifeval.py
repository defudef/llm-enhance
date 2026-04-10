import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from instruction_following_eval import evaluation_lib
from trinity.ifeval import build_accuracy_report, persist_ifeval_progress


class IFEvalTests(unittest.TestCase):
    def test_official_checker_passes_simple_followed_response(self) -> None:
        example = evaluation_lib.InputExample(
            key=1,
            instruction_id_list=[
                "punctuation:no_comma",
                "change_case:english_lowercase",
            ],
            prompt="answer in lowercase english without commas",
            kwargs=[{}, {}],
        )

        output = evaluation_lib.test_instruction_following_strict(
            example,
            {
                example.prompt: (
                    "this response is entirely in lowercase english words without commas"
                )
            },
        )

        self.assertTrue(output.follow_all_instructions)
        self.assertEqual(output.follow_instruction_list, [True, True])

    def test_official_checker_fails_simple_broken_response(self) -> None:
        example = evaluation_lib.InputExample(
            key=1,
            instruction_id_list=[
                "punctuation:no_comma",
                "change_case:english_lowercase",
            ],
            prompt="answer in lowercase english without commas",
            kwargs=[{}, {}],
        )

        output = evaluation_lib.test_instruction_following_strict(
            example,
            {example.prompt: "Hello, World"},
        )

        self.assertFalse(output.follow_all_instructions)
        self.assertEqual(output.follow_instruction_list, [False, False])

    def test_build_accuracy_report_aggregates_prompt_and_instruction_scores(self) -> None:
        outputs = [
            evaluation_lib.OutputExample(
                instruction_id_list=["punctuation:no_comma", "change_case:english_lowercase"],
                prompt="p1",
                response="hello world",
                follow_all_instructions=True,
                follow_instruction_list=[True, True],
            ),
            evaluation_lib.OutputExample(
                instruction_id_list=["punctuation:no_comma"],
                prompt="p2",
                response="Hello, World",
                follow_all_instructions=False,
                follow_instruction_list=[False],
            ),
        ]

        report = build_accuracy_report(outputs)

        self.assertEqual(report["prompt_total"], 2)
        self.assertEqual(report["prompt_correct"], 1)
        self.assertAlmostEqual(report["prompt_accuracy"], 0.5)
        self.assertEqual(report["instruction_total"], 3)
        self.assertEqual(report["instruction_correct"], 2)
        self.assertAlmostEqual(report["instruction_accuracy"], 2 / 3)
        self.assertAlmostEqual(report["tier0"]["punctuation"]["accuracy"], 0.5)
        self.assertAlmostEqual(
            report["tier1"]["change_case:english_lowercase"]["accuracy"],
            1.0,
        )

    def test_persist_ifeval_progress_writes_partial_summary_and_progress(self) -> None:
        example = evaluation_lib.InputExample(
            key=1,
            instruction_id_list=["punctuation:no_comma"],
            prompt="answer without commas",
            kwargs=[{}],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            progress = persist_ifeval_progress(
                inputs=[example],
                completed=1,
                records=[{"prompt": example.prompt, "response": "hello world"}],
                prompt_to_response={example.prompt: "hello world"},
                output_dir=output_dir,
                total_examples=5,
                elapsed_seconds=12.0,
                write_partial_eval=True,
            )

            self.assertEqual(progress["completed_examples"], 1)
            self.assertAlmostEqual(progress["avg_seconds_per_example"], 12.0)
            self.assertEqual(progress["eta_human"], "48s")
            self.assertTrue((output_dir / "responses.jsonl").exists())
            self.assertTrue((output_dir / "summary.partial.json").exists())
            self.assertTrue((output_dir / "progress.json").exists())


if __name__ == "__main__":
    unittest.main()
