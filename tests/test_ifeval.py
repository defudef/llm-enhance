import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from instruction_following_eval import evaluation_lib
from trinity.ifeval import build_accuracy_report


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


if __name__ == "__main__":
    unittest.main()
