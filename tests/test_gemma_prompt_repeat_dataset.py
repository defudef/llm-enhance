import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.build_gemma_prompt_repeat_dataset import (  # noqa: E402
    PROMPTS_PER_CATEGORY,
    build_records,
    validate_records,
)


class GemmaPromptRepeatDatasetTests(unittest.TestCase):
    def test_generated_records_are_balanced_prompt_only_and_unique(self) -> None:
        records = build_records()
        counts = Counter(record["category"] for record in records)

        self.assertEqual(len(records), 12 * PROMPTS_PER_CATEGORY)
        self.assertEqual(set(counts.values()), {PROMPTS_PER_CATEGORY})
        self.assertEqual(len({record["prompt"] for record in records}), len(records))
        self.assertTrue(all("response" not in record for record in records))
        self.assertGreaterEqual(len(counts), 12)
        validate_records(records)


if __name__ == "__main__":
    unittest.main()
