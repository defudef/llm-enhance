import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.build_gemma_ifeval_synthetic import build_records, validate_records  # noqa: E402


class GemmaIFEvalSyntheticDatasetTests(unittest.TestCase):
    def test_generated_records_are_balanced_and_valid(self) -> None:
        records = build_records()
        counts = Counter(record["category"] for record in records)

        self.assertEqual(len(records), 576)
        self.assertEqual(set(counts.values()), {32})
        validate_records(records)


if __name__ == "__main__":
    unittest.main()
