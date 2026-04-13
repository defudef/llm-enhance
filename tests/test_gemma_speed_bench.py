import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_enhance.gemma_speed_bench import (  # noqa: E402
    BACKEND_HOOK_SUPPORT,
    count_tokens,
    ns_to_seconds,
    parse_backend_names,
    rate,
    summarize_backend_result,
)
from llm_enhance.gemma_backend import resolve_backend  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": text.split()}


class GemmaSpeedBenchTests(unittest.TestCase):
    def test_parse_backend_names_trims_and_normalizes(self) -> None:
        self.assertEqual(
            parse_backend_names(" pytorch, Ollama ,mlx "),
            ["pytorch", "ollama", "mlx"],
        )

    def test_parse_backend_names_auto_prefers_pytorch_when_cuda_available(self) -> None:
        self.assertEqual(
            parse_backend_names(
                "auto",
                platform="darwin",
                cuda_available=True,
                mps_available=True,
            ),
            ["pytorch"],
        )

    def test_parse_backend_names_auto_prefers_mlx_on_macos_mps(self) -> None:
        self.assertEqual(
            parse_backend_names(
                "auto",
                platform="darwin",
                cuda_available=False,
                mps_available=True,
            ),
            ["mlx"],
        )

    def test_parse_backend_names_auto_falls_back_to_pytorch_cpu(self) -> None:
        self.assertEqual(
            parse_backend_names(
                "auto",
                platform="linux",
                cuda_available=False,
                mps_available=False,
            ),
            ["pytorch"],
        )

    def test_resolve_backend_auto_allows_mlx_without_controller_hooks(self) -> None:
        self.assertEqual(
            resolve_backend(
                "auto",
                needs_controller_hooks=False,
                platform="darwin",
                cuda_available=False,
                mps_available=True,
            ),
            "mlx",
        )

    def test_resolve_backend_auto_falls_back_for_hook_only_controllers(self) -> None:
        self.assertEqual(
            resolve_backend(
                "auto",
                needs_controller_hooks=True,
                platform="darwin",
                cuda_available=False,
                mps_available=True,
            ),
            "pytorch",
        )

    def test_resolve_backend_rejects_explicit_mlx_for_hook_only_controllers(self) -> None:
        with self.assertRaises(Exception):
            resolve_backend("mlx", needs_controller_hooks=True)

    def test_parse_backend_names_rejects_unknown_backend(self) -> None:
        with self.assertRaises(Exception):
            parse_backend_names("pytorch,vllm")

    def test_parse_backend_names_rejects_empty_selection(self) -> None:
        with self.assertRaises(Exception):
            parse_backend_names(" , ")

    def test_count_tokens_uses_tokenizer_input_ids(self) -> None:
        self.assertEqual(count_tokens(FakeTokenizer(), "one two three"), 3)

    def test_rate_handles_zero_seconds(self) -> None:
        self.assertIsNone(rate(10, 0.0))
        self.assertAlmostEqual(rate(10, 2.0), 5.0)

    def test_ns_to_seconds_converts_ollama_durations(self) -> None:
        self.assertEqual(ns_to_seconds(None), None)
        self.assertAlmostEqual(ns_to_seconds(1_500_000_000), 1.5)

    def test_summarize_backend_result_aggregates_prompt_metrics(self) -> None:
        summary = summarize_backend_result(
            backend="ollama",
            status_value="ok",
            load_seconds=0.5,
            prompt_results=[
                {
                    "seconds": 2.0,
                    "prompt_tokens": 10,
                    "generated_tokens": 20,
                },
                {
                    "seconds": 3.0,
                    "prompt_tokens": 15,
                    "generated_tokens": 30,
                },
            ],
        )

        self.assertEqual(summary["prompts"], 2)
        self.assertEqual(summary["prompt_tokens"], 25)
        self.assertEqual(summary["generated_tokens"], 50)
        self.assertAlmostEqual(summary["seconds_per_prompt"], 2.5)
        self.assertAlmostEqual(summary["generated_tokens_per_second_wall"], 10.0)
        self.assertEqual(summary["hook_support"], BACKEND_HOOK_SUPPORT["ollama"])

    def test_hook_matrix_records_required_backends(self) -> None:
        self.assertEqual(set(BACKEND_HOOK_SUPPORT), {"pytorch", "ollama", "mlx"})
        self.assertEqual(BACKEND_HOOK_SUPPORT["pytorch"]["hidden_states"], "yes")
        self.assertEqual(BACKEND_HOOK_SUPPORT["ollama"]["hidden_states"], "no_public_api")


if __name__ == "__main__":
    unittest.main()
