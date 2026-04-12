import json
import sys
import unittest
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from trinity.remote import build_generation_payload
from trinity.server import parse_generation_payload


class ServerTests(unittest.TestCase):
    def test_parse_generation_payload_uses_defaults(self) -> None:
        prompt, config = parse_generation_payload({"prompt": "hello"})

        self.assertEqual(prompt, "hello")
        self.assertEqual(config.max_new_tokens, 16)
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.top_k, 50)
        self.assertIsNone(config.controller_checkpoint)

    def test_parse_generation_payload_parses_controller_checkpoint(self) -> None:
        prompt, config = parse_generation_payload(
            {
                "prompt": "hello",
                "controller_checkpoint": "artifacts/controller-best.pt",
                "controller_strength": 0.5,
                "max_new_tokens": 64,
                "temperature": 0.2,
                "top_k": 20,
                "raw_prompt": True,
            }
        )

        self.assertEqual(prompt, "hello")
        self.assertEqual(
            config.controller_checkpoint,
            Path("artifacts/controller-best.pt"),
        )
        self.assertEqual(config.controller_strength, 0.5)
        self.assertEqual(config.max_new_tokens, 64)
        self.assertEqual(config.temperature, 0.2)
        self.assertEqual(config.top_k, 20)
        self.assertTrue(config.raw_prompt)

    def test_parse_generation_payload_rejects_empty_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty string"):
            parse_generation_payload({"prompt": ""})

    def test_build_generation_payload_serializes_optional_controller(self) -> None:
        payload = build_generation_payload(
            prompt="hello",
            controller_checkpoint=Path("artifacts/controller-best.pt"),
            controller_strength=0.3,
            system_prompt="be concise",
            max_new_tokens=32,
            temperature=0.1,
            top_k=10,
            raw_prompt=False,
        )

        self.assertEqual(payload["controller_checkpoint"], "artifacts/controller-best.pt")
        self.assertEqual(payload["prompt"], "hello")
        self.assertEqual(payload["system_prompt"], "be concise")
        json.dumps(payload)


if __name__ == "__main__":
    unittest.main()
