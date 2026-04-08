from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer


class AfmoeTokenizer:
    def __init__(self, repo_dir: str | Path):
        repo_dir = Path(repo_dir)
        self._tokenizer = Tokenizer.from_file(str(repo_dir / "tokenizer.json"))
        config = json.loads((repo_dir / "tokenizer_config.json").read_text())
        self.bos_token = config.get("bos_token", "<|begin_of_text|>")
        self.eos_token = config.get("eos_token", "<|im_end|>")
        self.pad_token = config.get("pad_token", "<|pad|>")

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def apply_chat_template(
        self, user_prompt: str, *, system_prompt: str | None = None
    ) -> str:
        parts: list[str] = []
        if system_prompt:
            parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
        parts.append(f"<|im_start|>user\n{user_prompt}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
