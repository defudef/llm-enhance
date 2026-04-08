from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


BASE_ALLOW_PATTERNS = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]

WEIGHT_ALLOW_PATTERNS = [
    "model.safetensors.index.json",
    "model-*.safetensors",
]


def ensure_local_repo(
    repo_id: str,
    cache_dir: str | None = None,
    *,
    include_weights: bool = True,
    local_dir: str | None = None,
    offline: bool = False,
) -> Path:
    allow_patterns = [*BASE_ALLOW_PATTERNS]
    if include_weights:
        allow_patterns.extend(WEIGHT_ALLOW_PATTERNS)
    return Path(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            local_files_only=offline,
        )
    )


def load_checkpoint_into_model(
    model: torch.nn.Module,
    repo_dir: str | Path,
    *,
    strict: bool = True,
) -> None:
    repo_dir = Path(repo_dir)
    index_path = repo_dir / "model.safetensors.index.json"
    index_data = json.loads(index_path.read_text())
    weight_map = index_data["weight_map"]
    model_tensors = model.state_dict()

    missing_in_checkpoint = set(model_tensors)
    unexpected_in_checkpoint: list[str] = []

    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    for shard_name, keys in shard_to_keys.items():
        shard_path = repo_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for key in keys:
                if key not in model_tensors:
                    unexpected_in_checkpoint.append(key)
                    continue
                target = model_tensors[key]
                tensor = shard.get_tensor(key)
                if tensor.shape != target.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: checkpoint {tuple(tensor.shape)} != model {tuple(target.shape)}"
                    )
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                missing_in_checkpoint.discard(key)

    if strict and (missing_in_checkpoint or unexpected_in_checkpoint):
        problems: list[str] = []
        if missing_in_checkpoint:
            problems.append(f"missing keys: {sorted(missing_in_checkpoint)[:10]}")
        if unexpected_in_checkpoint:
            problems.append(f"unexpected keys: {unexpected_in_checkpoint[:10]}")
        raise RuntimeError("; ".join(problems))
