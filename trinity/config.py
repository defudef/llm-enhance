from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AfmoeConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_dense_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int
    initializer_range: float
    rms_norm_eps: float
    use_cache: bool
    rope_theta: float
    rope_scaling: dict | None
    num_experts: int
    num_experts_per_tok: int
    num_shared_experts: int
    num_expert_groups: int
    num_limited_groups: int
    score_func: str
    route_norm: bool
    route_scale: float
    global_attn_every_n_layers: int
    sliding_window: int
    mup_enabled: bool
    layer_types: list[str]
    attention_dropout: float
    n_group: int
    topk_group: int
    dtype: str = "bfloat16"

    @classmethod
    def from_dict(cls, data: dict) -> "AfmoeConfig":
        layer_types = data.get("layer_types")
        if layer_types is None:
            every = data["global_attn_every_n_layers"]
            layer_types = [
                "sliding_attention" if (i + 1) % every else "full_attention"
                for i in range(data["num_hidden_layers"])
            ]
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            moe_intermediate_size=data["moe_intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_dense_layers=data["num_dense_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            hidden_act=data["hidden_act"],
            max_position_embeddings=data["max_position_embeddings"],
            initializer_range=data.get("initializer_range", 0.02),
            rms_norm_eps=data["rms_norm_eps"],
            use_cache=data.get("use_cache", True),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            rope_scaling=data.get("rope_scaling"),
            num_experts=data["num_experts"],
            num_experts_per_tok=data["num_experts_per_tok"],
            num_shared_experts=data["num_shared_experts"],
            num_expert_groups=data.get("num_expert_groups", 1),
            num_limited_groups=data.get("num_limited_groups", 1),
            score_func=data.get("score_func", "sigmoid"),
            route_norm=data.get("route_norm", True),
            route_scale=float(data.get("route_scale", 1.0)),
            global_attn_every_n_layers=data["global_attn_every_n_layers"],
            sliding_window=data["sliding_window"],
            mup_enabled=data.get("mup_enabled", False),
            layer_types=layer_types,
            attention_dropout=float(data.get("attention_dropout", 0.0)),
            n_group=data.get("n_group", 1),
            topk_group=data.get("topk_group", 1),
            dtype=data.get("dtype", "bfloat16"),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "AfmoeConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))
