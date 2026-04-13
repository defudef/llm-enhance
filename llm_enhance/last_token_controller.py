from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(slots=True)
class LastTokenHiddenStateControllerConfig:
    hidden_size: int
    controller_dim: int = 256
    hidden_dim: int = 1024
    dropout: float = 0.0
    zero_init_output: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class LastTokenHiddenStateController(nn.Module):
    def __init__(self, config: LastTokenHiddenStateControllerConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.controller_dim),
            nn.GELU(),
            nn.Linear(config.controller_dim, config.hidden_size),
        )
        if config.zero_init_output:
            output_projection = self.mlp[-1]
            if isinstance(output_projection, nn.Linear):
                nn.init.zeros_(output_projection.weight)
                nn.init.zeros_(output_projection.bias)

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        target_dtype = last_hidden_state.dtype
        normalized = self.input_norm(last_hidden_state.float())
        delta = self.mlp(normalized)
        return (last_hidden_state.float() + delta).to(dtype=target_dtype)
