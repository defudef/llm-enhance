from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(slots=True)
class PromptPoolingControllerConfig:
    hidden_size: int
    num_moe_layers: int
    num_experts: int
    controller_dim: int = 256
    hidden_dim: int = 1024
    bias_scale: float = 1.0
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class PromptPoolingController(nn.Module):
    def __init__(self, config: PromptPoolingControllerConfig):
        super().__init__()
        self.config = config
        self.pool_norm = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.controller_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(
            config.controller_dim,
            config.num_moe_layers * config.num_experts,
        )

    def pooled_prompt(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return prompt_embeds.mean(dim=1)
        mask = attention_mask.to(prompt_embeds.device, dtype=prompt_embeds.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (prompt_embeds * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        prompt_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.pool_norm(self.pooled_prompt(prompt_embeds, attention_mask))
        hidden = self.mlp(pooled)
        raw_bias = self.head(hidden).view(
            prompt_embeds.shape[0],
            self.config.num_moe_layers,
            self.config.num_experts,
        )
        return self.config.bias_scale * torch.tanh(raw_bias)
