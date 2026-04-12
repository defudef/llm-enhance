from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(slots=True)
class PromptPoolingSoftPromptControllerConfig:
    hidden_size: int
    num_virtual_tokens: int = 8
    controller_dim: int = 256
    hidden_dim: int = 1024
    dropout: float = 0.0
    init_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


class PromptPoolingSoftPromptController(nn.Module):
    def __init__(self, config: PromptPoolingSoftPromptControllerConfig):
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
        self.output_proj = nn.Linear(
            config.controller_dim,
            config.num_virtual_tokens * config.hidden_size,
        )
        self.base_soft_prompt = nn.Parameter(
            torch.empty(
                config.num_virtual_tokens,
                config.hidden_size,
            )
        )
        nn.init.normal_(self.base_soft_prompt, mean=0.0, std=config.init_std)

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
        target_dtype = prompt_embeds.dtype
        pooled = self.pooled_prompt(prompt_embeds, attention_mask).to(
            dtype=self.base_soft_prompt.dtype
        )
        pooled = self.pool_norm(pooled)
        hidden = self.mlp(pooled)
        delta = self.output_proj(hidden).view(
            prompt_embeds.shape[0],
            self.config.num_virtual_tokens,
            self.config.hidden_size,
        )
        soft_prompt = self.base_soft_prompt.unsqueeze(0) + delta
        return soft_prompt.to(dtype=target_dtype)


def prepend_soft_prompt(
    *,
    inputs_embeds: torch.Tensor,
    soft_prompt_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    combined_embeds = torch.cat([soft_prompt_embeds, inputs_embeds], dim=1)
    if attention_mask is None:
        return combined_embeds, None
    prefix_mask = torch.ones(
        (attention_mask.shape[0], soft_prompt_embeds.shape[1]),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)
    return combined_embeds, combined_mask
