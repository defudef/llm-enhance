from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .controller import PromptPoolingController
from .model import AfmoeForCausalLM, CausalLMOutput, PastKeyValues


@dataclass
class TrinityControllerOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    controller_router_biases: torch.Tensor | None = None
    past_key_values: PastKeyValues | None = None


class TrinityWithController(nn.Module):
    def __init__(
        self,
        base_model: AfmoeForCausalLM,
        controller: PromptPoolingController,
        *,
        freeze_base_model: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.controller = controller
        if freeze_base_model:
            self.freeze_base_model()

    def freeze_base_model(self) -> None:
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad = False

    def get_controller_prompt_embeds(
        self,
        controller_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.base_model.model.get_input_embeddings()(controller_input_ids)

    def compute_controller_router_biases(
        self,
        controller_input_ids: torch.Tensor,
        controller_attention_mask: torch.Tensor | None = None,
        controller_strength: float = 1.0,
    ) -> torch.Tensor:
        if controller_strength == 0:
            return torch.zeros(
                (
                    controller_input_ids.shape[0],
                    self.controller.config.num_moe_layers,
                    self.controller.config.num_experts,
                ),
                device=controller_input_ids.device,
                dtype=torch.float32,
            )

        prompt_embeds = self.get_controller_prompt_embeds(controller_input_ids)
        controller_dtype = next(self.controller.parameters()).dtype
        prompt_embeds = prompt_embeds.to(dtype=controller_dtype)
        return controller_strength * self.controller(
            prompt_embeds, controller_attention_mask
        ).to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        controller_input_ids: torch.Tensor | None = None,
        controller_attention_mask: torch.Tensor | None = None,
        controller_strength: float = 1.0,
        labels: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> TrinityControllerOutput:
        if controller_input_ids is None:
            controller_input_ids = input_ids
        router_biases = self.compute_controller_router_biases(
            controller_input_ids,
            controller_attention_mask=controller_attention_mask,
            controller_strength=controller_strength,
        )
        output: CausalLMOutput = self.base_model(
            input_ids,
            controller_router_biases=router_biases,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                output.logits.reshape(-1, output.logits.shape[-1]).to(torch.float32),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return TrinityControllerOutput(
            logits=output.logits,
            loss=loss,
            controller_router_biases=router_biases,
            past_key_values=output.past_key_values,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        controller_input_ids: torch.Tensor | None = None,
        controller_attention_mask: torch.Tensor | None = None,
        controller_strength: float = 1.0,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 50,
        eos_token_id: int | None = None,
        token_callback=None,
    ) -> torch.Tensor:
        if controller_input_ids is None:
            controller_input_ids = input_ids
        router_biases = self.compute_controller_router_biases(
            controller_input_ids,
            controller_attention_mask=controller_attention_mask,
            controller_strength=controller_strength,
        )
        return self.base_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=eos_token_id,
            token_callback=token_callback,
            controller_router_biases=router_biases,
        )
