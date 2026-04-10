from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .config import AfmoeConfig

PastKeyValue = tuple[torch.Tensor, torch.Tensor]
PastKeyValues = list[PastKeyValue | None]


def _make_linear(
    in_features: int,
    out_features: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Linear:
    return nn.Linear(
        in_features,
        out_features,
        bias=False,
        device=device,
        dtype=dtype,
    )


def _activation(name: str):
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported activation: {name}")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def causal_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    sliding_window: int | None = None,
) -> torch.Tensor:
    q_pos = query_positions[:, None]
    k_pos = key_positions[None, :]
    visible = k_pos <= q_pos
    if sliding_window is not None:
        visible = visible & (k_pos >= (q_pos - sliding_window + 1))
    mask = torch.zeros(
        (query_positions.numel(), key_positions.numel()), device=device, dtype=dtype
    )
    mask = mask.masked_fill(~visible, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)


class AfmoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight.to(hidden_states.device) * hidden_states
        return hidden_states.to(input_dtype)


class AfmoeRotaryEmbedding(nn.Module):
    def __init__(self, config: AfmoeConfig):
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=x.device)
        freqs = torch.outer(position_ids.reshape(-1).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).view(
            *position_ids.shape, inv_freq.numel() * 2
        )
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class AfmoeMLP(nn.Module):
    def __init__(
        self,
        config: AfmoeConfig,
        *,
        intermediate_size: int | None = None,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = _make_linear(
            self.hidden_size, self.intermediate_size, device=device, dtype=dtype
        )
        self.up_proj = _make_linear(
            self.hidden_size, self.intermediate_size, device=device, dtype=dtype
        )
        self.down_proj = _make_linear(
            self.intermediate_size, self.hidden_size, device=device, dtype=dtype
        )
        self.act_fn = _activation(config.hidden_act)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class AfmoeTokenChoiceRouter(nn.Module):
    def __init__(
        self,
        config: AfmoeConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.score_func = config.score_func
        self.route_norm = config.route_norm
        self.route_scale = config.route_scale
        self.gate = _make_linear(
            config.hidden_size, config.num_experts, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_bias: torch.Tensor | None,
        controller_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        scores = self.gate(hidden_states).to(torch.float32)

        if controller_bias is not None:
            if controller_bias.ndim == 1:
                controller_bias = controller_bias.unsqueeze(0)
            controller_bias = controller_bias.to(scores.device, dtype=torch.float32)
            controller_bias = controller_bias[:, None, :].expand(
                batch_size, seq_len, -1
            )
            scores = scores + controller_bias.reshape(-1, self.num_experts)

        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores)
        else:
            scores = F.softmax(scores, dim=-1)

        if expert_bias is not None:
            _, selected_experts = torch.topk(
                scores + expert_bias.to(scores.device), k=self.top_k, dim=-1
            )
            top_scores = scores.gather(dim=-1, index=selected_experts)
        else:
            top_scores, selected_experts = torch.topk(scores, k=self.top_k, dim=-1)

        if self.score_func == "sigmoid" and self.route_norm:
            top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)

        return top_scores * self.route_scale, selected_experts


class AfmoeMoE(nn.Module):
    def __init__(self, config: AfmoeConfig, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.router = AfmoeTokenChoiceRouter(config, device=device, dtype=dtype)
        self.shared_experts = None
        if config.num_shared_experts > 0:
            self.shared_experts = AfmoeMLP(
                config,
                intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
                device=device,
                dtype=dtype,
            )
        self.experts = nn.ModuleList(
            [
                AfmoeMLP(
                    config,
                    intermediate_size=config.moe_intermediate_size,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.expert_bias = nn.Parameter(
            torch.zeros(config.num_experts, device=device, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        controller_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        top_scores, selected_experts = self.router(
            hidden_states,
            self.expert_bias,
            controller_bias=controller_bias,
        )

        if self.shared_experts is not None:
            output = self.shared_experts(hidden_states_flat)
        else:
            output = torch.zeros_like(hidden_states_flat)

        flat_scores = top_scores.reshape(-1)
        flat_experts = selected_experts.reshape(-1)
        sort_order = torch.argsort(flat_experts, stable=True)
        token_to_expert = flat_experts[sort_order]
        token_indices = sort_order // self.config.num_experts_per_tok
        routed_input = hidden_states_flat[token_indices]
        routed_output = torch.zeros_like(routed_input)
        active_experts, counts = torch.unique_consecutive(
            token_to_expert, return_counts=True
        )
        offset = 0
        for expert_id, count in zip(active_experts.tolist(), counts.tolist()):
            next_offset = offset + count
            routed_output[offset:next_offset] = self.experts[expert_id](
                routed_input[offset:next_offset]
            )
            offset = next_offset

        routed_output = (
            routed_output.to(torch.float32)
            * flat_scores[sort_order].unsqueeze(-1)
        ).to(hidden_states.dtype)
        output.index_add_(0, token_indices, routed_output)
        return output.view(batch_size, seq_len, hidden_dim)


class AfmoeAttention(nn.Module):
    def __init__(
        self, config: AfmoeConfig, layer_idx: int, *, device: torch.device, dtype: torch.dtype
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.is_local_attention = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_local_attention else None

        self.q_proj = _make_linear(
            config.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype
        )
        self.k_proj = _make_linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.v_proj = _make_linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            device=device,
            dtype=dtype,
        )
        self.o_proj = _make_linear(
            self.num_heads * self.head_dim, config.hidden_size, device=device, dtype=dtype
        )
        self.q_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AfmoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gate_proj = _make_linear(
            config.hidden_size, self.num_heads * self.head_dim, device=device, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        gate_states = self.gate_proj(hidden_states)

        query_states = self.q_norm(query_states).transpose(1, 2)
        key_states = self.k_norm(key_states).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if self.is_local_attention:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )

        if past_key_value is not None:
            past_key, past_value = past_key_value
            if self.sliding_window is not None and past_key.shape[-2] >= self.sliding_window:
                keep = self.sliding_window - 1
                past_key = past_key[:, :, -keep:, :]
                past_value = past_value[:, :, -keep:, :]
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = attn_output * torch.sigmoid(gate_states)
        return self.o_proj(attn_output), present_key_value


class AfmoeDecoderLayer(nn.Module):
    def __init__(
        self, config: AfmoeConfig, layer_idx: int, *, device: torch.device, dtype: torch.dtype
    ):
        super().__init__()
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = AfmoeAttention(
            config, layer_idx, device=device, dtype=dtype
        )
        self.input_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = AfmoeRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_mlp_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if layer_idx >= config.num_dense_layers:
            self.mlp = AfmoeMoE(config, device=device, dtype=dtype)
        else:
            self.mlp = AfmoeMLP(config, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_value: PastKeyValue | None = None,
        use_cache: bool = False,
        router_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, PastKeyValue | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        if isinstance(self.mlp, AfmoeMoE):
            hidden_states = self.mlp(hidden_states, controller_bias=router_bias)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        return residual + hidden_states, present_key_value


@dataclass
class ModelOutput:
    last_hidden_state: torch.Tensor
    past_key_values: PastKeyValues | None = None


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues | None = None


class AfmoeModel(nn.Module):
    def __init__(self, config: AfmoeConfig, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                AfmoeDecoderLayer(config, idx, device=device, dtype=dtype)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = AfmoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = AfmoeRotaryEmbedding(config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        controller_router_biases: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            assert input_ids is not None
            batch_size, seq_len = input_ids.shape
            hidden_states = self.embed_tokens(input_ids)
            device = input_ids.device
        else:
            batch_size, seq_len, _ = inputs_embeds.shape
            hidden_states = inputs_embeds
            device = inputs_embeds.device
        if self.config.mup_enabled:
            hidden_states = hidden_states * math.sqrt(self.config.hidden_size)

        position_ids = torch.arange(
            cache_position, cache_position + seq_len, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_past_key_values: PastKeyValues | None = [] if use_cache else None
        moe_layer_idx = 0

        for layer_idx, layer in enumerate(self.layers):
            layer_past = past_key_values[layer_idx]
            past_len = 0 if layer_past is None else layer_past[0].shape[-2]
            if layer.attention_type == "sliding_attention" and past_len > 0:
                past_len = min(past_len, self.config.sliding_window - 1)
            key_positions = torch.arange(
                cache_position - past_len,
                cache_position + seq_len,
                device=device,
            )
            attention_mask = causal_mask(
                position_ids[0],
                key_positions,
                device=device,
                dtype=torch.float32,
                sliding_window=(
                    self.config.sliding_window
                    if layer.attention_type == "sliding_attention"
                    else None
                ),
            )
            hidden_states, present_key_value = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
                router_bias=(
                    None
                    if controller_router_biases is None
                    or layer_idx < self.config.num_dense_layers
                    else controller_router_biases[:, moe_layer_idx, :]
                ),
            )
            if layer_idx >= self.config.num_dense_layers:
                moe_layer_idx += 1
            if next_past_key_values is not None:
                next_past_key_values.append(present_key_value)

        return ModelOutput(
            last_hidden_state=self.norm(hidden_states),
            past_key_values=next_past_key_values,
        )


class AfmoeForCausalLM(nn.Module):
    def __init__(self, config: AfmoeConfig, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.model = AfmoeModel(config, device=device, dtype=dtype)
        self.lm_head = _make_linear(
            config.hidden_size, config.vocab_size, device=device, dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        controller_router_biases: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = False,
        cache_position: int = 0,
    ) -> CausalLMOutput:
        model_output = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            controller_router_biases=controller_router_biases,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        return CausalLMOutput(
            logits=self.lm_head(model_output.last_hidden_state),
            past_key_values=model_output.past_key_values,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 50,
        eos_token_id: int | None = None,
        token_callback: Callable[[torch.Tensor], None] | None = None,
        controller_router_biases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        generated = input_ids
        output = self(
            generated,
            use_cache=True,
            cache_position=0,
            controller_router_biases=controller_router_biases,
        )
        past_key_values = output.past_key_values
        logits = output.logits[:, -1, :]

        for step in range(max_new_tokens):
            next_token = sample_next_token(
                logits, temperature=temperature, top_k=top_k
            )
            generated = torch.cat([generated, next_token], dim=1)
            if token_callback is not None:
                token_callback(next_token)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            output = self(
                next_token,
                controller_router_biases=controller_router_biases,
                past_key_values=past_key_values,
                use_cache=True,
                cache_position=generated.shape[1] - 1,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :]
        return generated


def sample_next_token(
    logits: torch.Tensor, *, temperature: float, top_k: int
) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k > 0:
        top_values, _ = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        threshold = top_values[:, [-1]]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
