from __future__ import annotations

from pathlib import Path
from typing import Any

from .gemma_runtime import DEFAULT_GEMMA_MODEL_ID


def count_tokens(tokenizer: Any, text: str) -> int | None:
    tokenized = None
    if callable(tokenizer):
        try:
            tokenized = tokenizer(text, add_special_tokens=False)
        except TypeError:
            try:
                tokenized = tokenizer(text)
            except Exception:
                tokenized = None
        except Exception:
            tokenized = None
    if tokenized is None and hasattr(tokenizer, "encode"):
        try:
            tokenized = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            try:
                tokenized = tokenizer.encode(text)
            except Exception:
                return None
        except Exception:
            return None
    if isinstance(tokenized, dict):
        input_ids = tokenized.get("input_ids")
    else:
        input_ids = getattr(tokenized, "input_ids", tokenized)
    if input_ids is None:
        return None
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    return len(input_ids)


def build_mlx_prompt(tokenizer: Any, *, prompt: str, system_prompt: str | None) -> str:
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except AttributeError:
        return prompt


class MlxLastTokenController:
    def __init__(self, state_dict: dict[str, Any], *, eps: float = 1e-5) -> None:
        import mlx.core as mx

        self.eps = eps
        self.input_norm_weight = mx.array(
            state_dict["input_norm.weight"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.input_norm_bias = mx.array(
            state_dict["input_norm.bias"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.linear0_weight = mx.array(
            state_dict["mlp.0.weight"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.linear0_bias = mx.array(
            state_dict["mlp.0.bias"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.linear1_weight = mx.array(
            state_dict["mlp.3.weight"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.linear1_bias = mx.array(
            state_dict["mlp.3.bias"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.linear2_weight = mx.array(
            state_dict["mlp.5.weight"].detach().cpu().numpy(),
            dtype=mx.float32,
        )
        self.linear2_bias = mx.array(
            state_dict["mlp.5.bias"].detach().cpu().numpy(),
            dtype=mx.float32,
        )

    def _linear(self, x, weight, bias):
        import mlx.core as mx

        return x @ mx.transpose(weight) + bias

    def __call__(self, last_hidden_state):
        import mlx.core as mx
        import mlx.nn as nn

        target_dtype = last_hidden_state.dtype
        hidden = last_hidden_state.astype(mx.float32)
        mean = mx.mean(hidden, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(hidden - mean), axis=-1, keepdims=True)
        normalized = (hidden - mean) * mx.rsqrt(variance + self.eps)
        normalized = normalized * self.input_norm_weight + self.input_norm_bias
        delta = self._linear(normalized, self.linear0_weight, self.linear0_bias)
        delta = nn.gelu(delta)
        delta = self._linear(delta, self.linear1_weight, self.linear1_bias)
        delta = nn.gelu(delta)
        delta = self._linear(delta, self.linear2_weight, self.linear2_bias)
        return (hidden + delta).astype(target_dtype)


def load_mlx_last_token_controller_checkpoint(
    checkpoint_path: Path,
) -> MlxLastTokenController:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return MlxLastTokenController(checkpoint["controller_state_dict"])


class GemmaMlxRuntime:
    def __init__(
        self,
        *,
        model_id: str = DEFAULT_GEMMA_MODEL_ID,
        revision: str | None = None,
        local_dir: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx_lm is not installed in this Python environment. On macOS, "
                "run `uv sync` so the darwin-only mlx-lm dependency is installed."
            ) from exc

        model_source = local_dir or model_id
        tokenizer_config = None
        if trust_remote_code:
            tokenizer_config = {"trust_remote_code": True}
        if tokenizer_config is None:
            self.model, self.tokenizer = load(model_source, revision=revision)
        else:
            self.model, self.tokenizer = load(
                model_source,
                revision=revision,
                tokenizer_config=tokenizer_config,
            )

    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt_text = build_mlx_prompt(
            self.tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        sampler = make_sampler(temp=temperature, top_k=top_k)
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt_text,
            max_tokens=max_new_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()

    def generate_with_last_token_controller(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        controller: MlxLastTokenController,
        controller_strength: float,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        import mlx.core as mx
        from mlx_lm.sample_utils import make_sampler

        prompt_text = build_mlx_prompt(
            self.tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        add_special_tokens = self.tokenizer.bos_token is None or not prompt_text.startswith(
            self.tokenizer.bos_token
        )
        prompt_tokens = self.tokenizer.encode(
            prompt_text,
            add_special_tokens=add_special_tokens,
        )
        prompt_array = mx.array(prompt_tokens)
        cache = self.model.make_cache()
        hidden_states = self.model.language_model.model(
            prompt_array[None],
            cache=cache,
        )
        source_last_hidden = hidden_states[:, -1, :]
        controlled_last_hidden = controller(source_last_hidden)
        steered_last_hidden = source_last_hidden + controller_strength * (
            controlled_last_hidden - source_last_hidden
        )
        logits = self.logits_from_last_hidden_state(steered_last_hidden)
        sampler = make_sampler(temp=temperature, top_k=top_k)
        generated_tokens: list[int] = []
        eos_token_ids = set(self.tokenizer.eos_token_ids)

        for _ in range(max_new_tokens):
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            next_token = sampler(logprobs)
            mx.eval(next_token)
            token_id = int(next_token.item())
            if token_id in eos_token_ids:
                break
            generated_tokens.append(token_id)
            logits = self.model(next_token[None], cache=cache)[:, -1, :]

        if not generated_tokens:
            return ""
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def logits_from_last_hidden_state(self, last_hidden_state):
        from mlx_lm.models.gemma4_text import logit_softcap

        language_model = self.model.language_model
        if language_model.tie_word_embeddings:
            logits = language_model.model.embed_tokens.as_linear(
                last_hidden_state[:, None, :]
            ).squeeze(1)
        else:
            logits = language_model.lm_head(last_hidden_state[:, None, :]).squeeze(1)
        if language_model.final_logit_softcapping is not None:
            logits = logit_softcap(language_model.final_logit_softcapping, logits)
        return logits
