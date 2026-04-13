from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dense_controller import (
    PromptPoolingSoftPromptController,
    PromptPoolingSoftPromptControllerConfig,
    prepend_soft_prompt,
)
from .device import resolve_device, resolve_dtype
from .last_token_controller import (
    LastTokenHiddenStateController,
    LastTokenHiddenStateControllerConfig,
)


DEFAULT_GEMMA_MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_GEMMA_IFEVAL_SYNTHETIC_DATASET_PATH = Path(
    "data/gemma_ifeval_synthetic_en.jsonl"
)
DEFAULT_GEMMA_PROMPT_REPEAT_DATASET_PATH = Path(
    "data/gemma_prompt_repeat_diverse_en.jsonl"
)
DEFAULT_GEMMA_IFEVAL_CONTROLLER_PATH = Path("artifacts/gemma-ifeval-alignfix-small.pt")
DEFAULT_GEMMA_IFEVAL_BEST_CONTROLLER_PATH = Path(
    "artifacts/gemma-ifeval-alignfix-small-best.pt"
)
DEFAULT_GEMMA_PROMPT_REPEAT_CONTROLLER_PATH = Path(
    "artifacts/gemma-prompt-repeat-distill.pt"
)
DEFAULT_GEMMA_PROMPT_REPEAT_BEST_CONTROLLER_PATH = Path(
    "artifacts/gemma-prompt-repeat-distill-best.pt"
)
DEFAULT_GEMMA_IFEVAL_OUTPUT_DIR = Path("artifacts/gemma-ifeval-controller")
DEFAULT_GEMMA_IFEVAL_CONTROLLER_STRENGTH = 0.05


def status(message: str) -> None:
    from typer import colors, secho

    secho(message, err=True, fg=colors.BLUE)


def load_gemma_controller_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PromptPoolingSoftPromptController:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    controller = PromptPoolingSoftPromptController(
        PromptPoolingSoftPromptControllerConfig(**checkpoint["controller_config"])
    ).to(device=device, dtype=dtype)
    controller.load_state_dict(checkpoint["controller_state_dict"])
    controller.eval()
    return controller


def load_last_token_controller_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> LastTokenHiddenStateController:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    controller = LastTokenHiddenStateController(
        LastTokenHiddenStateControllerConfig(**checkpoint["controller_config"])
    ).to(device=device, dtype=dtype)
    controller.load_state_dict(checkpoint["controller_state_dict"])
    controller.eval()
    return controller


def save_gemma_controller_checkpoint(
    path: Path,
    controller: PromptPoolingSoftPromptController,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "controller_config": controller.config.to_dict(),
            "controller_state_dict": controller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def save_last_token_controller_checkpoint(
    path: Path,
    controller: LastTokenHiddenStateController,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "controller_config": controller.config.to_dict(),
            "controller_state_dict": controller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def build_messages(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_prompt_text(
    tokenizer,
    *,
    prompt: str,
    system_prompt: str | None,
) -> str:
    messages = build_messages(prompt, system_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_training_example(
    tokenizer,
    *,
    prompt: str,
    response: str,
    system_prompt: str | None,
    max_response_tokens: int | None,
) -> tuple[list[int], list[int], list[int]]:
    prompt_text = build_prompt_text(
        tokenizer,
        prompt=prompt,
        system_prompt=system_prompt,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if max_response_tokens is not None:
        response_ids = response_ids[:max_response_tokens]
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        response_ids = [*response_ids, eos_token_id]
    full_ids = [*prompt_ids, *response_ids]
    if len(full_ids) < 2:
        raise ValueError("Training example is too short after tokenization.")
    input_ids = full_ids[:-1]
    labels = full_ids[1:]
    prompt_prefix = max(len(prompt_ids) - 1, 0)
    labels[:prompt_prefix] = [-100] * prompt_prefix
    return prompt_ids, input_ids, labels


def prepend_per_layer_inputs(
    *,
    per_layer_inputs: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    if prefix_length <= 0:
        return per_layer_inputs
    prefix = torch.zeros(
        (
            per_layer_inputs.shape[0],
            prefix_length,
            per_layer_inputs.shape[2],
            per_layer_inputs.shape[3],
        ),
        device=per_layer_inputs.device,
        dtype=per_layer_inputs.dtype,
    )
    return torch.cat([prefix, per_layer_inputs], dim=1)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled_logits = logits / temperature
    if top_k > 0:
        k = min(top_k, scaled_logits.shape[-1])
        values, _ = torch.topk(scaled_logits, k=k, dim=-1)
        cutoff = values[..., -1, None]
        scaled_logits = scaled_logits.masked_fill(scaled_logits < cutoff, float("-inf"))
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_final_logit_softcapping(logits: torch.Tensor, softcapping: float | None) -> torch.Tensor:
    if softcapping is None:
        return logits
    logits = logits / softcapping
    logits = torch.tanh(logits)
    return logits * softcapping


class GemmaSoftPromptRuntime:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str | None,
        cache_dir: str | None,
        local_dir: str | None,
        offline: bool,
        device: str,
        dtype: str,
    ) -> None:
        self.device = resolve_device(device.lower())
        self.param_dtype = resolve_dtype(dtype.lower(), self.device)
        self.model_id = model_id
        model_source = local_dir or model_id
        model_revision = None if local_dir is not None else revision
        revision_suffix = f" revision {model_revision}" if model_revision else ""

        status(f"Loading Gemma tokenizer for {model_source}{revision_suffix}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            revision=model_revision,
            local_files_only=offline,
        )
        status(
            f"Loading Gemma base model on {self.device.type} "
            f"with {str(self.param_dtype).replace('torch.', '')}..."
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            cache_dir=cache_dir,
            revision=model_revision,
            local_files_only=offline,
            dtype=self.param_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def build_inputs(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        prompt_text = build_prompt_text(
            self.tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
        )
        tokenized = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        prompt_embeds = self.model.get_input_embeddings()(input_ids)
        return input_ids, attention_mask, prompt_embeds, prompt_text

    def build_per_layer_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.language_model.get_per_layer_inputs(input_ids, None)

    def compute_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        controller: PromptPoolingSoftPromptController | None,
        controller_strength: float,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        language_model = self.model.model.language_model
        if controller is None or controller_strength <= 0:
            outputs = language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        else:
            soft_prompt = controller_strength * controller(
                prompt_embeds,
                prompt_attention_mask,
            )
            input_embeds = self.model.get_input_embeddings()(input_ids)
            combined_embeds, combined_mask = prepend_soft_prompt(
                inputs_embeds=input_embeds,
                soft_prompt_embeds=soft_prompt,
                attention_mask=attention_mask,
            )
            combined_per_layer_inputs = prepend_per_layer_inputs(
                per_layer_inputs=self.build_per_layer_inputs(input_ids),
                prefix_length=soft_prompt.shape[1],
            )
            outputs = language_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                per_layer_inputs=combined_per_layer_inputs,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.last_hidden_state
        logits = self.model.lm_head(hidden_states[:, -1:, :]).squeeze(1)
        final_logit_softcapping = self.model.config.get_text_config().final_logit_softcapping
        return apply_final_logit_softcapping(logits, final_logit_softcapping)

    def logits_from_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(hidden_states[:, -1:, :]).squeeze(1)
        final_logit_softcapping = self.model.config.get_text_config().final_logit_softcapping
        return apply_final_logit_softcapping(logits, final_logit_softcapping)

    def logits_from_last_hidden_state(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(last_hidden_state.unsqueeze(1)).squeeze(1)
        final_logit_softcapping = self.model.config.get_text_config().final_logit_softcapping
        return apply_final_logit_softcapping(logits, final_logit_softcapping)

    @torch.no_grad()
    def final_prompt_hidden_state(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
    ) -> torch.Tensor:
        input_ids, attention_mask, _, _ = self.build_inputs(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        outputs = self.model.model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, -1, :].detach()

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        controller: PromptPoolingSoftPromptController | None,
        controller_strength: float,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        last_token_controller: LastTokenHiddenStateController | None = None,
    ) -> str:
        input_ids, attention_mask, prompt_embeds, _ = self.build_inputs(
            prompt=prompt,
            system_prompt=system_prompt,
        )
        prompt_attention_mask = attention_mask.clone()
        generated_tokens: list[int] = []
        eos_token_id = self.tokenizer.eos_token_id
        language_model = self.model.model.language_model
        if controller is not None and last_token_controller is not None:
            raise ValueError("Use either soft-prompt controller or last-token controller.")

        if controller is None or controller_strength <= 0:
            outputs = language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            active_attention_mask = attention_mask
        else:
            soft_prompt = controller_strength * controller(
                prompt_embeds,
                prompt_attention_mask,
            )
            input_embeds = self.model.get_input_embeddings()(input_ids)
            combined_embeds, active_attention_mask = prepend_soft_prompt(
                inputs_embeds=input_embeds,
                soft_prompt_embeds=soft_prompt,
                attention_mask=attention_mask,
            )
            combined_per_layer_inputs = prepend_per_layer_inputs(
                per_layer_inputs=self.build_per_layer_inputs(input_ids),
                prefix_length=soft_prompt.shape[1],
            )
            outputs = language_model(
                inputs_embeds=combined_embeds,
                attention_mask=active_attention_mask,
                per_layer_inputs=combined_per_layer_inputs,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values
        if last_token_controller is None or controller_strength <= 0:
            logits = self.logits_from_hidden_states(outputs.last_hidden_state)
        else:
            source_last_hidden = outputs.last_hidden_state[:, -1, :]
            controlled_last_hidden = last_token_controller(source_last_hidden)
            steered_last_hidden = source_last_hidden + controller_strength * (
                controlled_last_hidden - source_last_hidden
            )
            logits = self.logits_from_last_hidden_state(steered_last_hidden)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
            )
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                break
            generated_tokens.append(token_id)
            next_token = next_token.to(device=input_ids.device, dtype=input_ids.dtype)
            next_mask = torch.ones(
                (active_attention_mask.shape[0], 1),
                device=active_attention_mask.device,
                dtype=active_attention_mask.dtype,
            )
            active_attention_mask = torch.cat([active_attention_mask, next_mask], dim=1)
            outputs = language_model(
                input_ids=next_token,
                attention_mask=active_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = self.logits_from_hidden_states(outputs.last_hidden_state)

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
