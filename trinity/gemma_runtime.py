from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dense_controller import (
    PromptPoolingSoftPromptController,
    PromptPoolingSoftPromptControllerConfig,
    prepend_soft_prompt,
)
from .runtime import resolve_device, resolve_dtype


DEFAULT_GEMMA_MODEL_ID = "google/gemma-4-E2B-it"


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


class GemmaSoftPromptRuntime:
    def __init__(
        self,
        *,
        model_id: str,
        cache_dir: str | None,
        local_dir: str | None,
        offline: bool,
        device: str,
        dtype: str,
    ) -> None:
        self.device = resolve_device(device.lower())
        self.param_dtype = resolve_dtype(dtype.lower(), self.device)
        self.model_id = model_id

        status(f"Loading Gemma tokenizer for {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=offline,
        )
        status(
            f"Loading Gemma base model on {self.device.type} "
            f"with {str(self.param_dtype).replace('torch.', '')}..."
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=offline,
            torch_dtype=self.param_dtype,
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

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt: str,
        system_prompt: str | None,
        controller: PromptPoolingSoftPromptController | None,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        input_ids, attention_mask, prompt_embeds, _ = self.build_inputs(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        model_kwargs: dict[str, torch.Tensor | bool | float | int] = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else 1.0,
            "top_k": top_k,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        input_length = input_ids.shape[1]
        if controller is not None:
            soft_prompt = controller(prompt_embeds, attention_mask)
            combined_embeds, combined_mask = prepend_soft_prompt(
                inputs_embeds=prompt_embeds,
                soft_prompt_embeds=soft_prompt,
                attention_mask=attention_mask,
            )
            model_kwargs["inputs_embeds"] = combined_embeds
            if combined_mask is not None:
                model_kwargs["attention_mask"] = combined_mask
            input_length = combined_embeds.shape[1]
        else:
            model_kwargs["input_ids"] = input_ids

        output_ids = self.model.generate(**model_kwargs)
        generated_ids = output_ids[0, input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
