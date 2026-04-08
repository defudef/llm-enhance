from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import torch
import typer
from torch.optim import AdamW

from trinity import AfmoeConfig, AfmoeForCausalLM, AfmoeTokenizer
from trinity import ensure_local_repo, load_checkpoint_into_model
from trinity.controller import (
    PromptPoolingController,
    PromptPoolingControllerConfig,
)
from trinity.wrapper import TrinityWithController

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def build_training_example(
    tokenizer: AfmoeTokenizer,
    *,
    prompt: str,
    response: str,
    system_prompt: str | None,
    eos_token_id: int | None,
) -> tuple[list[int], list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(prompt, system_prompt=system_prompt)
    prompt_ids = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(response)
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


def save_controller_checkpoint(
    path: Path,
    controller: PromptPoolingController,
    optimizer: AdamW,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "controller_config": controller.config.to_dict(),
            "controller_state_dict": controller.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
        },
        path,
    )


@app.command()
def train(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="JSONL dataset with prompt and response fields."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(help="Where to save the trained controller checkpoint."),
    ] = Path("artifacts/controller.pt"),
    repo_id: Annotated[
        str, typer.Option(help="Base Trinity repo on Hugging Face.")
    ] = "arcee-ai/Trinity-Nano-Preview",
    cache_dir: Annotated[
        str | None, typer.Option(help="Optional Hugging Face cache directory.")
    ] = None,
    local_dir: Annotated[
        str | None, typer.Option(help="Optional local directory for model files.")
    ] = None,
    offline: Annotated[
        bool, typer.Option(help="Use only locally cached model files.")
    ] = False,
    device: Annotated[
        str, typer.Option(help="Device override: auto, cuda, mps, or cpu.")
    ] = "auto",
    dtype: Annotated[
        str,
        typer.Option(help="Parameter dtype: auto, float16, bfloat16, or float32."),
    ] = "auto",
    controller_dim: Annotated[
        int, typer.Option(help="Latent dimension inside the controller.")
    ] = 256,
    controller_hidden_dim: Annotated[
        int, typer.Option(help="Hidden size of the controller MLP.")
    ] = 1024,
    controller_bias_scale: Annotated[
        float, typer.Option(help="Max absolute router bias after tanh squashing.")
    ] = 1.0,
    epochs: Annotated[
        int, typer.Option(help="Number of full passes over the dataset.")
    ] = 1,
    learning_rate: Annotated[
        float, typer.Option(help="AdamW learning rate for the controller.")
    ] = 1e-4,
    weight_decay: Annotated[
        float, typer.Option(help="AdamW weight decay.")
    ] = 0.01,
    grad_accum_steps: Annotated[
        int, typer.Option(help="Gradient accumulation steps.")
    ] = 1,
    max_examples: Annotated[
        int | None, typer.Option(help="Optional limit for quick smoke runs.")
    ] = None,
) -> None:
    dataset = load_jsonl(dataset_path)
    if max_examples is not None:
        dataset = dataset[:max_examples]
    if not dataset:
        raise typer.BadParameter("Dataset is empty.")

    device_obj = resolve_device(device)
    param_dtype = resolve_dtype(dtype, device_obj)

    status("Resolving Trinity checkpoint...")
    repo_dir = ensure_local_repo(
        repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
    )
    config = AfmoeConfig.from_json_file(repo_dir / "config.json")
    tokenizer = AfmoeTokenizer(repo_dir)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)

    status(f"Initializing frozen Trinity on {device_obj.type}...")
    base_model = AfmoeForCausalLM(config, device=device_obj, dtype=param_dtype)
    load_checkpoint_into_model(base_model, repo_dir)

    controller = PromptPoolingController(
        PromptPoolingControllerConfig(
            hidden_size=config.hidden_size,
            num_moe_layers=config.num_moe_layers,
            num_experts=config.num_experts,
            controller_dim=controller_dim,
            hidden_dim=controller_hidden_dim,
            bias_scale=controller_bias_scale,
        )
    ).to(device=device_obj, dtype=param_dtype)
    model = TrinityWithController(base_model, controller, freeze_base_model=True)
    optimizer = AdamW(
        model.controller.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    global_step = 0
    status(f"Training on {len(dataset)} example(s)...")
    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for example_idx, record in enumerate(dataset, start=1):
            prompt = record["prompt"]
            response = record["response"]
            system_prompt = record.get("system_prompt")
            prompt_ids, model_input_ids, labels = build_training_example(
                tokenizer,
                prompt=prompt,
                response=response,
                system_prompt=system_prompt,
                eos_token_id=eos_token_id,
            )
            controller_input_ids = torch.tensor(
                [prompt_ids], device=device_obj, dtype=torch.long
            )
            input_ids = torch.tensor(
                [model_input_ids], device=device_obj, dtype=torch.long
            )
            label_tensor = torch.tensor(
                [labels], device=device_obj, dtype=torch.long
            )

            output = model(
                input_ids,
                controller_input_ids=controller_input_ids,
                labels=label_tensor,
            )
            if output.loss is None:
                raise RuntimeError("Expected a loss value during controller training.")
            loss = output.loss / grad_accum_steps
            loss.backward()
            total_loss += output.loss.item()

            if example_idx % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if example_idx == len(dataset) and example_idx % grad_accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            status(
                f"epoch {epoch + 1}/{epochs} example {example_idx}/{len(dataset)} loss={output.loss.item():.4f}"
            )

        avg_loss = total_loss / len(dataset)
        status(f"epoch {epoch + 1} avg_loss={avg_loss:.4f}")
        save_controller_checkpoint(output_path, model.controller, optimizer, global_step)

    status(f"Saved controller checkpoint to {output_path}")


if __name__ == "__main__":
    app()
