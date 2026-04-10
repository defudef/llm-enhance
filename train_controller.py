from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.optim import AdamW

from trinity import AfmoeConfig, AfmoeForCausalLM, AfmoeTokenizer
from trinity import ensure_local_repo, load_checkpoint_into_model
from trinity.controller import (
    PromptPoolingController,
    PromptPoolingControllerConfig,
)
from trinity.wrapper import TrinityWithController

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console(stderr=True)


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


def format_train_metrics(
    *,
    train_loss: float | None,
    bias_l2: float | None,
    lr: float,
    skipped_steps: int,
) -> str:
    parts: list[str] = []
    if train_loss is not None:
        parts.append(f"loss={train_loss:.4f}")
    if bias_l2 is not None:
        parts.append(f"bias={bias_l2:.2e}")
    parts.append(f"lr={lr:.2e}")
    if skipped_steps:
        parts.append(f"skip={skipped_steps}")
    return " ".join(parts)


def format_epoch_summary(
    *,
    train_loss: float,
    train_bias_l2: float,
    val_loss: float | None,
    val_bias_l2: float | None,
    lr: float,
    finite_examples: int,
    skipped_steps: int,
) -> str:
    parts = [
        f"train={train_loss:.4f}",
        f"bias={train_bias_l2:.2e}",
    ]
    if val_loss is None or val_bias_l2 is None:
        parts.append("val=n/a")
    else:
        parts.append(f"val={val_loss:.4f}")
        parts.append(f"val_bias={val_bias_l2:.2e}")
    parts.extend(
        [
            f"lr={lr:.2e}",
            f"finite={finite_examples}",
            f"skipped={skipped_steps}",
        ]
    )
    return " ".join(parts)


def select_best_metric(*, train_loss: float, val_loss: float | None) -> float:
    return train_loss if val_loss is None else val_loss


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


def split_train_val(
    records: list[dict],
    *,
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if val_split <= 0 or len(records) < 2:
        return records, []
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_split)))
    val_count = min(val_count, len(shuffled) - 1)
    return shuffled[val_count:], shuffled[:val_count]


def build_training_example(
    tokenizer: AfmoeTokenizer,
    *,
    prompt: str,
    response: str,
    system_prompt: str | None,
    eos_token_id: int | None,
    max_response_tokens: int | None,
) -> tuple[list[int], list[int], list[int]]:
    prompt_text = tokenizer.apply_chat_template(prompt, system_prompt=system_prompt)
    prompt_ids = tokenizer.encode(prompt_text)
    response_ids = tokenizer.encode(response)
    if max_response_tokens is not None:
        response_ids = response_ids[:max_response_tokens]
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


def set_optimizer_lr(
    optimizer: AdamW,
    *,
    base_lr: float,
    global_step: int,
    warmup_steps: int,
) -> float:
    if warmup_steps <= 0:
        lr = base_lr
    else:
        lr = base_lr * min(1.0, (global_step + 1) / warmup_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def build_tensors(
    tokenizer: AfmoeTokenizer,
    record: dict,
    *,
    eos_token_id: int | None,
    device: torch.device,
    max_response_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids, model_input_ids, labels = build_training_example(
        tokenizer,
        prompt=record["prompt"],
        response=record["response"],
        system_prompt=record.get("system_prompt"),
        eos_token_id=eos_token_id,
        max_response_tokens=max_response_tokens,
    )
    controller_input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    input_ids = torch.tensor([model_input_ids], device=device, dtype=torch.long)
    label_tensor = torch.tensor([labels], device=device, dtype=torch.long)
    return controller_input_ids, input_ids, label_tensor


@torch.no_grad()
def evaluate_loss(
    model: TrinityWithController,
    tokenizer: AfmoeTokenizer,
    dataset: list[dict],
    *,
    eos_token_id: int | None,
    device: torch.device,
    router_bias_l2: float,
    max_response_tokens: int | None,
) -> tuple[float | None, float | None]:
    if not dataset:
        return None, None

    was_training = model.controller.training
    model.controller.eval()
    total_loss = 0.0
    total_bias_l2 = 0.0
    finite_count = 0

    for record in dataset:
        controller_input_ids, input_ids, label_tensor = build_tensors(
            tokenizer,
            record,
            eos_token_id=eos_token_id,
            device=device,
            max_response_tokens=max_response_tokens,
        )
        output = model(
            input_ids,
            controller_input_ids=controller_input_ids,
            labels=label_tensor,
        )
        if output.loss is None or output.controller_router_biases is None:
            continue
        bias_penalty = output.controller_router_biases.pow(2).mean()
        val_loss = output.loss + router_bias_l2 * bias_penalty
        if not torch.isfinite(val_loss):
            continue
        total_loss += output.loss.item()
        total_bias_l2 += bias_penalty.item()
        finite_count += 1

    if was_training:
        model.controller.train()

    if finite_count == 0:
        return None, None
    return total_loss / finite_count, total_bias_l2 / finite_count


@app.command()
def train(
    dataset_path: Annotated[
        Path,
        typer.Argument(
            help="JSONL dataset with prompt and response fields.",
        ),
    ] = Path("data/sarcastic_en.jsonl"),
    output_path: Annotated[
        Path,
        typer.Option(help="Where to save the latest controller checkpoint."),
    ] = Path("artifacts/controller.pt"),
    best_output_path: Annotated[
        Path | None,
        typer.Option(
            help="Optional checkpoint path updated only when validation/train metric improves."
        ),
    ] = Path("artifacts/controller-best.pt"),
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
    ] = "float32",
    controller_dim: Annotated[
        int, typer.Option(help="Latent dimension inside the controller.")
    ] = 256,
    controller_hidden_dim: Annotated[
        int, typer.Option(help="Hidden size of the controller MLP.")
    ] = 1024,
    controller_bias_scale: Annotated[
        float, typer.Option(help="Max absolute router bias after tanh squashing.")
    ] = 0.02,
    epochs: Annotated[
        int, typer.Option(help="Number of full passes over the dataset.")
    ] = 1,
    learning_rate: Annotated[
        float, typer.Option(help="AdamW learning rate for the controller.")
    ] = 1e-6,
    warmup_steps: Annotated[
        int, typer.Option(help="Linear learning-rate warmup over optimizer steps.")
    ] = 20,
    weight_decay: Annotated[
        float, typer.Option(help="AdamW weight decay.")
    ] = 0.01,
    max_grad_norm: Annotated[
        float, typer.Option(help="Clip controller gradient norm to this value.")
    ] = 1.0,
    router_bias_l2: Annotated[
        float, typer.Option(help="L2 penalty on controller router biases.")
    ] = 1e-4,
    val_dataset_path: Annotated[
        Path | None,
        typer.Option(help="Optional JSONL validation dataset. If omitted, --val-split is used."),
    ] = None,
    val_split: Annotated[
        float,
        typer.Option(help="Validation split from train data when no validation file is provided."),
    ] = 0.2,
    seed: Annotated[
        int, typer.Option(help="Random seed used for train/validation split.")
    ] = 42,
    grad_accum_steps: Annotated[
        int, typer.Option(help="Gradient accumulation steps.")
    ] = 1,
    max_examples: Annotated[
        int | None, typer.Option(help="Optional limit for quick smoke runs.")
    ] = None,
    max_response_tokens: Annotated[
        int | None,
        typer.Option(help="Optional cap on response tokens used for loss/training."),
    ] = None,
    eval_every: Annotated[
        int,
        typer.Option(
            help="Run validation every N epochs. Use 0 to skip validation during training."
        ),
    ] = 1,
) -> None:
    if epochs <= 0:
        raise typer.BadParameter("--epochs must be greater than 0.")
    if grad_accum_steps <= 0:
        raise typer.BadParameter("--grad-accum-steps must be greater than 0.")
    if max_response_tokens is not None and max_response_tokens <= 0:
        raise typer.BadParameter("--max-response-tokens must be greater than 0.")
    if eval_every < 0:
        raise typer.BadParameter("--eval-every must be >= 0.")

    dataset = load_jsonl(dataset_path)
    if max_examples is not None:
        dataset = dataset[:max_examples]
    if not dataset:
        raise typer.BadParameter("Dataset is empty.")
    if val_dataset_path is not None:
        train_dataset = dataset
        val_dataset = load_jsonl(val_dataset_path)
    else:
        train_dataset, val_dataset = split_train_val(
            dataset,
            val_split=val_split,
            seed=seed,
        )
    if not train_dataset:
        raise typer.BadParameter("Training split is empty.")

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
    ).to(device=device_obj, dtype=torch.float32)
    model = TrinityWithController(base_model, controller, freeze_base_model=True)
    optimizer = AdamW(
        model.controller.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    global_step = 0
    skipped_steps = 0
    current_lr = 0.0
    best_metric: float | None = None
    status(
        f"Training on {len(train_dataset)} example(s), validating on {len(val_dataset)} example(s)..."
    )
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[metrics]}"),
        console=console,
    )
    with progress:
        for epoch in range(epochs):
            total_loss = 0.0
            total_bias_l2 = 0.0
            finite_examples = 0
            last_train_loss: float | None = None
            last_bias_l2: float | None = None
            epoch_skipped_start = skipped_steps
            optimizer.zero_grad(set_to_none=True)
            task_id = progress.add_task(
                f"epoch {epoch + 1}/{epochs}",
                total=len(train_dataset),
                metrics=format_train_metrics(
                    train_loss=None,
                    bias_l2=None,
                    lr=current_lr,
                    skipped_steps=0,
                ),
            )
            for example_idx, record in enumerate(train_dataset, start=1):
                controller_input_ids, input_ids, label_tensor = build_tensors(
                    tokenizer,
                    eos_token_id=eos_token_id,
                    device=device_obj,
                    record=record,
                    max_response_tokens=max_response_tokens,
                )

                output = model(
                    input_ids,
                    controller_input_ids=controller_input_ids,
                    labels=label_tensor,
                )
                if output.loss is None:
                    raise RuntimeError("Expected a loss value during controller training.")
                if output.controller_router_biases is None:
                    raise RuntimeError("Expected controller router biases during training.")

                bias_penalty = output.controller_router_biases.pow(2).mean()
                train_loss = output.loss + router_bias_l2 * bias_penalty
                if not torch.isfinite(train_loss):
                    skipped_steps += 1
                    optimizer.zero_grad(set_to_none=True)
                    progress.console.print(
                        f"epoch {epoch + 1}/{epochs} example {example_idx}/{len(train_dataset)} skipped non-finite loss={train_loss.item()}",
                        style="yellow",
                    )
                    progress.update(
                        task_id,
                        advance=1,
                        metrics=format_train_metrics(
                            train_loss=last_train_loss,
                            bias_l2=last_bias_l2,
                            lr=current_lr,
                            skipped_steps=skipped_steps - epoch_skipped_start,
                        ),
                    )
                    continue

                loss = train_loss / grad_accum_steps
                loss.backward()
                total_loss += output.loss.item()
                total_bias_l2 += bias_penalty.item()
                finite_examples += 1
                last_train_loss = output.loss.item()
                last_bias_l2 = bias_penalty.item()

                if example_idx % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.controller.parameters(),
                        max_norm=max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(grad_norm):
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        progress.console.print(
                            f"epoch {epoch + 1}/{epochs} example {example_idx}/{len(train_dataset)} skipped non-finite grad_norm={grad_norm.item()}",
                            style="yellow",
                        )
                        progress.update(
                            task_id,
                            advance=1,
                            metrics=format_train_metrics(
                                train_loss=last_train_loss,
                                bias_l2=last_bias_l2,
                                lr=current_lr,
                                skipped_steps=skipped_steps - epoch_skipped_start,
                            ),
                        )
                        continue
                    current_lr = set_optimizer_lr(
                        optimizer,
                        base_lr=learning_rate,
                        global_step=global_step,
                        warmup_steps=warmup_steps,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if (
                    example_idx == len(train_dataset)
                    and example_idx % grad_accum_steps != 0
                ):
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.controller.parameters(),
                        max_norm=max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    if not torch.isfinite(grad_norm):
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        progress.console.print(
                            f"epoch {epoch + 1}/{epochs} example {example_idx}/{len(train_dataset)} skipped non-finite grad_norm={grad_norm.item()}",
                            style="yellow",
                        )
                        progress.update(
                            task_id,
                            advance=1,
                            metrics=format_train_metrics(
                                train_loss=last_train_loss,
                                bias_l2=last_bias_l2,
                                lr=current_lr,
                                skipped_steps=skipped_steps - epoch_skipped_start,
                            ),
                        )
                        continue
                    current_lr = set_optimizer_lr(
                        optimizer,
                        base_lr=learning_rate,
                        global_step=global_step,
                        warmup_steps=warmup_steps,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                progress.update(
                    task_id,
                    advance=1,
                    metrics=format_train_metrics(
                        train_loss=last_train_loss,
                        bias_l2=last_bias_l2,
                        lr=current_lr,
                        skipped_steps=skipped_steps - epoch_skipped_start,
                    ),
                )

            avg_loss = total_loss / max(finite_examples, 1)
            avg_bias_l2 = total_bias_l2 / max(finite_examples, 1)
            should_validate = bool(val_dataset) and eval_every > 0 and (
                (epoch + 1) % eval_every == 0 or epoch + 1 == epochs
            )
            if should_validate:
                progress.update(
                    task_id,
                    description=f"epoch {epoch + 1}/{epochs} validating",
                    metrics="running validation...",
                )
                val_loss, val_bias_l2 = evaluate_loss(
                    model,
                    tokenizer,
                    val_dataset,
                    eos_token_id=eos_token_id,
                    device=device_obj,
                    router_bias_l2=router_bias_l2,
                    max_response_tokens=max_response_tokens,
                )
            else:
                val_loss = None
                val_bias_l2 = None
            summary = format_epoch_summary(
                train_loss=avg_loss,
                train_bias_l2=avg_bias_l2,
                val_loss=val_loss,
                val_bias_l2=val_bias_l2,
                lr=current_lr,
                finite_examples=finite_examples,
                skipped_steps=skipped_steps,
            )
            progress.update(
                task_id,
                description=f"epoch {epoch + 1}/{epochs}",
                metrics="done",
            )
            progress.console.print(f"epoch {epoch + 1} {summary}", style="green")
            checkpoint_metrics = {
                "train_loss": avg_loss,
                "train_bias_l2": avg_bias_l2,
                "val_loss": val_loss,
                "val_bias_l2": val_bias_l2,
                "lr": current_lr,
                "finite_examples": finite_examples,
                "skipped_steps": skipped_steps,
            }
            save_controller_checkpoint(
                output_path,
                model.controller,
                optimizer,
                global_step,
                epoch=epoch + 1,
                metrics=checkpoint_metrics,
            )
            metric = select_best_metric(train_loss=avg_loss, val_loss=val_loss)
            if (
                best_output_path is not None
                and (best_metric is None or metric < best_metric)
            ):
                best_metric = metric
                save_controller_checkpoint(
                    best_output_path,
                    model.controller,
                    optimizer,
                    global_step,
                    epoch=epoch + 1,
                    metrics=checkpoint_metrics,
                )
                progress.console.print(
                    f"saved best checkpoint to {best_output_path} metric={metric:.4f}",
                    style="cyan",
                )

    status(f"Saved controller checkpoint to {output_path}")
    if best_output_path is not None and best_metric is not None:
        status(f"Best controller checkpoint: {best_output_path} metric={best_metric:.4f}")


if __name__ == "__main__":
    app()
