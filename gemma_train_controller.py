from __future__ import annotations

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

from trinity.dense_controller import (
    PromptPoolingSoftPromptController,
    PromptPoolingSoftPromptControllerConfig,
    prepend_soft_prompt,
)
from trinity.gemma_runtime import (
    DEFAULT_GEMMA_MODEL_ID,
    GemmaSoftPromptRuntime,
    build_training_example,
    load_jsonl,
    prepend_per_layer_inputs,
    save_gemma_controller_checkpoint,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console(stderr=True)


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


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


def format_epoch_summary(
    *,
    train_loss: float,
    val_loss: float | None,
    lr: float,
    finite_examples: int,
) -> str:
    parts = [f"train={train_loss:.4f}"]
    parts.append("val=n/a" if val_loss is None else f"val={val_loss:.4f}")
    parts.extend([f"lr={lr:.2e}", f"finite={finite_examples}"])
    return " ".join(parts)


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
    runtime: GemmaSoftPromptRuntime,
    record: dict,
    *,
    max_response_tokens: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_ids, model_input_ids, labels = build_training_example(
        runtime.tokenizer,
        prompt=record["prompt"],
        response=record["response"],
        system_prompt=record.get("system_prompt"),
        max_response_tokens=max_response_tokens,
    )
    prompt_tensor = torch.tensor([prompt_ids], device=runtime.device, dtype=torch.long)
    input_ids = torch.tensor([model_input_ids], device=runtime.device, dtype=torch.long)
    label_tensor = torch.tensor([labels], device=runtime.device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return prompt_tensor, input_ids, attention_mask, label_tensor


def forward_soft_prompt_loss(
    runtime: GemmaSoftPromptRuntime,
    controller: PromptPoolingSoftPromptController,
    *,
    prompt_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor | None:
    prompt_embeds = runtime.model.get_input_embeddings()(prompt_tensor)
    input_embeds = runtime.model.get_input_embeddings()(input_ids)
    soft_prompt = controller(prompt_embeds)
    combined_embeds, combined_mask = prepend_soft_prompt(
        inputs_embeds=input_embeds,
        soft_prompt_embeds=soft_prompt,
        attention_mask=attention_mask,
    )
    combined_per_layer_inputs = prepend_per_layer_inputs(
        per_layer_inputs=runtime.build_per_layer_inputs(input_ids),
        prefix_length=soft_prompt.shape[1],
    )
    prefix_labels = torch.full(
        (labels.shape[0], soft_prompt.shape[1]),
        -100,
        device=labels.device,
        dtype=labels.dtype,
    )
    combined_labels = torch.cat([prefix_labels, labels], dim=1)

    outputs = runtime.model.model.language_model(
        inputs_embeds=combined_embeds,
        attention_mask=combined_mask,
        per_layer_inputs=combined_per_layer_inputs,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    logits = runtime.model.lm_head(hidden_states)
    final_logit_softcapping = runtime.model.config.get_text_config().final_logit_softcapping
    if final_logit_softcapping is not None:
        logits = logits / final_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * final_logit_softcapping
    logits = logits.float()
    shift_logits = logits[..., :-1, :]
    shift_labels = combined_labels[..., 1:]
    if combined_mask is not None:
        shift_attention_mask = combined_mask[:, -shift_logits.shape[1] :].to(logits.device)
        shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
        shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
    else:
        shift_logits = shift_logits.contiguous()
        shift_labels = shift_labels.contiguous()
    if shift_labels.numel() == 0:
        return None
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, runtime.model.config.get_text_config().vocab_size),
        shift_labels.view(-1).to(shift_logits.device),
    )
    return loss


@torch.no_grad()
def evaluate_loss(
    runtime: GemmaSoftPromptRuntime,
    controller: PromptPoolingSoftPromptController,
    dataset: list[dict],
    *,
    max_response_tokens: int | None,
) -> float | None:
    if not dataset:
        return None
    controller.eval()
    total_loss = 0.0
    finite_examples = 0
    for record in dataset:
        prompt_tensor, input_ids, attention_mask, labels = build_tensors(
            runtime,
            record,
            max_response_tokens=max_response_tokens,
        )
        loss = forward_soft_prompt_loss(
            runtime,
            controller,
            prompt_tensor=prompt_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        if loss is None or not torch.isfinite(loss):
            continue
        total_loss += loss.item()
        finite_examples += 1
    if finite_examples == 0:
        return None
    return total_loss / finite_examples


@app.command()
def train(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="JSONL dataset with prompt and response fields."),
    ],
    output_path: Annotated[
        Path,
        typer.Option(help="Where to save the latest controller checkpoint."),
    ] = Path("artifacts/gemma-e2b-controller.pt"),
    best_output_path: Annotated[
        Path | None,
        typer.Option(help="Optional best checkpoint path."),
    ] = Path("artifacts/gemma-e2b-controller-best.pt"),
    model_id: Annotated[
        str, typer.Option(help="Gemma model id on Hugging Face.")
    ] = DEFAULT_GEMMA_MODEL_ID,
    cache_dir: Annotated[
        str | None, typer.Option(help="Optional Hugging Face cache directory.")
    ] = None,
    local_dir: Annotated[
        str | None, typer.Option(help="Unused placeholder for symmetry with other CLIs.")
    ] = None,
    offline: Annotated[
        bool, typer.Option(help="Use only locally cached model files.")
    ] = False,
    device: Annotated[
        str, typer.Option(help="Device override: auto, cuda, mps, or cpu.")
    ] = "auto",
    dtype: Annotated[
        str, typer.Option(help="Parameter dtype: auto, float16, bfloat16, or float32.")
    ] = "auto",
    num_virtual_tokens: Annotated[
        int, typer.Option(help="Number of soft prompt tokens generated by the controller.")
    ] = 8,
    controller_dim: Annotated[
        int, typer.Option(help="Latent dimension inside the controller.")
    ] = 256,
    controller_hidden_dim: Annotated[
        int, typer.Option(help="Hidden size of the controller MLP.")
    ] = 1024,
    epochs: Annotated[
        int, typer.Option(help="Number of full passes over the dataset.")
    ] = 1,
    learning_rate: Annotated[
        float, typer.Option(help="AdamW learning rate for the controller.")
    ] = 1e-4,
    warmup_steps: Annotated[
        int, typer.Option(help="Linear learning-rate warmup over optimizer steps.")
    ] = 20,
    weight_decay: Annotated[
        float, typer.Option(help="AdamW weight decay.")
    ] = 0.01,
    max_grad_norm: Annotated[
        float, typer.Option(help="Clip controller gradient norm to this value.")
    ] = 1.0,
    val_split: Annotated[
        float, typer.Option(help="Validation split from train data.")
    ] = 0.2,
    seed: Annotated[
        int, typer.Option(help="Random seed used for train/validation split.")
    ] = 42,
    max_examples: Annotated[
        int | None, typer.Option(help="Optional limit for quick smoke runs.")
    ] = None,
    max_response_tokens: Annotated[
        int | None,
        typer.Option(help="Optional cap on response tokens used for loss/training."),
    ] = None,
) -> None:
    if epochs <= 0:
        raise typer.BadParameter("--epochs must be greater than 0.")
    dataset = load_jsonl(dataset_path)
    if max_examples is not None:
        dataset = dataset[:max_examples]
    if not dataset:
        raise typer.BadParameter("Dataset is empty.")
    train_dataset, val_dataset = split_train_val(
        dataset,
        val_split=val_split,
        seed=seed,
    )

    runtime = GemmaSoftPromptRuntime(
        model_id=model_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
        device=device,
        dtype=dtype,
    )
    hidden_size = runtime.model.get_input_embeddings().embedding_dim
    controller = PromptPoolingSoftPromptController(
        PromptPoolingSoftPromptControllerConfig(
            hidden_size=hidden_size,
            num_virtual_tokens=num_virtual_tokens,
            controller_dim=controller_dim,
            hidden_dim=controller_hidden_dim,
        )
    ).to(device=runtime.device, dtype=torch.float32)
    optimizer = AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    global_step = 0
    current_lr = 0.0
    best_metric: float | None = None
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
            finite_examples = 0
            task_id = progress.add_task(
                f"epoch {epoch + 1}/{epochs}",
                total=len(train_dataset),
                metrics="starting",
            )
            for record in train_dataset:
                prompt_tensor, input_ids, attention_mask, labels = build_tensors(
                    runtime,
                    record,
                    max_response_tokens=max_response_tokens,
                )
                loss = forward_soft_prompt_loss(
                    runtime,
                    controller,
                    prompt_tensor=prompt_tensor,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                if loss is None or not torch.isfinite(loss):
                    progress.update(task_id, advance=1, metrics="skipped non-finite")
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    controller.parameters(),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=False,
                )
                current_lr = set_optimizer_lr(
                    optimizer,
                    base_lr=learning_rate,
                    global_step=global_step,
                    warmup_steps=warmup_steps,
                )
                optimizer.step()
                global_step += 1
                total_loss += loss.item()
                finite_examples += 1
                progress.update(
                    task_id,
                    advance=1,
                    metrics=f"loss={loss.item():.4f} lr={current_lr:.2e}",
                )

            avg_loss = total_loss / max(finite_examples, 1)
            val_loss = evaluate_loss(
                runtime,
                controller,
                val_dataset,
                max_response_tokens=max_response_tokens,
            )
            summary = format_epoch_summary(
                train_loss=avg_loss,
                val_loss=val_loss,
                lr=current_lr,
                finite_examples=finite_examples,
            )
            progress.console.print(f"epoch {epoch + 1} {summary}", style="green")

            metric = avg_loss if val_loss is None else val_loss
            checkpoint_metrics = {
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "finite_examples": finite_examples,
            }
            save_gemma_controller_checkpoint(
                output_path,
                controller,
                optimizer,
                global_step,
                epoch=epoch + 1,
                metrics=checkpoint_metrics,
            )
            if (
                best_output_path is not None
                and (best_metric is None or metric < best_metric)
            ):
                best_metric = metric
                save_gemma_controller_checkpoint(
                    best_output_path,
                    controller,
                    optimizer,
                    global_step,
                    epoch=epoch + 1,
                    metrics=checkpoint_metrics,
                )

    status(f"Saved Gemma controller checkpoint to {output_path}")
    if best_output_path is not None and best_metric is not None:
        status(f"Best Gemma controller checkpoint: {best_output_path} metric={best_metric:.4f}")


if __name__ == "__main__":
    app()
