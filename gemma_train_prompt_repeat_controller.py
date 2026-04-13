from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
import torch.nn.functional as F
import typer
from torch.optim import AdamW

from gemma_train_controller import (
    average_accumulated_gradients,
    build_training_progress,
    console,
    format_epoch_summary,
    format_step_metrics,
    optimizer_steps_per_epoch,
    resolve_warmup_steps,
    set_optimizer_lr,
    split_train_val,
)
from llm_enhance.gemma_runtime import (
    DEFAULT_GEMMA_MODEL_ID,
    DEFAULT_GEMMA_PROMPT_REPEAT_BEST_CONTROLLER_PATH,
    DEFAULT_GEMMA_PROMPT_REPEAT_CONTROLLER_PATH,
    DEFAULT_GEMMA_PROMPT_REPEAT_DATASET_PATH,
    GemmaSoftPromptRuntime,
    load_jsonl,
    save_last_token_controller_checkpoint,
)
from llm_enhance.last_token_controller import (
    LastTokenHiddenStateController,
    LastTokenHiddenStateControllerConfig,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


def repeat_prompt(prompt: str, repeats: int) -> str:
    if repeats <= 1:
        return prompt
    return "\n\n".join(prompt for _ in range(repeats))


@dataclass(slots=True)
class HiddenStatePairs:
    sources: torch.Tensor
    targets: torch.Tensor
    categories: list[str]

    def __len__(self) -> int:
        return self.sources.shape[0]


@torch.no_grad()
def build_last_hidden_pair(
    runtime: GemmaSoftPromptRuntime,
    record: dict,
    *,
    prompt_repeats: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    system_prompt = record.get("system_prompt")
    source_last_hidden = runtime.final_prompt_hidden_state(
        prompt=record["prompt"],
        system_prompt=system_prompt,
    )
    target_last_hidden = runtime.final_prompt_hidden_state(
        prompt=repeat_prompt(record["prompt"], prompt_repeats),
        system_prompt=system_prompt,
    )
    return source_last_hidden.detach(), target_last_hidden.detach()


@torch.no_grad()
def build_hidden_state_pairs(
    runtime: GemmaSoftPromptRuntime,
    dataset: list[dict],
    *,
    prompt_repeats: int,
) -> HiddenStatePairs:
    source_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    categories: list[str] = []
    for record in dataset:
        source_last_hidden, target_last_hidden = build_last_hidden_pair(
            runtime,
            record,
            prompt_repeats=prompt_repeats,
        )
        source_last_hidden = source_last_hidden.float()
        target_last_hidden = target_last_hidden.float()
        if not (
            torch.isfinite(source_last_hidden).all()
            and torch.isfinite(target_last_hidden).all()
        ):
            continue
        source_rows.append(source_last_hidden)
        target_rows.append(target_last_hidden)
        categories.append(str(record.get("category", "default")))

    if not source_rows:
        empty = torch.empty(0, 0, device=runtime.device, dtype=torch.float32)
        return HiddenStatePairs(sources=empty, targets=empty, categories=[])
    return HiddenStatePairs(
        sources=torch.cat(source_rows, dim=0),
        targets=torch.cat(target_rows, dim=0),
        categories=categories,
    )


def shuffled_batch_indices(
    total_examples: int,
    *,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    indices = list(range(total_examples))
    random.Random(seed).shuffle(indices)
    return [
        indices[start : start + batch_size]
        for start in range(0, total_examples, batch_size)
    ]


def prompt_repeat_distillation_loss(
    controller: LastTokenHiddenStateController,
    *,
    source_last_hidden: torch.Tensor,
    target_last_hidden: torch.Tensor,
    cosine_weight: float,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'.")
    predicted_last_hidden = controller(source_last_hidden)
    mse = F.mse_loss(
        predicted_last_hidden.float(),
        target_last_hidden.float(),
        reduction="none",
    ).mean(dim=-1)
    if cosine_weight > 0:
        cosine = 1.0 - F.cosine_similarity(
            predicted_last_hidden.float(),
            target_last_hidden.float(),
            dim=-1,
        )
        loss = mse + cosine_weight * cosine
    else:
        loss = mse
    if reduction == "none":
        return loss
    return loss.mean()


def hidden_state_pair_loss(
    *,
    predicted_last_hidden: torch.Tensor,
    target_last_hidden: torch.Tensor,
    cosine_weight: float,
    reduction: str = "mean",
) -> torch.Tensor:
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'.")
    mse = F.mse_loss(
        predicted_last_hidden.float(),
        target_last_hidden.float(),
        reduction="none",
    ).mean(dim=-1)
    if cosine_weight > 0:
        cosine = 1.0 - F.cosine_similarity(
            predicted_last_hidden.float(),
            target_last_hidden.float(),
            dim=-1,
        )
        loss = mse + cosine_weight * cosine
    else:
        loss = mse
    if reduction == "none":
        return loss
    return loss.mean()


def source_baseline_loss(
    pairs: HiddenStatePairs,
    *,
    cosine_weight: float,
) -> float | None:
    if len(pairs) == 0:
        return None
    loss = hidden_state_pair_loss(
        predicted_last_hidden=pairs.sources,
        target_last_hidden=pairs.targets,
        cosine_weight=cosine_weight,
        reduction="mean",
    )
    return loss.item()


@torch.no_grad()
def evaluate_cached_loss_report(
    controller: LastTokenHiddenStateController,
    pairs: HiddenStatePairs,
    *,
    batch_size: int,
    cosine_weight: float,
) -> tuple[float | None, dict[str, float]]:
    if len(pairs) == 0:
        return None, {}
    controller.eval()
    total_loss = 0.0
    finite_examples = 0
    category_loss: dict[str, float] = defaultdict(float)
    category_examples: dict[str, int] = defaultdict(int)
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        source_batch = pairs.sources[start:end]
        target_batch = pairs.targets[start:end]
        per_example_loss = prompt_repeat_distillation_loss(
            controller,
            source_last_hidden=source_batch,
            target_last_hidden=target_batch,
            cosine_weight=cosine_weight,
            reduction="none",
        )
        for offset, loss in enumerate(per_example_loss):
            if not torch.isfinite(loss):
                continue
            loss_value = loss.item()
            category = pairs.categories[start + offset]
            total_loss += loss_value
            finite_examples += 1
            category_loss[category] += loss_value
            category_examples[category] += 1
    controller.train()
    if finite_examples == 0:
        return None, {}
    return total_loss / finite_examples, {
        category: category_loss[category] / category_examples[category]
        for category in sorted(category_examples)
    }


@app.command()
def train(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="Prompt-only JSONL dataset with prompt/category fields."),
    ] = DEFAULT_GEMMA_PROMPT_REPEAT_DATASET_PATH,
    output_path: Annotated[
        Path,
        typer.Option(help="Where to save the latest last-token controller checkpoint."),
    ] = DEFAULT_GEMMA_PROMPT_REPEAT_CONTROLLER_PATH,
    best_output_path: Annotated[
        Path | None,
        typer.Option(help="Optional best last-token controller checkpoint path."),
    ] = DEFAULT_GEMMA_PROMPT_REPEAT_BEST_CONTROLLER_PATH,
    model_id: Annotated[
        str, typer.Option(help="Gemma model id on Hugging Face.")
    ] = DEFAULT_GEMMA_MODEL_ID,
    revision: Annotated[
        str | None, typer.Option(help="Optional Hugging Face model revision or snapshot hash.")
    ] = None,
    cache_dir: Annotated[
        str | None, typer.Option(help="Optional Hugging Face cache directory.")
    ] = None,
    local_dir: Annotated[
        str | None, typer.Option(help="Optional local model directory. Overrides --model-id.")
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
    prompt_repeats: Annotated[
        int, typer.Option(help="Target repeated-prompt count used for distillation.")
    ] = 2,
    cosine_weight: Annotated[
        float, typer.Option(help="Weight for last-hidden-state cosine distance.")
    ] = 0.1,
    controller_dim: Annotated[
        int, typer.Option(help="Latent dimension inside the controller.")
    ] = 256,
    controller_hidden_dim: Annotated[
        int, typer.Option(help="Hidden size of the controller MLP.")
    ] = 1024,
    controller_dropout: Annotated[
        float, typer.Option(help="Dropout inside the controller MLP.")
    ] = 0.0,
    epochs: Annotated[
        int, typer.Option(help="Number of full passes over the dataset.")
    ] = 20,
    learning_rate: Annotated[
        float, typer.Option(help="AdamW learning rate for the controller.")
    ] = 3e-4,
    grad_accum_steps: Annotated[
        int, typer.Option(help="Accumulate gradients over this many examples before each optimizer step.")
    ] = 16,
    batch_size: Annotated[
        int, typer.Option(help="Hidden-state training batch size.")
    ] = 16,
    warmup_steps: Annotated[
        int | None,
        typer.Option(help="Linear learning-rate warmup over optimizer steps. Defaults to --warmup-ratio of total steps."),
    ] = None,
    warmup_ratio: Annotated[
        float, typer.Option(help="Warmup ratio used when --warmup-steps is omitted.")
    ] = 0.03,
    min_lr_ratio: Annotated[
        float, typer.Option(help="Minimum learning-rate ratio reached after cosine decay.")
    ] = 0.1,
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
    patience: Annotated[
        int, typer.Option(help="Early stopping patience in epochs without validation improvement. Use 0 to disable.")
    ] = 5,
    min_improvement: Annotated[
        float, typer.Option(help="Minimum validation improvement required to reset patience.")
    ] = 1e-4,
) -> None:
    if prompt_repeats <= 1:
        raise typer.BadParameter("--prompt-repeats must be greater than 1.")
    if cosine_weight < 0:
        raise typer.BadParameter("--cosine-weight must be greater than or equal to 0.")
    if epochs <= 0:
        raise typer.BadParameter("--epochs must be greater than 0.")
    if not 0.0 <= controller_dropout < 1.0:
        raise typer.BadParameter("--controller-dropout must be in [0, 1).")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise typer.BadParameter("--min-lr-ratio must be in [0, 1].")
    if grad_accum_steps <= 0:
        raise typer.BadParameter("--grad-accum-steps must be greater than 0.")
    if batch_size <= 0:
        raise typer.BadParameter("--batch-size must be greater than 0.")
    if warmup_steps is not None and warmup_steps < 0:
        raise typer.BadParameter("--warmup-steps must be greater than or equal to 0.")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise typer.BadParameter("--warmup-ratio must be in [0, 1].")

    dataset = load_jsonl(dataset_path)
    if max_examples is not None:
        dataset = dataset[:max_examples]
    if not dataset:
        raise typer.BadParameter("Dataset is empty.")
    missing_prompt = [idx for idx, record in enumerate(dataset) if "prompt" not in record]
    if missing_prompt:
        raise typer.BadParameter(f"Records missing prompt field: {missing_prompt[:5]}")
    train_dataset, val_dataset = split_train_val(
        dataset,
        val_split=val_split,
        seed=seed,
    )

    runtime = GemmaSoftPromptRuntime(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
        device=device,
        dtype=dtype,
    )
    status("Precomputing frozen Gemma last-token hidden-state pairs...")
    train_pairs = build_hidden_state_pairs(
        runtime,
        train_dataset,
        prompt_repeats=prompt_repeats,
    )
    val_pairs = build_hidden_state_pairs(
        runtime,
        val_dataset,
        prompt_repeats=prompt_repeats,
    )
    if len(train_pairs) == 0:
        raise typer.BadParameter("No finite train hidden-state pairs were produced.")
    train_source_baseline_loss = source_baseline_loss(
        train_pairs,
        cosine_weight=cosine_weight,
    )
    val_source_baseline_loss = source_baseline_loss(
        val_pairs,
        cosine_weight=cosine_weight,
    )
    baseline_parts = [f"{len(train_pairs)} train pairs", f"{len(val_pairs)} val pairs"]
    if train_source_baseline_loss is not None:
        baseline_parts.append(f"train_source_baseline={train_source_baseline_loss:.4f}")
    if val_source_baseline_loss is not None:
        baseline_parts.append(f"val_source_baseline={val_source_baseline_loss:.4f}")
    status("Cached " + ", ".join(baseline_parts) + ".")

    hidden_size = runtime.model.get_input_embeddings().embedding_dim
    controller = LastTokenHiddenStateController(
        LastTokenHiddenStateControllerConfig(
            hidden_size=hidden_size,
            controller_dim=controller_dim,
            hidden_dim=controller_hidden_dim,
            dropout=controller_dropout,
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
    best_epoch: int | None = None
    epochs_without_improvement = 0
    steps_per_epoch = optimizer_steps_per_epoch(
        len(train_pairs),
        grad_accum_steps,
    )
    total_steps = max(steps_per_epoch * epochs, 1)
    effective_warmup_steps = resolve_warmup_steps(
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
    )
    progress = build_training_progress(console)

    status(
        "Training last-token prompt-repeat controller: "
        f"{len(train_pairs)} train / {len(val_pairs)} val hidden pairs, "
        f"target prompt repeats={prompt_repeats}, batch_size={batch_size}."
    )
    with progress:
        for epoch in range(epochs):
            controller.train()
            epoch_batches = shuffled_batch_indices(
                len(train_pairs),
                batch_size=batch_size,
                seed=seed + epoch,
            )
            total_loss = 0.0
            finite_examples = 0
            accumulated_examples = 0
            optimizer.zero_grad(set_to_none=True)
            task_id = progress.add_task(
                f"epoch {epoch + 1}/{epochs}",
                total=len(train_pairs),
                metrics="starting",
            )
            for batch_indices in epoch_batches:
                batch_index_tensor = torch.tensor(
                    batch_indices,
                    device=train_pairs.sources.device,
                    dtype=torch.long,
                )
                source_last_hidden = train_pairs.sources.index_select(
                    0,
                    batch_index_tensor,
                )
                target_last_hidden = train_pairs.targets.index_select(
                    0,
                    batch_index_tensor,
                )
                loss = prompt_repeat_distillation_loss(
                    controller,
                    source_last_hidden=source_last_hidden,
                    target_last_hidden=target_last_hidden,
                    cosine_weight=cosine_weight,
                )
                if not torch.isfinite(loss):
                    progress.update(
                        task_id,
                        advance=len(batch_indices),
                        metrics="skipped non-finite",
                    )
                    continue
                batch_examples = len(batch_indices)
                (loss * batch_examples).backward()
                accumulated_examples += batch_examples
                total_loss += loss.item() * batch_examples
                finite_examples += batch_examples
                if accumulated_examples >= grad_accum_steps:
                    average_accumulated_gradients(
                        controller,
                        accumulated_examples=accumulated_examples,
                    )
                    torch.nn.utils.clip_grad_norm_(
                        controller.parameters(),
                        max_norm=max_grad_norm,
                        error_if_nonfinite=False,
                    )
                    current_lr = set_optimizer_lr(
                        optimizer,
                        base_lr=learning_rate,
                        global_step=global_step,
                        warmup_steps=effective_warmup_steps,
                        total_steps=total_steps,
                        min_lr_ratio=min_lr_ratio,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accumulated_examples = 0
                progress.update(
                    task_id,
                    advance=batch_examples,
                    metrics=format_step_metrics(
                        loss_value=loss.item(),
                        current_lr=current_lr,
                        accumulated_examples=accumulated_examples,
                        grad_accum_steps=grad_accum_steps,
                    ),
                )

            if accumulated_examples > 0:
                average_accumulated_gradients(
                    controller,
                    accumulated_examples=accumulated_examples,
                )
                torch.nn.utils.clip_grad_norm_(
                    controller.parameters(),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=False,
                )
                current_lr = set_optimizer_lr(
                    optimizer,
                    base_lr=learning_rate,
                    global_step=global_step,
                    warmup_steps=effective_warmup_steps,
                    total_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            avg_loss = total_loss / max(finite_examples, 1)
            val_loss, _ = evaluate_cached_loss_report(
                controller,
                val_pairs,
                batch_size=batch_size,
                cosine_weight=cosine_weight,
            )
            summary = format_epoch_summary(
                train_loss=avg_loss,
                val_loss=val_loss,
                lr=current_lr,
                finite_examples=finite_examples,
            )
            if val_loss is not None and val_source_baseline_loss is not None:
                summary += (
                    f" val_source={val_source_baseline_loss:.4f} "
                    f"val_gain={val_source_baseline_loss - val_loss:.4f}"
                )
            progress.console.print(f"epoch {epoch + 1} {summary}", style="green")

            metric = avg_loss if val_loss is None else val_loss
            checkpoint_metrics = {
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "finite_examples": finite_examples,
                "best_metric_so_far": best_metric,
                "objective": "last_token_prompt_repeat_distillation",
                "prompt_repeats": prompt_repeats,
                "cosine_weight": cosine_weight,
                "batch_size": batch_size,
                "train_source_baseline_loss": train_source_baseline_loss,
                "val_source_baseline_loss": val_source_baseline_loss,
            }
            save_last_token_controller_checkpoint(
                output_path,
                controller,
                optimizer,
                global_step,
                epoch=epoch + 1,
                metrics=checkpoint_metrics,
            )
            if (
                best_output_path is not None
                and (best_metric is None or metric < best_metric - min_improvement)
            ):
                best_metric = metric
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                save_last_token_controller_checkpoint(
                    best_output_path,
                    controller,
                    optimizer,
                    global_step,
                    epoch=epoch + 1,
                    metrics=checkpoint_metrics,
                )
            else:
                epochs_without_improvement += 1

            if best_epoch is not None:
                progress.console.print(
                    f"best so far: epoch {best_epoch} metric={best_metric:.4f}",
                    style="cyan",
                )

            if (
                patience > 0
                and val_loss is not None
                and epochs_without_improvement >= patience
            ):
                status(
                    "Early stopping triggered after "
                    f"{epochs_without_improvement} epochs without validation improvement."
                )
                break

    status(f"Saved Gemma last-token controller checkpoint to {output_path}")
    if best_output_path is not None and best_metric is not None:
        best_epoch_suffix = "" if best_epoch is None else f" epoch={best_epoch}"
        status(
            f"Best Gemma last-token controller checkpoint: {best_output_path} "
            f"metric={best_metric:.4f}{best_epoch_suffix}"
        )


if __name__ == "__main__":
    app()
