from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

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
    DEFAULT_GEMMA_IFEVAL_BEST_CONTROLLER_PATH,
    DEFAULT_GEMMA_IFEVAL_CONTROLLER_PATH,
    DEFAULT_GEMMA_IFEVAL_SYNTHETIC_DATASET_PATH,
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


def category_counts(records: list[dict]) -> dict[str, int]:
    counts = Counter(str(record.get("category", "default")) for record in records)
    return dict(sorted(counts.items()))


def clean_mlflow_value(value: object) -> object:
    if value is None:
        return "none"
    if isinstance(value, Path):
        return str(value)
    return value


def mlflow_key_part(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in "._-" else "_" for char in value
    )


def build_mlflow_params(
    *,
    dataset_path: Path,
    dataset: list[dict],
    train_dataset: list[dict],
    val_dataset: list[dict],
    model_id: str,
    revision: str | None,
    cache_dir: str | None,
    local_dir: str | None,
    offline: bool,
    requested_device: str,
    requested_dtype: str,
    actual_device: torch.device,
    param_dtype: torch.dtype,
    hidden_size: int,
    output_path: Path,
    best_output_path: Path | None,
    num_virtual_tokens: int,
    controller_dim: int,
    controller_hidden_dim: int,
    controller_dropout: float,
    epochs: int,
    learning_rate: float,
    warmup_steps: int | None,
    effective_warmup_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    val_split: float,
    seed: int,
    max_examples: int | None,
    max_response_tokens: int | None,
    patience: int,
    min_improvement: float,
) -> dict[str, object]:
    params: dict[str, object] = {
        "dataset.path": dataset_path,
        "dataset.examples": len(dataset),
        "dataset.train_examples": len(train_dataset),
        "dataset.val_examples": len(val_dataset),
        "dataset.max_examples": max_examples,
        "model.id": model_id,
        "model.revision": revision,
        "model.cache_dir": cache_dir,
        "model.local_dir": local_dir,
        "model.offline": offline,
        "model.requested_device": requested_device,
        "model.requested_dtype": requested_dtype,
        "model.actual_device": actual_device.type,
        "model.param_dtype": str(param_dtype).replace("torch.", ""),
        "model.hidden_size": hidden_size,
        "checkpoint.output_path": output_path,
        "checkpoint.best_output_path": best_output_path,
        "controller.num_virtual_tokens": num_virtual_tokens,
        "controller.controller_dim": controller_dim,
        "controller.hidden_dim": controller_hidden_dim,
        "controller.dropout": controller_dropout,
        "train.epochs": epochs,
        "train.learning_rate": learning_rate,
        "train.warmup_steps": warmup_steps,
        "train.effective_warmup_steps": effective_warmup_steps,
        "train.warmup_ratio": warmup_ratio,
        "train.min_lr_ratio": min_lr_ratio,
        "train.weight_decay": weight_decay,
        "train.max_grad_norm": max_grad_norm,
        "train.val_split": val_split,
        "train.seed": seed,
        "train.max_response_tokens": max_response_tokens,
        "train.patience": patience,
        "train.min_improvement": min_improvement,
    }
    for category, count in category_counts(dataset).items():
        params[f"dataset.category.{mlflow_key_part(category)}.examples"] = count
    for category, count in category_counts(train_dataset).items():
        params[f"dataset.train_category.{mlflow_key_part(category)}.examples"] = count
    for category, count in category_counts(val_dataset).items():
        params[f"dataset.val_category.{mlflow_key_part(category)}.examples"] = count
    return {key: clean_mlflow_value(value) for key, value in params.items()}


def load_mlflow(enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        import mlflow
    except ImportError as exc:
        raise typer.BadParameter(
            "MLflow is not installed. Run `uv sync` or disable --mlflow."
        ) from exc
    return mlflow


def normalize_mlflow_artifact_location(location: str | None) -> str | None:
    if location is None:
        return None
    if "://" in location or location.startswith("dbfs:"):
        return location
    return Path(location).expanduser().resolve().as_uri()


def start_mlflow_run(
    mlflow: Any,
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None,
    artifact_location: str | None,
) -> Any:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=normalize_mlflow_artifact_location(artifact_location),
        )
    else:
        experiment_id = experiment.experiment_id
    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        nested=mlflow.active_run() is not None,
    )


def log_mlflow_artifact_if_exists(
    mlflow: Any,
    path: Path | None,
    *,
    artifact_path: str,
) -> None:
    if path is not None and path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def split_train_val(
    records: list[dict],
    *,
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if val_split <= 0 or len(records) < 2:
        return records, []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("category", "default"))].append(record)

    rng = random.Random(seed)
    train_records: list[dict] = []
    val_records: list[dict] = []
    for category in sorted(grouped):
        items = list(grouped[category])
        rng.shuffle(items)
        if len(items) < 2:
            train_records.extend(items)
            continue
        val_count = int(round(len(items) * val_split))
        if val_count <= 0:
            val_count = 1
        val_count = min(val_count, len(items) - 1)
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:])

    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


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
    total_steps: int,
    min_lr_ratio: float,
) -> float:
    step_index = global_step + 1
    if warmup_steps > 0 and step_index <= warmup_steps:
        lr = base_lr * (step_index / warmup_steps)
    else:
        if total_steps <= warmup_steps:
            progress = 1.0
        else:
            progress = (step_index - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)
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

    # build_training_example already returns next-token labels:
    # input_ids = full_ids[:-1], labels = full_ids[1:].
    # Shifting again here trains the controller against the token two positions
    # ahead, which can lower token loss while destroying generation quality.
    if combined_mask is not None:
        active_positions = combined_mask.to(device=combined_labels.device, dtype=torch.bool)
        active_logits = logits[active_positions].contiguous()
        active_labels = combined_labels[active_positions].contiguous()
    else:
        active_logits = logits.contiguous()
        active_labels = combined_labels.contiguous()
    if active_labels.numel() == 0 or not torch.any(active_labels != -100):
        return None
    loss = torch.nn.functional.cross_entropy(
        active_logits.view(-1, runtime.model.config.get_text_config().vocab_size),
        active_labels.view(-1).to(active_logits.device),
    )
    return loss


@torch.no_grad()
def evaluate_loss_report(
    runtime: GemmaSoftPromptRuntime,
    controller: PromptPoolingSoftPromptController,
    dataset: list[dict],
    *,
    max_response_tokens: int | None,
) -> tuple[float | None, dict[str, float]]:
    if not dataset:
        return None, {}
    controller.eval()
    total_loss = 0.0
    finite_examples = 0
    category_loss: dict[str, float] = defaultdict(float)
    category_examples: dict[str, int] = defaultdict(int)
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
        loss_value = loss.item()
        category = str(record.get("category", "default"))
        total_loss += loss_value
        finite_examples += 1
        category_loss[category] += loss_value
        category_examples[category] += 1
    if finite_examples == 0:
        return None, {}
    return total_loss / finite_examples, {
        category: category_loss[category] / category_examples[category]
        for category in sorted(category_examples)
    }


def evaluate_loss(
    runtime: GemmaSoftPromptRuntime,
    controller: PromptPoolingSoftPromptController,
    dataset: list[dict],
    *,
    max_response_tokens: int | None,
) -> float | None:
    val_loss, _ = evaluate_loss_report(
        runtime,
        controller,
        dataset,
        max_response_tokens=max_response_tokens,
    )
    return val_loss


def resolve_warmup_steps(
    *,
    warmup_steps: int | None,
    warmup_ratio: float,
    total_steps: int,
) -> int:
    if warmup_steps is not None:
        return warmup_steps
    if warmup_ratio <= 0:
        return 0
    return max(1, int(round(total_steps * warmup_ratio)))


@app.command()
def train(
    dataset_path: Annotated[
        Path,
        typer.Argument(help="JSONL dataset with prompt and response fields."),
    ] = DEFAULT_GEMMA_IFEVAL_SYNTHETIC_DATASET_PATH,
    output_path: Annotated[
        Path,
        typer.Option(help="Where to save the latest controller checkpoint."),
    ] = DEFAULT_GEMMA_IFEVAL_CONTROLLER_PATH,
    best_output_path: Annotated[
        Path | None,
        typer.Option(help="Optional best checkpoint path."),
    ] = DEFAULT_GEMMA_IFEVAL_BEST_CONTROLLER_PATH,
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
    num_virtual_tokens: Annotated[
        int, typer.Option(help="Number of soft prompt tokens generated by the controller.")
    ] = 16,
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
    ] = 24,
    learning_rate: Annotated[
        float, typer.Option(help="AdamW learning rate for the controller.")
    ] = 2e-4,
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
    max_response_tokens: Annotated[
        int | None,
        typer.Option(help="Optional cap on response tokens used for loss/training."),
    ] = 512,
    patience: Annotated[
        int, typer.Option(help="Early stopping patience in epochs without validation improvement. Use 0 to disable.")
    ] = 10,
    min_improvement: Annotated[
        float, typer.Option(help="Minimum validation improvement required to reset patience.")
    ] = 1e-3,
    mlflow_enabled: Annotated[
        bool, typer.Option("--mlflow/--no-mlflow", help="Log training params, metrics, and artifacts to MLflow.")
    ] = True,
    mlflow_tracking_uri: Annotated[
        str, typer.Option(help="MLflow tracking URI used when --mlflow is enabled.")
    ] = "sqlite:///artifacts/mlflow.db",
    mlflow_experiment: Annotated[
        str, typer.Option(help="MLflow experiment name used when --mlflow is enabled.")
    ] = "llm-enhance-gemma",
    mlflow_run_name: Annotated[
        str | None, typer.Option(help="Optional MLflow run name.")
    ] = "gemma-ifeval-synthetic",
    mlflow_artifact_location: Annotated[
        str | None, typer.Option(help="Artifact location for newly created MLflow experiments.")
    ] = "artifacts/mlflow-artifacts",
    mlflow_log_artifacts: Annotated[
        bool, typer.Option("--mlflow-log-artifacts/--no-mlflow-log-artifacts", help="Log dataset and checkpoint files as MLflow artifacts.")
    ] = True,
) -> None:
    if epochs <= 0:
        raise typer.BadParameter("--epochs must be greater than 0.")
    if not 0.0 <= controller_dropout < 1.0:
        raise typer.BadParameter("--controller-dropout must be in [0, 1).")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise typer.BadParameter("--min-lr-ratio must be in [0, 1].")
    if warmup_steps is not None and warmup_steps < 0:
        raise typer.BadParameter("--warmup-steps must be greater than or equal to 0.")
    if not 0.0 <= warmup_ratio <= 1.0:
        raise typer.BadParameter("--warmup-ratio must be in [0, 1].")
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
        revision=revision,
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
            dropout=controller_dropout,
        )
    ).to(device=runtime.device, dtype=torch.float32)
    optimizer = AdamW(
        controller.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    mlflow = load_mlflow(mlflow_enabled)
    mlflow_run = nullcontext()
    if mlflow is not None:
        mlflow_run = start_mlflow_run(
            mlflow,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment,
            run_name=mlflow_run_name,
            artifact_location=mlflow_artifact_location,
        )

    global_step = 0
    current_lr = 0.0
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_without_improvement = 0
    total_steps = max(len(train_dataset) * epochs, 1)
    effective_warmup_steps = resolve_warmup_steps(
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        total_steps=total_steps,
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

    with mlflow_run:
        if mlflow is not None:
            mlflow.log_params(
                build_mlflow_params(
                    dataset_path=dataset_path,
                    dataset=dataset,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    model_id=model_id,
                    revision=revision,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                    offline=offline,
                    requested_device=device,
                    requested_dtype=dtype,
                    actual_device=runtime.device,
                    param_dtype=runtime.param_dtype,
                    hidden_size=hidden_size,
                    output_path=output_path,
                    best_output_path=best_output_path,
                    num_virtual_tokens=num_virtual_tokens,
                    controller_dim=controller_dim,
                    controller_hidden_dim=controller_hidden_dim,
                    controller_dropout=controller_dropout,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    warmup_steps=warmup_steps,
                    effective_warmup_steps=effective_warmup_steps,
                    warmup_ratio=warmup_ratio,
                    min_lr_ratio=min_lr_ratio,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    val_split=val_split,
                    seed=seed,
                    max_examples=max_examples,
                    max_response_tokens=max_response_tokens,
                    patience=patience,
                    min_improvement=min_improvement,
                )
            )
            mlflow.log_params(
                {
                    "mlflow.tracking_uri": mlflow_tracking_uri,
                    "mlflow.experiment": mlflow_experiment,
                    "mlflow.artifact_location": clean_mlflow_value(
                        mlflow_artifact_location
                    ),
                    "mlflow.log_artifacts": mlflow_log_artifacts,
                }
            )
            if mlflow_log_artifacts:
                log_mlflow_artifact_if_exists(
                    mlflow,
                    dataset_path,
                    artifact_path="datasets",
                )

        with progress:
            for epoch in range(epochs):
                epoch_train_dataset = list(train_dataset)
                random.Random(seed + epoch).shuffle(epoch_train_dataset)
                total_loss = 0.0
                finite_examples = 0
                task_id = progress.add_task(
                    f"epoch {epoch + 1}/{epochs}",
                    total=len(epoch_train_dataset),
                    metrics="starting",
                )
                for record in epoch_train_dataset:
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
                        warmup_steps=effective_warmup_steps,
                        total_steps=total_steps,
                        min_lr_ratio=min_lr_ratio,
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
                val_loss, val_loss_by_category = evaluate_loss_report(
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
                    "best_metric_so_far": best_metric,
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
                    and (best_metric is None or metric < best_metric - min_improvement)
                ):
                    best_metric = metric
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    save_gemma_controller_checkpoint(
                        best_output_path,
                        controller,
                        optimizer,
                        global_step,
                        epoch=epoch + 1,
                        metrics=checkpoint_metrics,
                    )
                else:
                    epochs_without_improvement += 1

                if mlflow is not None:
                    mlflow_metrics = {
                        "train_loss": avg_loss,
                        "lr": current_lr,
                        "finite_examples": finite_examples,
                        "global_step": global_step,
                        "selection_metric": metric,
                        "epochs_without_improvement": epochs_without_improvement,
                    }
                    if val_loss is not None:
                        mlflow_metrics["val_loss"] = val_loss
                    for category, category_val_loss in val_loss_by_category.items():
                        key = f"val_loss_by_category.{mlflow_key_part(category)}"
                        mlflow_metrics[key] = category_val_loss
                    if best_metric is not None:
                        mlflow_metrics["best_metric"] = best_metric
                    mlflow.log_metrics(mlflow_metrics, step=epoch + 1)

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

        if mlflow is not None:
            mlflow.set_tag("best_epoch", "none" if best_epoch is None else str(best_epoch))
            if best_metric is not None:
                mlflow.log_metric("final_best_metric", best_metric)
            mlflow.log_metric("final_global_step", global_step)
            if mlflow_log_artifacts:
                log_mlflow_artifact_if_exists(
                    mlflow,
                    output_path,
                    artifact_path="checkpoints",
                )
                log_mlflow_artifact_if_exists(
                    mlflow,
                    best_output_path,
                    artifact_path="checkpoints",
                )

    status(f"Saved Gemma controller checkpoint to {output_path}")
    if best_output_path is not None and best_metric is not None:
        best_epoch_suffix = "" if best_epoch is None else f" epoch={best_epoch}"
        status(
            f"Best Gemma controller checkpoint: {best_output_path} "
            f"metric={best_metric:.4f}{best_epoch_suffix}"
        )


if __name__ == "__main__":
    app()
