from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated

import nltk
import torch
import typer

from instruction_following_eval import evaluation_lib

from .dense_controller import PromptPoolingSoftPromptController
from .gemma_backend import resolve_backend
from .gemma_mlx_runtime import (
    GemmaMlxRuntime,
    load_mlx_last_token_controller_checkpoint,
)
from .gemma_runtime import (
    DEFAULT_GEMMA_IFEVAL_BEST_CONTROLLER_PATH,
    DEFAULT_GEMMA_IFEVAL_CONTROLLER_STRENGTH,
    DEFAULT_GEMMA_IFEVAL_OUTPUT_DIR,
    DEFAULT_GEMMA_MODEL_ID,
    GemmaSoftPromptRuntime,
    load_gemma_controller_checkpoint,
    load_last_token_controller_checkpoint,
)
from .last_token_controller import LastTokenHiddenStateController
from .ifeval_common import (
    build_progress_message,
    compute_final_score,
    default_ifeval_input_path,
    evaluate_prompt_responses,
    persist_ifeval_progress,
    write_json,
    write_prompt_responses,
)

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


def ensure_nltk_punkt() -> None:
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab/english", "punkt_tab"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            status(
                f"Downloading NLTK resource '{resource_name}' required by IFEval..."
            )
            if not nltk.download(resource_name, quiet=True):
                raise RuntimeError(
                    f"Failed to download NLTK resource '{resource_name}' required by IFEval."
                )


def repeat_prompt(prompt: str, prompt_repeats: int) -> str:
    if prompt_repeats <= 1:
        return prompt
    return "\n\n".join(prompt for _ in range(prompt_repeats))


def build_response_record(
    *,
    prompt: str,
    generation_prompt: str,
    response: str,
) -> dict[str, str]:
    record = {"prompt": prompt, "response": response}
    if generation_prompt != prompt:
        record["generation_prompt"] = generation_prompt
    return record


@app.command()
def run_ifeval(
    input_data_path: Annotated[
        Path,
        typer.Option(help="IFEval input jsonl. Defaults to the vendored official input_data.jsonl."),
    ] = default_ifeval_input_path(),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where generated responses and IFEval reports are saved."),
    ] = DEFAULT_GEMMA_IFEVAL_OUTPUT_DIR,
    model_id: Annotated[
        str, typer.Option(help="Gemma model id on Hugging Face.")
    ] = DEFAULT_GEMMA_MODEL_ID,
    revision: Annotated[
        str | None, typer.Option(help="Optional Hugging Face model revision or snapshot hash.")
    ] = None,
    controller_checkpoint: Annotated[
        Path,
        typer.Option(help="Soft-prompt controller checkpoint used unless --base-only is set."),
    ] = DEFAULT_GEMMA_IFEVAL_BEST_CONTROLLER_PATH,
    last_token_controller_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Optional last-token hidden-state controller checkpoint."),
    ] = None,
    run_controller: Annotated[
        bool,
        typer.Option("--controller/--base-only", help="Run controller comparison in addition to base Gemma."),
    ] = True,
    controller_only: Annotated[
        bool,
        typer.Option(help="Run only the controller variant and skip base generation."),
    ] = False,
    controller_strength: Annotated[
        float, typer.Option(help="Multiplier applied to the loaded Gemma controller.")
    ] = DEFAULT_GEMMA_IFEVAL_CONTROLLER_STRENGTH,
    backend: Annotated[
        str,
        typer.Option(help="Generation backend where possible: auto, mlx, or pytorch."),
    ] = "auto",
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
    system_prompt: Annotated[
        str | None, typer.Option(help="Optional system prompt for chat formatting.")
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option(help="Maximum tokens to generate per IFEval prompt.")
    ] = 512,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature. Use 0 for greedy decoding.")
    ] = 0.0,
    top_k: Annotated[
        int, typer.Option(help="Top-k cutoff used when sampling.")
    ] = 50,
    max_examples: Annotated[
        int | None, typer.Option(help="Optional limit for smoke runs.")
    ] = None,
    prompt_repeats: Annotated[
        int,
        typer.Option(
            help=(
                "Number of times to repeat each IFEval prompt for generation. "
                "Scoring still uses the original prompt."
            )
        ),
    ] = 1,
    save_every: Annotated[
        int, typer.Option(help="How often to refresh partial eval artifacts and summaries.")
    ] = 10,
) -> None:
    if max_new_tokens <= 0:
        raise typer.BadParameter("--max-new-tokens must be greater than 0.")
    if controller_strength < 0:
        raise typer.BadParameter("--controller-strength must be greater than or equal to 0.")
    if save_every <= 0:
        raise typer.BadParameter("--save-every must be greater than 0.")
    if prompt_repeats <= 0:
        raise typer.BadParameter("--prompt-repeats must be greater than 0.")
    if controller_only and not run_controller:
        raise typer.BadParameter("--controller-only cannot be combined with --base-only.")
    if last_token_controller_checkpoint is not None and not run_controller:
        raise typer.BadParameter(
            "--last-token-controller-checkpoint cannot be combined with --base-only."
        )
    if (
        run_controller
        and last_token_controller_checkpoint is None
        and not controller_checkpoint.exists()
    ):
        raise typer.BadParameter(
            f"Controller checkpoint not found at {controller_checkpoint}. "
            "Train the controller first or pass --base-only."
        )
    if (
        last_token_controller_checkpoint is not None
        and not last_token_controller_checkpoint.exists()
    ):
        raise typer.BadParameter(
            f"Last-token controller checkpoint not found at {last_token_controller_checkpoint}."
        )

    base_backend = (
        None
        if controller_only
        else resolve_backend(
            backend,
            needs_controller_hooks=False,
        )
    )
    controller_backend = (
        resolve_backend(
            backend,
            needs_controller_hooks=last_token_controller_checkpoint is None,
        )
        if run_controller
        else None
    )

    ensure_nltk_punkt()

    inputs = evaluation_lib.read_prompt_list(input_data_path)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    if not inputs:
        raise typer.BadParameter("IFEval input set is empty.")
    if prompt_repeats > 1:
        status(
            f"Repeating each IFEval prompt {prompt_repeats}x for generation; "
            "scoring uses the original prompts."
        )

    torch_runtime: GemmaSoftPromptRuntime | None = None
    base_mlx_runtime: GemmaMlxRuntime | None = None
    controller_mlx_runtime: GemmaMlxRuntime | None = None
    if base_backend == "mlx":
        status("Loading Gemma MLX runtime for base generation...")
        base_mlx_runtime = GemmaMlxRuntime(
            model_id=model_id,
            revision=revision,
            local_dir=local_dir,
        )
    if controller_backend == "mlx":
        if base_mlx_runtime is None:
            status("Loading Gemma MLX runtime for controller generation...")
            controller_mlx_runtime = GemmaMlxRuntime(
                model_id=model_id,
                revision=revision,
                local_dir=local_dir,
            )
        else:
            controller_mlx_runtime = base_mlx_runtime
    if base_backend == "pytorch" or controller_backend == "pytorch":
        torch_runtime = GemmaSoftPromptRuntime(
            model_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=local_dir,
            offline=offline,
            device=device,
            dtype=dtype,
        )
    controller: PromptPoolingSoftPromptController | None = None
    last_token_controller: LastTokenHiddenStateController | None = None
    mlx_last_token_controller = None
    if run_controller:
        if last_token_controller_checkpoint is None:
            if torch_runtime is None:
                raise RuntimeError("Soft-prompt controller generation requires PyTorch.")
            status("Loading Gemma soft-prompt controller checkpoint...")
            controller = load_gemma_controller_checkpoint(
                controller_checkpoint,
                device=torch_runtime.device,
                dtype=torch.float32,
            )
        else:
            if controller_backend == "mlx":
                status("Loading Gemma MLX last-token controller checkpoint...")
                mlx_last_token_controller = load_mlx_last_token_controller_checkpoint(
                    last_token_controller_checkpoint
                )
            else:
                if torch_runtime is None:
                    raise RuntimeError(
                        "Last-token controller generation requires a runtime."
                    )
                status("Loading Gemma last-token hidden-state controller checkpoint...")
                last_token_controller = load_last_token_controller_checkpoint(
                    last_token_controller_checkpoint,
                    device=torch_runtime.device,
                    dtype=torch.float32,
                )

    total_examples = len(inputs)
    base_records: list[dict[str, str]] = []
    controller_records: list[dict[str, str]] = []
    base_prompt_to_response: dict[str, str] = {}
    controller_prompt_to_response: dict[str, str] = {}
    base_dir = output_dir / "base"
    controller_dir = output_dir / "controller"
    started_at = perf_counter()
    run_config = {
        "input_data_path": str(input_data_path),
        "model_id": model_id,
        "revision": revision,
        "run_controller": run_controller,
        "controller_only": controller_only,
        "controller_kind": (
            None
            if not run_controller
            else (
                "last_token_hidden_state"
                if last_token_controller_checkpoint is not None
                else "soft_prompt"
            )
        ),
        "requested_backend": backend,
        "base_backend": base_backend,
        "controller_backend": controller_backend,
        "local_dir": local_dir,
        "cache_dir": cache_dir,
        "offline": offline,
        "device": device,
        "dtype": dtype,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "max_examples": max_examples,
        "save_every": save_every,
        "prompt_repeats": prompt_repeats,
    }

    if not controller_only:
        persist_ifeval_progress(
            inputs=inputs,
            completed=0,
            records=base_records,
            prompt_to_response=base_prompt_to_response,
            output_dir=base_dir,
            total_examples=total_examples,
            elapsed_seconds=0.0,
            write_partial_eval=False,
        )
        write_json(base_dir / "run_config.json", run_config)
    has_controller_generation = (
        controller is not None
        or last_token_controller is not None
        or mlx_last_token_controller is not None
    )

    if has_controller_generation:
        persist_ifeval_progress(
            inputs=inputs,
            completed=0,
            records=controller_records,
            prompt_to_response=controller_prompt_to_response,
            output_dir=controller_dir,
            total_examples=total_examples,
            elapsed_seconds=0.0,
            write_partial_eval=False,
        )
        write_json(
            controller_dir / "run_config.json",
            {
                **run_config,
                "controller_checkpoint": str(
                    last_token_controller_checkpoint or controller_checkpoint
                ),
                "controller_strength": controller_strength,
            },
        )

    for idx, inp in enumerate(inputs, start=1):
        generation_prompt = repeat_prompt(inp.prompt, prompt_repeats)
        base_progress: dict | None = None
        if not controller_only:
            status(f"generating base {idx}/{total_examples}...")
            if base_backend == "mlx":
                if base_mlx_runtime is None:
                    raise RuntimeError("MLX base runtime was not initialized.")
                base_response = base_mlx_runtime.generate(
                    prompt=generation_prompt,
                    system_prompt=system_prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            else:
                if torch_runtime is None:
                    raise RuntimeError("PyTorch base runtime was not initialized.")
                base_response = torch_runtime.generate(
                    prompt=generation_prompt,
                    system_prompt=system_prompt,
                    controller=None,
                    controller_strength=0.0,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            base_records.append(
                build_response_record(
                    prompt=inp.prompt,
                    generation_prompt=generation_prompt,
                    response=base_response,
                )
            )
            base_prompt_to_response[inp.prompt] = base_response

        if has_controller_generation:
            status(f"generating controller {idx}/{total_examples}...")
            if mlx_last_token_controller is not None:
                if controller_mlx_runtime is None:
                    raise RuntimeError("MLX controller runtime was not initialized.")
                controller_response = (
                    controller_mlx_runtime.generate_with_last_token_controller(
                        prompt=generation_prompt,
                        system_prompt=system_prompt,
                        controller=mlx_last_token_controller,
                        controller_strength=controller_strength,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )
                )
            else:
                if torch_runtime is None:
                    raise RuntimeError("Controller generation requires PyTorch.")
                controller_response = torch_runtime.generate(
                    prompt=generation_prompt,
                    system_prompt=system_prompt,
                    controller=controller,
                    controller_strength=controller_strength,
                    last_token_controller=last_token_controller,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
            controller_records.append(
                build_response_record(
                    prompt=inp.prompt,
                    generation_prompt=generation_prompt,
                    response=controller_response,
                )
            )
            controller_prompt_to_response[inp.prompt] = controller_response

        elapsed_seconds = perf_counter() - started_at
        should_write_partial_eval = idx % save_every == 0 or idx == total_examples

        if not controller_only:
            base_progress = persist_ifeval_progress(
                inputs=inputs,
                completed=idx,
                records=base_records,
                prompt_to_response=base_prompt_to_response,
                output_dir=base_dir,
                total_examples=total_examples,
                elapsed_seconds=elapsed_seconds,
                write_partial_eval=should_write_partial_eval,
            )
        controller_progress: dict | None = None
        if has_controller_generation:
            controller_progress = persist_ifeval_progress(
                inputs=inputs,
                completed=idx,
                records=controller_records,
                prompt_to_response=controller_prompt_to_response,
                output_dir=controller_dir,
                total_examples=total_examples,
                elapsed_seconds=elapsed_seconds,
                write_partial_eval=should_write_partial_eval,
            )

        label = "generated"
        if controller_only:
            label += " for controller"
        elif has_controller_generation:
            label += " for base+controller"
        progress_message = build_progress_message(
            completed=idx,
            total=total_examples,
            elapsed_seconds=elapsed_seconds,
            label=label,
        )
        if should_write_partial_eval:
            if base_progress is not None:
                progress_message += (
                    f" | base partial final {base_progress['partial_final_score']:.4f}"
                )
            if controller_progress is not None:
                progress_message += (
                    " | controller partial final "
                    f"{controller_progress['partial_final_score']:.4f}"
                )
        status(progress_message)

    base_summary: dict | None = None
    if not controller_only:
        write_prompt_responses(base_dir / "responses.jsonl", base_records)
        base_summary = evaluate_prompt_responses(
            inputs,
            base_prompt_to_response,
            output_dir=base_dir,
        )
        status(
            "base strict_prompt_accuracy="
            f"{base_summary['strict']['prompt_accuracy']:.4f} "
            "strict_instruction_accuracy="
            f"{base_summary['strict']['instruction_accuracy']:.4f}"
        )

    controller_summary: dict | None = None
    if has_controller_generation:
        write_prompt_responses(controller_dir / "responses.jsonl", controller_records)
        controller_summary = evaluate_prompt_responses(
            inputs,
            controller_prompt_to_response,
            output_dir=controller_dir,
        )
        persist_ifeval_progress(
            inputs=inputs,
            completed=total_examples,
            records=controller_records,
            prompt_to_response=controller_prompt_to_response,
            output_dir=controller_dir,
            total_examples=total_examples,
            elapsed_seconds=perf_counter() - started_at,
            write_partial_eval=True,
        )
        status(
            "controller strict_prompt_accuracy="
            f"{controller_summary['strict']['prompt_accuracy']:.4f} "
            "strict_instruction_accuracy="
            f"{controller_summary['strict']['instruction_accuracy']:.4f}"
        )

    if not controller_only:
        persist_ifeval_progress(
            inputs=inputs,
            completed=total_examples,
            records=base_records,
            prompt_to_response=base_prompt_to_response,
            output_dir=base_dir,
            total_examples=total_examples,
            elapsed_seconds=perf_counter() - started_at,
            write_partial_eval=True,
        )

    final_parts: list[str] = []
    if base_summary is not None:
        final_parts.append(f"base={compute_final_score(base_summary):.4f}")
    if controller_summary is not None:
        final_parts.append(f"controller={compute_final_score(controller_summary):.4f}")
    status("final score summary: " + " ".join(final_parts))


def run() -> None:
    app()


if __name__ == "__main__":
    run()
