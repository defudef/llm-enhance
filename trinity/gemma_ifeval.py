from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Annotated

import nltk
import torch
import typer

from instruction_following_eval import evaluation_lib

from .dense_controller import PromptPoolingSoftPromptController
from .gemma_runtime import (
    DEFAULT_GEMMA_MODEL_ID,
    GemmaSoftPromptRuntime,
    load_gemma_controller_checkpoint,
)
from .ifeval import (
    build_progress_message,
    compute_final_score,
    default_ifeval_input_path,
    evaluate_prompt_responses,
    persist_ifeval_progress,
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


@app.command()
def run_ifeval(
    input_data_path: Annotated[
        Path,
        typer.Option(help="IFEval input jsonl. Defaults to the vendored official input_data.jsonl."),
    ] = default_ifeval_input_path(),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where generated responses and IFEval reports are saved."),
    ] = Path("artifacts/gemma-ifeval"),
    model_id: Annotated[
        str, typer.Option(help="Gemma model id on Hugging Face.")
    ] = DEFAULT_GEMMA_MODEL_ID,
    controller_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Optional soft-prompt controller checkpoint. If omitted, only base Gemma is benchmarked."),
    ] = None,
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
    save_every: Annotated[
        int, typer.Option(help="How often to refresh partial eval artifacts and summaries.")
    ] = 10,
) -> None:
    if max_new_tokens <= 0:
        raise typer.BadParameter("--max-new-tokens must be greater than 0.")
    if save_every <= 0:
        raise typer.BadParameter("--save-every must be greater than 0.")
    if controller_checkpoint is not None and not controller_checkpoint.exists():
        raise typer.BadParameter(
            f"Controller checkpoint not found at {controller_checkpoint}."
        )

    ensure_nltk_punkt()

    inputs = evaluation_lib.read_prompt_list(input_data_path)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    if not inputs:
        raise typer.BadParameter("IFEval input set is empty.")

    runtime = GemmaSoftPromptRuntime(
        model_id=model_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
        device=device,
        dtype=dtype,
    )
    controller: PromptPoolingSoftPromptController | None = None
    if controller_checkpoint is not None:
        status("Loading Gemma soft-prompt controller checkpoint...")
        controller = load_gemma_controller_checkpoint(
            controller_checkpoint,
            device=runtime.device,
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
    if controller is not None:
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

    for idx, inp in enumerate(inputs, start=1):
        base_response = runtime.generate(
            prompt=inp.prompt,
            system_prompt=system_prompt,
            controller=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        base_records.append({"prompt": inp.prompt, "response": base_response})
        base_prompt_to_response[inp.prompt] = base_response

        if controller is not None:
            controller_response = runtime.generate(
                prompt=inp.prompt,
                system_prompt=system_prompt,
                controller=controller,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            controller_records.append(
                {"prompt": inp.prompt, "response": controller_response}
            )
            controller_prompt_to_response[inp.prompt] = controller_response

        elapsed_seconds = perf_counter() - started_at
        should_write_partial_eval = idx % save_every == 0 or idx == total_examples

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
        if controller is not None:
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
        if controller is not None:
            label += " for base+controller"
        progress_message = build_progress_message(
            completed=idx,
            total=total_examples,
            elapsed_seconds=elapsed_seconds,
            label=label,
        )
        if should_write_partial_eval:
            progress_message += (
                f" | base partial final {base_progress['partial_final_score']:.4f}"
            )
            if controller_progress is not None:
                progress_message += (
                    " | controller partial final "
                    f"{controller_progress['partial_final_score']:.4f}"
                )
        status(progress_message)

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

    if controller is not None:
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
    status(
        "final score summary: "
        f"base={compute_final_score(base_summary):.4f}"
        + (
            ""
            if controller is None
            else f" controller={compute_final_score(controller_summary):.4f}"
        )
    )


def run() -> None:
    app()


if __name__ == "__main__":
    run()
