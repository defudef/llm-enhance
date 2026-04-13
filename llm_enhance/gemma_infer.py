from __future__ import annotations

from pathlib import Path
from typing import Annotated

import torch
import typer

from .dense_controller import PromptPoolingSoftPromptController
from .gemma_backend import resolve_backend
from .gemma_mlx_runtime import (
    GemmaMlxRuntime,
    load_mlx_last_token_controller_checkpoint,
)
from .gemma_runtime import (
    DEFAULT_GEMMA_MODEL_ID,
    GemmaSoftPromptRuntime,
    load_gemma_controller_checkpoint,
    load_last_token_controller_checkpoint,
)
from .last_token_controller import LastTokenHiddenStateController

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def infer(
    prompt: Annotated[str, typer.Argument(help="User prompt to run through Gemma.")],
    model_id: Annotated[
        str, typer.Option(help="Gemma model id on Hugging Face.")
    ] = DEFAULT_GEMMA_MODEL_ID,
    revision: Annotated[
        str | None, typer.Option(help="Optional Hugging Face model revision or snapshot hash.")
    ] = None,
    controller_checkpoint: Annotated[
        Path | None, typer.Option(help="Optional soft-prompt controller checkpoint.")
    ] = None,
    last_token_controller_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Optional last-token hidden-state controller checkpoint."),
    ] = None,
    controller_strength: Annotated[
        float, typer.Option(help="Multiplier applied to the loaded Gemma controller.")
    ] = 1.0,
    backend: Annotated[
        str,
        typer.Option(help="Inference backend: auto, mlx, or pytorch."),
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
        int, typer.Option(help="Maximum number of tokens to generate.")
    ] = 128,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature. Use 0 for greedy decoding.")
    ] = 0.0,
    top_k: Annotated[
        int, typer.Option(help="Top-k cutoff used when sampling.")
    ] = 50,
) -> None:
    if controller_checkpoint is not None and not controller_checkpoint.exists():
        raise typer.BadParameter(
            f"Controller checkpoint not found at {controller_checkpoint}."
        )
    if (
        last_token_controller_checkpoint is not None
        and not last_token_controller_checkpoint.exists()
    ):
        raise typer.BadParameter(
            f"Last-token controller checkpoint not found at {last_token_controller_checkpoint}."
        )
    if controller_checkpoint is not None and last_token_controller_checkpoint is not None:
        raise typer.BadParameter(
            "--controller-checkpoint cannot be combined with --last-token-controller-checkpoint."
        )
    if controller_strength < 0:
        raise typer.BadParameter("--controller-strength must be greater than or equal to 0.")

    needs_controller_hooks = controller_checkpoint is not None
    selected_backend = resolve_backend(
        backend,
        needs_controller_hooks=needs_controller_hooks,
    )
    if selected_backend == "mlx":
        runtime = GemmaMlxRuntime(
            model_id=model_id,
            revision=revision,
            local_dir=local_dir,
        )
        if last_token_controller_checkpoint is not None:
            mlx_controller = load_mlx_last_token_controller_checkpoint(
                last_token_controller_checkpoint
            )
            text = runtime.generate_with_last_token_controller(
                prompt=prompt,
                system_prompt=system_prompt,
                controller=mlx_controller,
                controller_strength=controller_strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            text = runtime.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        typer.echo(text)
        return

    runtime = GemmaSoftPromptRuntime(
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
    if controller_checkpoint is not None:
        controller = load_gemma_controller_checkpoint(
            controller_checkpoint,
            device=runtime.device,
            dtype=torch.float32,
        )
    if last_token_controller_checkpoint is not None:
        last_token_controller = load_last_token_controller_checkpoint(
            last_token_controller_checkpoint,
            device=runtime.device,
            dtype=torch.float32,
        )

    text = runtime.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        controller=controller,
        controller_strength=controller_strength,
        last_token_controller=last_token_controller,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    typer.echo(text)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
