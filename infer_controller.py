from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import torch
import typer

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
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available in this PyTorch build.")
        return torch.device("mps")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this PyTorch build.")
        return torch.device("cuda")
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


def load_controller_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PromptPoolingController:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    controller = PromptPoolingController(
        PromptPoolingControllerConfig(**checkpoint["controller_config"])
    ).to(device=device, dtype=dtype)
    controller.load_state_dict(checkpoint["controller_state_dict"])
    controller.eval()
    return controller


@app.command()
def infer(
    prompt: Annotated[str, typer.Argument(help="User prompt to run through the controller-enhanced model.")],
    controller_checkpoint: Annotated[
        Path, typer.Option(help="Path to a saved controller checkpoint.")
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
        str, typer.Option(help="Parameter dtype: auto, float16, bfloat16, or float32.")
    ] = "auto",
    controller_strength: Annotated[
        float,
        typer.Option(
            help="Multiplier applied to controller router biases during inference."
        ),
    ] = 0.2,
    system_prompt: Annotated[
        str | None, typer.Option(help="Optional system prompt for chat formatting.")
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens to generate.")
    ] = 16,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature. Use 0 for greedy decoding.")
    ] = 0.0,
    top_k: Annotated[
        int, typer.Option(help="Top-k cutoff used when sampling.")
    ] = 50,
    raw_prompt: Annotated[
        bool, typer.Option(help="Treat prompt as already formatted for the model.")
    ] = False,
) -> None:
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

    prompt_text = prompt
    if not raw_prompt:
        prompt_text = tokenizer.apply_chat_template(prompt, system_prompt=system_prompt)

    status(f"Initializing frozen Trinity on {device_obj.type}...")
    base_model = AfmoeForCausalLM(config, device=device_obj, dtype=param_dtype)
    load_checkpoint_into_model(base_model, repo_dir)
    controller = load_controller_checkpoint(
        controller_checkpoint,
        device=device_obj,
        dtype=torch.float32,
    )
    model = TrinityWithController(base_model, controller, freeze_base_model=True)

    input_ids = torch.tensor(
        [tokenizer.encode(prompt_text)],
        device=device_obj,
        dtype=torch.long,
    )
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)

    status(f"Generating up to {max_new_tokens} token(s) with controller...")

    def on_token(next_token: torch.Tensor) -> None:
        token_id = int(next_token[0, 0].item())
        if eos_token_id is not None and token_id == eos_token_id:
            return
        sys.stdout.write(tokenizer.decode([token_id], skip_special_tokens=False))
        sys.stdout.flush()

    _ = model.generate(
        input_ids,
        controller_input_ids=input_ids,
        controller_strength=controller_strength,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
        token_callback=on_token,
    )
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    app()
