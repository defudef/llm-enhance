from __future__ import annotations

import json
import re
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


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


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


def generate_completion(
    model: AfmoeForCausalLM,
    tokenizer: AfmoeTokenizer,
    prompt_text: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt_text)],
        device=device,
        dtype=torch.long,
    )
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
    generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
    if eos_token_id is not None:
        generated_ids = [token for token in generated_ids if token != eos_token_id]
    return tokenizer.decode(generated_ids, skip_special_tokens=False).strip()


def generate_completion_with_controller(
    model: TrinityWithController,
    tokenizer: AfmoeTokenizer,
    prompt_text: str,
    *,
    device: torch.device,
    controller_strength: float,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    input_ids = torch.tensor(
        [tokenizer.encode(prompt_text)],
        device=device,
        dtype=torch.long,
    )
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    output_ids = model.generate(
        input_ids,
        controller_input_ids=input_ids,
        controller_strength=controller_strength,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
    generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
    if eos_token_id is not None:
        generated_ids = [token for token in generated_ids if token != eos_token_id]
    return tokenizer.decode(generated_ids, skip_special_tokens=False).strip()


@app.command()
def evaluate(
    dataset_path: Annotated[
        Path, typer.Argument(help="JSONL dataset with prompt and response fields.")
    ],
    controller_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Optional controller checkpoint. If omitted, only base model is evaluated."),
    ] = None,
    results_path: Annotated[
        Path | None,
        typer.Option(help="Optional path to save per-example results as JSONL."),
    ] = None,
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
            help="Multiplier applied to controller router biases during controller eval."
        ),
    ] = 0.2,
    max_new_tokens: Annotated[
        int, typer.Option(help="Maximum number of tokens to generate per example.")
    ] = 16,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature. Use 0 for greedy decoding.")
    ] = 0.0,
    top_k: Annotated[
        int, typer.Option(help="Top-k cutoff used when sampling.")
    ] = 50,
    max_examples: Annotated[
        int | None, typer.Option(help="Optional limit for quick smoke runs.")
    ] = None,
) -> None:
    rows = load_jsonl(dataset_path)
    if max_examples is not None:
        rows = rows[:max_examples]
    if not rows:
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

    status(f"Initializing base Trinity on {device_obj.type}...")
    base_model = AfmoeForCausalLM(config, device=device_obj, dtype=param_dtype)
    load_checkpoint_into_model(base_model, repo_dir)
    base_model.eval()

    controller_model: TrinityWithController | None = None
    if controller_checkpoint is not None:
        status("Loading controller checkpoint...")
        controller = load_controller_checkpoint(
            controller_checkpoint,
            device=device_obj,
            dtype=torch.float32,
        )
        controller_model = TrinityWithController(
            base_model,
            controller,
            freeze_base_model=True,
        )
        controller_model.eval()

    results: list[dict] = []
    base_exact = 0
    controller_exact = 0

    for idx, row in enumerate(rows, start=1):
        prompt = row["prompt"]
        target = row["response"]
        system_prompt = row.get("system_prompt")
        prompt_text = tokenizer.apply_chat_template(prompt, system_prompt=system_prompt)

        base_output = generate_completion(
            base_model,
            tokenizer,
            prompt_text,
            device=device_obj,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        base_match = normalize_text(base_output) == normalize_text(target)
        base_exact += int(base_match)

        record = {
            "prompt": prompt,
            "target": target,
            "base_output": base_output,
            "base_exact_match": base_match,
        }

        if controller_model is not None:
            controller_output = generate_completion_with_controller(
                controller_model,
                tokenizer,
                prompt_text,
                device=device_obj,
                controller_strength=controller_strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            controller_match = (
                normalize_text(controller_output) == normalize_text(target)
            )
            controller_exact += int(controller_match)
            record["controller_output"] = controller_output
            record["controller_exact_match"] = controller_match

        results.append(record)
        status(
            f"example {idx}/{len(rows)} base_exact={int(base_match)}"
            + (
                ""
                if controller_model is None
                else f" controller_exact={int(record['controller_exact_match'])}"
            )
        )

    base_accuracy = base_exact / len(rows)
    summary = {
        "examples": len(rows),
        "base_exact_match": base_accuracy,
    }
    if controller_model is not None:
        summary["controller_exact_match"] = controller_exact / len(rows)
        summary["delta_exact_match"] = summary["controller_exact_match"] - base_accuracy

    typer.echo(json.dumps(summary, indent=2))

    if results_path is not None:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            "\n".join(json.dumps(record, ensure_ascii=False) for record in results) + "\n"
        )
        status(f"Saved per-example results to {results_path}")


if __name__ == "__main__":
    app()
