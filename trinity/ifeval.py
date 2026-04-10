from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import nltk
import torch
import typer

from instruction_following_eval import evaluation_lib

from . import AfmoeConfig, AfmoeForCausalLM, AfmoeTokenizer
from . import ensure_local_repo, load_checkpoint_into_model
from .controller import PromptPoolingController, PromptPoolingControllerConfig
from .wrapper import TrinityWithController

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


def default_ifeval_input_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "instruction_following_eval"
        / "data"
        / "input_data.jsonl"
    )


def ensure_nltk_punkt() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        status("Downloading NLTK punkt tokenizer required by IFEval...")
        if not nltk.download("punkt", quiet=True):
            raise RuntimeError(
                "Failed to download NLTK punkt tokenizer required by IFEval."
            )


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


def write_prompt_responses(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n"
    )


def build_accuracy_report(outputs: list[evaluation_lib.OutputExample]) -> dict:
    prompt_total = len(outputs)
    prompt_correct = sum(int(output.follow_all_instructions) for output in outputs)
    instruction_total = sum(len(output.follow_instruction_list) for output in outputs)
    instruction_correct = sum(
        sum(int(followed) for followed in output.follow_instruction_list)
        for output in outputs
    )

    tier0_total: dict[str, int] = {}
    tier0_correct: dict[str, int] = {}
    tier1_total: dict[str, int] = {}
    tier1_correct: dict[str, int] = {}

    for output in outputs:
        for instruction_id, followed in zip(
            output.instruction_id_list, output.follow_instruction_list
        ):
            tier1_total[instruction_id] = tier1_total.get(instruction_id, 0) + 1
            tier1_correct[instruction_id] = tier1_correct.get(instruction_id, 0) + int(
                followed
            )

            tier0_id = instruction_id.split(":")[0]
            tier0_total[tier0_id] = tier0_total.get(tier0_id, 0) + 1
            tier0_correct[tier0_id] = tier0_correct.get(tier0_id, 0) + int(followed)

    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": 0.0 if prompt_total == 0 else prompt_correct / prompt_total,
        "instruction_total": instruction_total,
        "instruction_correct": instruction_correct,
        "instruction_accuracy": (
            0.0 if instruction_total == 0 else instruction_correct / instruction_total
        ),
        "tier0": {
            key: {
                "correct": tier0_correct[key],
                "total": tier0_total[key],
                "accuracy": tier0_correct[key] / tier0_total[key],
            }
            for key in sorted(tier0_total)
        },
        "tier1": {
            key: {
                "correct": tier1_correct[key],
                "total": tier1_total[key],
                "accuracy": tier1_correct[key] / tier1_total[key],
            }
            for key in sorted(tier1_total)
        },
    }


def evaluate_prompt_responses(
    inputs: list[evaluation_lib.InputExample],
    prompt_to_response: dict[str, str],
    *,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    strict_outputs = [
        evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
        for inp in inputs
    ]
    loose_outputs = [
        evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
        for inp in inputs
    ]

    evaluation_lib.write_outputs(output_dir / "eval_results_strict.jsonl", strict_outputs)
    evaluation_lib.write_outputs(output_dir / "eval_results_loose.jsonl", loose_outputs)

    summary = {
        "examples": len(inputs),
        "strict": build_accuracy_report(strict_outputs),
        "loose": build_accuracy_report(loose_outputs),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


@app.command()
def run_ifeval(
    input_data_path: Annotated[
        Path,
        typer.Option(help="IFEval input jsonl. Defaults to the vendored official input_data.jsonl."),
    ] = default_ifeval_input_path(),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where generated responses and IFEval reports are saved."),
    ] = Path("artifacts/ifeval"),
    controller_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Optional controller checkpoint. If omitted, only base model is benchmarked."),
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
        typer.Option(help="Multiplier applied to controller router biases during generation."),
    ] = 0.2,
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
) -> None:
    if max_new_tokens <= 0:
        raise typer.BadParameter("--max-new-tokens must be greater than 0.")
    if controller_checkpoint is not None and not controller_checkpoint.exists():
        raise typer.BadParameter(
            f"Controller checkpoint not found at {controller_checkpoint}. "
            "Train one first or pass --controller-checkpoint."
        )

    ensure_nltk_punkt()

    inputs = evaluation_lib.read_prompt_list(input_data_path)
    if max_examples is not None:
        inputs = inputs[:max_examples]
    if not inputs:
        raise typer.BadParameter("IFEval input set is empty.")

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

    base_records: list[dict[str, str]] = []
    controller_records: list[dict[str, str]] = []
    base_prompt_to_response: dict[str, str] = {}
    controller_prompt_to_response: dict[str, str] = {}

    for idx, inp in enumerate(inputs, start=1):
        prompt_text = tokenizer.apply_chat_template(
            inp.prompt,
            system_prompt=system_prompt,
        )

        base_response = generate_completion(
            base_model,
            tokenizer,
            prompt_text,
            device=device_obj,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        base_records.append({"prompt": inp.prompt, "response": base_response})
        base_prompt_to_response[inp.prompt] = base_response

        if controller_model is not None:
            controller_response = generate_completion_with_controller(
                controller_model,
                tokenizer,
                prompt_text,
                device=device_obj,
                controller_strength=controller_strength,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            controller_records.append(
                {"prompt": inp.prompt, "response": controller_response}
            )
            controller_prompt_to_response[inp.prompt] = controller_response

        status(
            f"generated {idx}/{len(inputs)}"
            + ("" if controller_model is None else " for base+controller")
        )

    base_dir = output_dir / "base"
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

    if controller_model is not None:
        controller_dir = output_dir / "controller"
        write_prompt_responses(controller_dir / "responses.jsonl", controller_records)
        controller_summary = evaluate_prompt_responses(
            inputs,
            controller_prompt_to_response,
            output_dir=controller_dir,
        )
        status(
            "controller strict_prompt_accuracy="
            f"{controller_summary['strict']['prompt_accuracy']:.4f} "
            "strict_instruction_accuracy="
            f"{controller_summary['strict']['instruction_accuracy']:.4f}"
        )


def run() -> None:
    app()


if __name__ == "__main__":
    run()
