from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any
from urllib import error, request

import typer

from instruction_following_eval import evaluation_lib

from .gemma_backend import parse_backend_names
from .gemma_mlx_runtime import (
    GemmaMlxRuntime,
    build_mlx_prompt,
    count_tokens,
)
from .gemma_runtime import DEFAULT_GEMMA_MODEL_ID, GemmaSoftPromptRuntime, build_prompt_text
from .ifeval_common import default_ifeval_input_path, write_json

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

DEFAULT_OUTPUT_DIR = Path("artifacts/gemma-speed-bench")
DEFAULT_OLLAMA_MODEL = "gemma4:e2b"


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


@dataclass(slots=True)
class SpeedBenchConfig:
    model_id: str
    revision: str | None
    cache_dir: str | None
    local_dir: str | None
    offline: bool
    device: str
    dtype: str
    system_prompt: str | None
    max_new_tokens: int
    temperature: float
    top_k: int
    ollama_model: str
    ollama_url: str
    ollama_think: bool
    mlx_model: str
    mlx_trust_remote_code: bool


def select_ifeval_prompts(
    input_data_path: Path,
    *,
    max_prompts: int,
    offset: int,
) -> list[evaluation_lib.InputExample]:
    if max_prompts <= 0:
        raise typer.BadParameter("--max-prompts must be greater than 0.")
    if offset < 0:
        raise typer.BadParameter("--offset must be greater than or equal to 0.")
    prompts = evaluation_lib.read_prompt_list(input_data_path)
    selected = prompts[offset : offset + max_prompts]
    if not selected:
        raise typer.BadParameter("Selected IFEval prompt subset is empty.")
    return selected


def rate(numerator: int | float | None, seconds: float | None) -> float | None:
    if numerator is None or seconds is None or seconds <= 0:
        return None
    return float(numerator) / seconds


def summarize_backend_result(
    *,
    backend: str,
    status_value: str,
    load_seconds: float | None,
    prompt_results: list[dict],
    error_message: str | None = None,
    load_seconds_in_generation: bool = False,
) -> dict:
    generation_seconds = sum(result["seconds"] for result in prompt_results)
    prompt_tokens = sum(
        result["prompt_tokens"]
        for result in prompt_results
        if isinstance(result.get("prompt_tokens"), int)
    )
    generated_tokens = sum(
        result["generated_tokens"]
        for result in prompt_results
        if isinstance(result.get("generated_tokens"), int)
    )
    measured_prompts = len(prompt_results)
    total_seconds = (
        generation_seconds
        if load_seconds_in_generation
        else generation_seconds + (load_seconds or 0.0)
    )
    return {
        "backend": backend,
        "status": status_value,
        "error": error_message,
        "load_seconds": load_seconds,
        "load_seconds_in_generation": load_seconds_in_generation,
        "generation_seconds": generation_seconds,
        "total_seconds": total_seconds,
        "prompts": measured_prompts,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "seconds_per_prompt": (
            None if measured_prompts == 0 else generation_seconds / measured_prompts
        ),
        "generated_tokens_per_second_wall": rate(
            generated_tokens,
            generation_seconds,
        ),
        "prompt_tokens_per_second_wall": rate(prompt_tokens, generation_seconds),
        "per_prompt": prompt_results,
        "hook_support": BACKEND_HOOK_SUPPORT[backend],
    }


def build_prompt_record(
    *,
    index: int,
    inp: evaluation_lib.InputExample,
    response: str,
    seconds: float,
    prompt_tokens: int | None,
    generated_tokens: int | None,
    extra: dict | None = None,
) -> dict:
    record = {
        "index": index,
        "ifeval_key": inp.key,
        "prompt_chars": len(inp.prompt),
        "response_chars": len(response),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "seconds": seconds,
        "generated_tokens_per_second_wall": rate(generated_tokens, seconds),
        "response_preview": response[:200],
    }
    if extra:
        record.update(extra)
    return record


def benchmark_pytorch(
    prompts: list[evaluation_lib.InputExample],
    config: SpeedBenchConfig,
) -> dict:
    load_start = perf_counter()
    runtime = GemmaSoftPromptRuntime(
        model_id=config.model_id,
        revision=config.revision,
        cache_dir=config.cache_dir,
        local_dir=config.local_dir,
        offline=config.offline,
        device=config.device,
        dtype=config.dtype,
    )
    load_seconds = perf_counter() - load_start
    prompt_results: list[dict] = []
    for index, inp in enumerate(prompts, start=1):
        prompt_text = build_prompt_text(
            runtime.tokenizer,
            prompt=inp.prompt,
            system_prompt=config.system_prompt,
        )
        prompt_tokens = count_tokens(runtime.tokenizer, prompt_text)
        started_at = perf_counter()
        response = runtime.generate(
            prompt=inp.prompt,
            system_prompt=config.system_prompt,
            controller=None,
            controller_strength=0.0,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        seconds = perf_counter() - started_at
        generated_tokens = count_tokens(runtime.tokenizer, response)
        prompt_results.append(
            build_prompt_record(
                index=index,
                inp=inp,
                response=response,
                seconds=seconds,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
            )
        )
    return summarize_backend_result(
        backend="pytorch",
        status_value="ok",
        load_seconds=load_seconds,
        prompt_results=prompt_results,
    )


def post_ollama_generate(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    think: bool,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        },
    }
    if system_prompt is not None:
        payload["system"] = system_prompt
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint.rstrip("/") + "/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def ns_to_seconds(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def benchmark_ollama(
    prompts: list[evaluation_lib.InputExample],
    config: SpeedBenchConfig,
) -> dict:
    prompt_results: list[dict] = []
    load_seconds_total = 0.0
    for index, inp in enumerate(prompts, start=1):
        started_at = perf_counter()
        response = post_ollama_generate(
            endpoint=config.ollama_url,
            model=config.ollama_model,
            prompt=inp.prompt,
            system_prompt=config.system_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
            think=config.ollama_think,
        )
        seconds = perf_counter() - started_at
        load_seconds_total += ns_to_seconds(response.get("load_duration")) or 0.0
        prompt_tokens = response.get("prompt_eval_count")
        generated_tokens = response.get("eval_count")
        eval_seconds = ns_to_seconds(response.get("eval_duration"))
        prompt_eval_seconds = ns_to_seconds(response.get("prompt_eval_duration"))
        prompt_results.append(
            build_prompt_record(
                index=index,
                inp=inp,
                response=str(response.get("response", "")),
                seconds=seconds,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                extra={
                    "backend_reported_eval_seconds": eval_seconds,
                    "backend_reported_prompt_eval_seconds": prompt_eval_seconds,
                    "backend_reported_generated_tokens_per_second": rate(
                        generated_tokens,
                        eval_seconds,
                    ),
                    "backend_reported_prompt_tokens_per_second": rate(
                        prompt_tokens,
                        prompt_eval_seconds,
                    ),
                    "backend_reported_load_seconds": ns_to_seconds(
                        response.get("load_duration")
                    ),
                    "backend_reported_total_seconds": ns_to_seconds(
                        response.get("total_duration")
                    ),
                    "thinking_chars": len(str(response.get("thinking", ""))),
                },
            )
        )
    return summarize_backend_result(
        backend="ollama",
        status_value="ok",
        load_seconds=load_seconds_total,
        prompt_results=prompt_results,
        load_seconds_in_generation=True,
    )


def benchmark_mlx(
    prompts: list[evaluation_lib.InputExample],
    config: SpeedBenchConfig,
) -> dict:
    load_start = perf_counter()
    runtime = GemmaMlxRuntime(
        model_id=config.mlx_model,
        revision=config.revision,
        local_dir=config.local_dir,
        trust_remote_code=config.mlx_trust_remote_code,
    )
    load_seconds = perf_counter() - load_start

    prompt_results: list[dict] = []
    for index, inp in enumerate(prompts, start=1):
        prompt_text = build_mlx_prompt(
            runtime.tokenizer,
            prompt=inp.prompt,
            system_prompt=config.system_prompt,
        )
        prompt_tokens = count_tokens(runtime.tokenizer, prompt_text)
        started_at = perf_counter()
        response = runtime.generate(
            prompt=inp.prompt,
            system_prompt=config.system_prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        seconds = perf_counter() - started_at
        generated_tokens = count_tokens(runtime.tokenizer, response)
        prompt_results.append(
            build_prompt_record(
                index=index,
                inp=inp,
                response=response,
                seconds=seconds,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
            )
        )
    return summarize_backend_result(
        backend="mlx",
        status_value="ok",
        load_seconds=load_seconds,
        prompt_results=prompt_results,
    )


BACKEND_HOOK_SUPPORT = {
    "pytorch": {
        "hidden_states": "yes",
        "logits_before_sampling": "yes",
        "kv_cache_continuation": "yes",
        "final_answer_parsing_compatibility": "yes",
        "integration_note": "Current research runtime; slow on MPS but exposes the steering hooks directly.",
    },
    "ollama": {
        "hidden_states": "no_public_api",
        "logits_before_sampling": "no_public_api",
        "kv_cache_continuation": "server_managed_only",
        "final_answer_parsing_compatibility": "yes",
        "integration_note": "Useful as a GGUF/Metal speed baseline; not a good target for custom hidden-state steering.",
    },
    "mlx": {
        "hidden_states": "possible_with_python_integration",
        "logits_before_sampling": "possible_with_custom_generation_loop",
        "kv_cache_continuation": "yes",
        "final_answer_parsing_compatibility": "yes",
        "integration_note": "Best Mac-native candidate for faster custom steering if the model is available in MLX format.",
    },
}


def run_backend(
    backend: str,
    prompts: list[evaluation_lib.InputExample],
    config: SpeedBenchConfig,
) -> dict:
    started_at = perf_counter()
    try:
        if backend == "pytorch":
            return benchmark_pytorch(prompts, config)
        if backend == "ollama":
            return benchmark_ollama(prompts, config)
        if backend == "mlx":
            return benchmark_mlx(prompts, config)
    except Exception as exc:
        return summarize_backend_result(
            backend=backend,
            status_value="error",
            load_seconds=None,
            prompt_results=[],
            error_message=str(exc),
        ) | {"failed_after_seconds": perf_counter() - started_at}
    raise ValueError(f"Unsupported backend: {backend}")


@app.command()
def bench(
    input_data_path: Annotated[
        Path,
        typer.Option(help="IFEval input jsonl used for the fixed prompt subset."),
    ] = default_ifeval_input_path(),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory where benchmark summary JSON is written."),
    ] = DEFAULT_OUTPUT_DIR,
    backends: Annotated[
        str,
        typer.Option(help="Comma-separated backends: auto,pytorch,ollama,mlx."),
    ] = "auto",
    max_prompts: Annotated[
        int,
        typer.Option(help="Number of IFEval prompts to benchmark from the fixed subset."),
    ] = 8,
    offset: Annotated[
        int,
        typer.Option(help="Offset into the official IFEval prompt order."),
    ] = 0,
    model_id: Annotated[
        str, typer.Option(help="Hugging Face Gemma model id for the PyTorch backend.")
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
        bool, typer.Option(help="Use only locally cached Hugging Face model files.")
    ] = False,
    device: Annotated[
        str, typer.Option(help="PyTorch device override: auto, cuda, mps, or cpu.")
    ] = "auto",
    dtype: Annotated[
        str, typer.Option(help="PyTorch dtype: auto, float16, bfloat16, or float32.")
    ] = "auto",
    system_prompt: Annotated[
        str | None, typer.Option(help="Optional system prompt for chat formatting.")
    ] = None,
    max_new_tokens: Annotated[
        int, typer.Option(help="Maximum tokens to generate per prompt.")
    ] = 512,
    temperature: Annotated[
        float, typer.Option(help="Sampling temperature for PyTorch and Ollama.")
    ] = 0.0,
    top_k: Annotated[
        int, typer.Option(help="Top-k cutoff for PyTorch and Ollama.")
    ] = 50,
    ollama_model: Annotated[
        str, typer.Option(help="Ollama model tag to benchmark.")
    ] = DEFAULT_OLLAMA_MODEL,
    ollama_url: Annotated[
        str, typer.Option(help="Ollama server URL.")
    ] = "http://localhost:11434",
    ollama_think: Annotated[
        bool,
        typer.Option(help="Enable Ollama thinking tokens when the model supports them."),
    ] = False,
    mlx_model: Annotated[
        str,
        typer.Option(help="MLX model id or local MLX model path."),
    ] = DEFAULT_GEMMA_MODEL_ID,
    mlx_trust_remote_code: Annotated[
        bool,
        typer.Option(help="Pass trust_remote_code to the MLX tokenizer loader."),
    ] = False,
) -> None:
    if max_new_tokens <= 0:
        raise typer.BadParameter("--max-new-tokens must be greater than 0.")
    if temperature < 0:
        raise typer.BadParameter("--temperature must be greater than or equal to 0.")
    if top_k < 0:
        raise typer.BadParameter("--top-k must be greater than or equal to 0.")

    backend_names = parse_backend_names(backends)
    prompts = select_ifeval_prompts(
        input_data_path,
        max_prompts=max_prompts,
        offset=offset,
    )
    config = SpeedBenchConfig(
        model_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
        device=device,
        dtype=dtype,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
        ollama_think=ollama_think,
        mlx_model=mlx_model,
        mlx_trust_remote_code=mlx_trust_remote_code,
    )

    summary = {
        "schema_version": 1,
        "benchmark": "gemma_speed_bench",
        "input_data_path": str(input_data_path),
        "prompt_subset": {
            "source": "IFEval official input order",
            "offset": offset,
            "max_prompts": max_prompts,
            "selected_keys": [inp.key for inp in prompts],
        },
        "config": {
            "model_id": model_id,
            "revision": revision,
            "local_dir": local_dir,
            "offline": offline,
            "device": device,
            "dtype": dtype,
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "ollama_model": ollama_model,
            "ollama_url": ollama_url,
            "ollama_think": ollama_think,
            "mlx_model": mlx_model,
            "mlx_trust_remote_code": mlx_trust_remote_code,
        },
        "backend_hook_support": BACKEND_HOOK_SUPPORT,
        "results": [],
    }

    for backend in backend_names:
        status(f"benchmarking {backend} on {len(prompts)} prompts...")
        result = run_backend(backend, prompts, config)
        summary["results"].append(result)
        if result["status"] == "ok":
            status(
                f"{backend}: {result['seconds_per_prompt']:.2f}s/prompt, "
                f"{result['generated_tokens_per_second_wall']:.2f} generated tok/s wall"
            )
        else:
            status(f"{backend}: error: {result['error']}")

    output_path = output_dir / "summary.json"
    write_json(output_path, summary)
    status(f"Wrote speed benchmark summary to {output_path}")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
