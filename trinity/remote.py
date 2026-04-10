from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated
from urllib import request

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def build_generation_payload(
    *,
    prompt: str,
    controller_checkpoint: Path | None,
    controller_strength: float,
    system_prompt: str | None,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    raw_prompt: bool,
) -> dict:
    return {
        "prompt": prompt,
        "controller_checkpoint": (
            None if controller_checkpoint is None else str(controller_checkpoint)
        ),
        "controller_strength": controller_strength,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "raw_prompt": raw_prompt,
    }


@app.command()
def infer(
    prompt: Annotated[str, typer.Argument(help="User prompt to run through the remote server.")],
    server_url: Annotated[
        str, typer.Option(help="Base URL of the Trinity inference server.")
    ] = "http://127.0.0.1:8000",
    controller_checkpoint: Annotated[
        Path | None,
        typer.Option(help="Optional controller checkpoint path resolved by the server."),
    ] = None,
    controller_strength: Annotated[
        float,
        typer.Option(help="Multiplier applied to controller router biases during inference."),
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
    payload = build_generation_payload(
        prompt=prompt,
        controller_checkpoint=controller_checkpoint,
        controller_strength=controller_strength,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        raw_prompt=raw_prompt,
    )
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        f"{server_url.rstrip('/')}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        raise typer.BadParameter(f"Remote inference request failed: {exc}") from exc

    parsed = json.loads(raw)
    text = parsed.get("text")
    if not isinstance(text, str):
        error = parsed.get("error", "Remote server returned an invalid response.")
        raise typer.BadParameter(str(error))
    sys.stdout.write(text + "\n")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
