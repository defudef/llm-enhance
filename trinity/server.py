from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Annotated
from urllib.parse import urlparse

import typer

from .runtime import GenerationConfig, TrinityRuntime

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def status(message: str) -> None:
    typer.secho(message, err=True, fg=typer.colors.BLUE)


@dataclass(slots=True)
class ServerConfig:
    host: str
    port: int
    repo_id: str
    cache_dir: str | None
    local_dir: str | None
    offline: bool
    device: str
    dtype: str


def parse_generation_payload(payload: dict[str, Any]) -> tuple[str, GenerationConfig]:
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("`prompt` must be a non-empty string.")

    max_new_tokens = int(payload.get("max_new_tokens", 16))
    if max_new_tokens <= 0:
        raise ValueError("`max_new_tokens` must be greater than 0.")

    temperature = float(payload.get("temperature", 0.0))
    top_k = int(payload.get("top_k", 50))
    controller_strength = float(payload.get("controller_strength", 0.2))
    system_prompt_raw = payload.get("system_prompt")
    system_prompt = None if system_prompt_raw is None else str(system_prompt_raw)
    raw_prompt = bool(payload.get("raw_prompt", False))
    controller_checkpoint_raw = payload.get("controller_checkpoint")
    controller_checkpoint = (
        None
        if controller_checkpoint_raw in (None, "")
        else Path(str(controller_checkpoint_raw))
    )

    return prompt, GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        controller_strength=controller_strength,
        system_prompt=system_prompt,
        raw_prompt=raw_prompt,
        controller_checkpoint=controller_checkpoint,
    )


def make_handler(
    runtime: TrinityRuntime,
    generation_lock: threading.Lock,
    config: ServerConfig,
) -> type[BaseHTTPRequestHandler]:
    class TrinityRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if parsed.path == "/info":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "repo_id": config.repo_id,
                        "device": runtime.device.type,
                        "dtype": str(runtime.param_dtype).replace("torch.", ""),
                        "controller_cache_size": len(runtime._controller_cache),
                    },
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/generate":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Request body must be a JSON object.")
                prompt, generation_config = parse_generation_payload(payload)
                started_at = time.perf_counter()
                status(
                    "Received /generate request"
                    + (
                        f" with controller={generation_config.controller_checkpoint}"
                        if generation_config.controller_checkpoint is not None
                        else " with base model"
                    )
                )
                with generation_lock:
                    text = runtime.generate(prompt, config=generation_config)
                elapsed_s = time.perf_counter() - started_at
                status(f"Completed /generate request in {elapsed_s:.2f}s.")
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "text": text,
                        "used_controller": generation_config.controller_checkpoint
                        is not None,
                    },
                )
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": str(exc)},
                )

    return TrinityRequestHandler


@app.command()
def serve(
    host: Annotated[
        str, typer.Option(help="Host interface to bind the server to.")
    ] = "127.0.0.1",
    port: Annotated[
        int, typer.Option(help="TCP port to bind the server to.")
    ] = 8000,
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
) -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    runtime = TrinityRuntime(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
        device=device,
        dtype=dtype,
        status_callback=status,
    )
    generation_lock = threading.Lock()
    server_config = ServerConfig(
        host=host,
        port=port,
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_dir=local_dir,
        offline=offline,
        device=device,
        dtype=dtype,
    )
    httpd = ThreadingHTTPServer(
        (host, port),
        make_handler(runtime, generation_lock, server_config),
    )
    status(
        f"Trinity server listening on http://{host}:{port} "
        f"(device={runtime.device.type}, dtype={str(runtime.param_dtype).replace('torch.', '')})"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        status("Shutting down Trinity server...")
    finally:
        httpd.server_close()


def run() -> None:
    app()


if __name__ == "__main__":
    run()
