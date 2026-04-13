from __future__ import annotations

import sys

import typer


GEMMA_BACKENDS = ("pytorch", "ollama", "mlx")


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def torch_mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def select_auto_backend(
    *,
    platform: str | None = None,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    current_platform = sys.platform if platform is None else platform
    has_cuda = torch_cuda_available() if cuda_available is None else cuda_available
    if has_cuda:
        return "pytorch"
    has_mps = torch_mps_available() if mps_available is None else mps_available
    if current_platform == "darwin" and has_mps:
        return "mlx"
    return "pytorch"


def select_auto_backends(
    *,
    platform: str | None = None,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> list[str]:
    return [
        select_auto_backend(
            platform=platform,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
    ]


def parse_backend_names(
    backends: str,
    *,
    platform: str | None = None,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> list[str]:
    parsed = [backend.strip().lower() for backend in backends.split(",")]
    parsed = [backend for backend in parsed if backend]
    if not parsed:
        raise typer.BadParameter("At least one backend must be selected.")
    if parsed == ["auto"]:
        return select_auto_backends(
            platform=platform,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
    unknown = sorted(set(parsed) - set(GEMMA_BACKENDS))
    if unknown:
        raise typer.BadParameter(
            f"Unknown backend(s): {', '.join(unknown)}. "
            f"Expected one or more of: {', '.join(GEMMA_BACKENDS)}."
        )
    return parsed


def resolve_backend(
    backend: str,
    *,
    needs_controller_hooks: bool,
    platform: str | None = None,
    cuda_available: bool | None = None,
    mps_available: bool | None = None,
) -> str:
    selected = backend.strip().lower()
    if selected == "auto":
        selected = select_auto_backend(
            platform=platform,
            cuda_available=cuda_available,
            mps_available=mps_available,
        )
    if selected not in {"pytorch", "mlx"}:
        raise typer.BadParameter("--backend must be auto, pytorch, or mlx.")
    if selected == "mlx" and needs_controller_hooks:
        if backend.strip().lower() == "auto":
            return "pytorch"
        raise typer.BadParameter(
            "--backend mlx cannot run Gemma controllers yet; use --backend pytorch."
        )
    return selected
