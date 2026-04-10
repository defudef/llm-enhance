from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import AfmoeConfig
from .controller import PromptPoolingController, PromptPoolingControllerConfig
from .hf import ensure_local_repo, load_checkpoint_into_model
from .model import AfmoeForCausalLM
from .tokenizer import AfmoeTokenizer
from .wrapper import TrinityWithController


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


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_k: int = 50
    controller_strength: float = 0.2
    system_prompt: str | None = None
    raw_prompt: bool = False
    controller_checkpoint: Path | None = None


@dataclass(slots=True)
class ControllerCacheEntry:
    path: Path
    mtime_ns: int
    controller: PromptPoolingController


class TrinityRuntime:
    def __init__(
        self,
        *,
        repo_id: str,
        cache_dir: str | None,
        local_dir: str | None,
        offline: bool,
        device: str,
        dtype: str,
        status_callback=None,
    ) -> None:
        self._status = status_callback or (lambda _message: None)
        device_name = device.lower()
        dtype_name = dtype.lower()
        self._status(
            f"Resolving runtime device and dtype (device={device_name}, dtype={dtype_name})..."
        )
        self.device = resolve_device(device_name)
        self.param_dtype = resolve_dtype(dtype_name, self.device)
        self.repo_id = repo_id
        self._status(f"Resolving Trinity repo files for {repo_id}...")
        self.repo_dir = ensure_local_repo(
            repo_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            offline=offline,
        )
        self._status(f"Loading Trinity config from {self.repo_dir}...")
        self.config = AfmoeConfig.from_json_file(self.repo_dir / "config.json")
        self.tokenizer = AfmoeTokenizer(self.repo_dir)
        self._status(
            f"Initializing base model on {self.device.type} with {str(self.param_dtype).replace('torch.', '')}..."
        )
        self.base_model = AfmoeForCausalLM(
            self.config,
            device=self.device,
            dtype=self.param_dtype,
        )
        self._status("Loading base model weights into memory...")
        load_checkpoint_into_model(self.base_model, self.repo_dir)
        self.base_model.eval()
        self._status("Base model is ready.")
        self._controller_cache: dict[Path, ControllerCacheEntry] = {}

    def _format_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str | None,
        raw_prompt: bool,
    ) -> str:
        if raw_prompt:
            return prompt
        return self.tokenizer.apply_chat_template(prompt, system_prompt=system_prompt)

    def _load_cached_controller(
        self,
        checkpoint_path: Path,
    ) -> PromptPoolingController:
        resolved_path = checkpoint_path.expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Controller checkpoint not found at {resolved_path}.")
        stat = resolved_path.stat()
        cache_entry = self._controller_cache.get(resolved_path)
        if cache_entry is not None and cache_entry.mtime_ns == stat.st_mtime_ns:
            self._status(f"Reusing cached controller {resolved_path}...")
            return cache_entry.controller

        self._status(f"Loading controller checkpoint {resolved_path}...")
        controller = load_controller_checkpoint(
            resolved_path,
            device=self.device,
            dtype=torch.float32,
        )
        self._controller_cache[resolved_path] = ControllerCacheEntry(
            path=resolved_path,
            mtime_ns=stat.st_mtime_ns,
            controller=controller,
        )
        self._status(f"Controller ready: {resolved_path}.")
        return controller

    @torch.inference_mode()
    def generate(self, prompt: str, *, config: GenerationConfig) -> str:
        self._status(
            "Starting generation"
            + (
                " with controller..."
                if config.controller_checkpoint is not None
                else " with base model..."
            )
        )
        prompt_text = self._format_prompt(
            prompt,
            system_prompt=config.system_prompt,
            raw_prompt=config.raw_prompt,
        )
        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt_text)],
            device=self.device,
            dtype=torch.long,
        )
        eos_token_id = self.tokenizer.token_to_id(self.tokenizer.eos_token)

        if config.controller_checkpoint is not None:
            controller = self._load_cached_controller(config.controller_checkpoint)
            model = TrinityWithController(
                self.base_model,
                controller,
                freeze_base_model=True,
            )
            output_ids = model.generate(
                input_ids,
                controller_input_ids=input_ids,
                controller_strength=config.controller_strength,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                eos_token_id=eos_token_id,
            )
        else:
            output_ids = self.base_model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                eos_token_id=eos_token_id,
            )

        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        if eos_token_id is not None:
            generated_ids = [token for token in generated_ids if token != eos_token_id]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        self._status("Generation finished.")
        return text
