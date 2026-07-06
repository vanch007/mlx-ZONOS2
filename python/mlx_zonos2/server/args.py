from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlx_zonos2.paths import resolve_default_model


DEFAULT_MODEL = resolve_default_model()


def _require_apple_silicon() -> None:
    if sys.platform != "darwin":
        raise SystemExit("mlx-ZONOS2 requires macOS with Apple Silicon (MLX).")
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        raise SystemExit(f"mlx-ZONOS2 requires Apple Silicon; found machine={machine!r}.")


@dataclass(frozen=True)
class ServerArgs:
    server_host: str = "127.0.0.1"
    server_port: int = 1920
    model_path: str = DEFAULT_MODEL
    lazy_load: bool = True
    warm_on_startup: bool = False
    max_tokens_default: int = 1024
    log_level: str = "info"

    @property
    def model_config(self) -> dict[str, Any]:
        return self._model_config

    def __post_init__(self) -> None:
        object.__setattr__(self, "_model_config", {})


def _load_model_config(model_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if path.is_dir():
        config_path = path / "config.json"
        if config_path.is_file():
            return json.loads(config_path.read_text(encoding="utf-8"))
        return {}
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=model_path, filename="config.json")
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_args(argv: list[str] | None = None) -> ServerArgs:
    _require_apple_silicon()
    parser = argparse.ArgumentParser(description="mlx-ZONOS2 FastAPI TTS server")
    parser.add_argument("--host", default=os.getenv("MLX_ZONOS2_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MLX_ZONOS2_PORT", "1920")),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MLX_ZONOS2_MODEL", DEFAULT_MODEL),
        help="HF repo id or local model directory",
    )
    parser.add_argument(
        "--no-lazy",
        action="store_true",
        help="Eagerly materialize weights at load time",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="Run a tiny warmup generate after model load",
    )
    parser.add_argument(
        "--max-tokens-default",
        type=int,
        default=int(os.getenv("MLX_ZONOS2_MAX_TOKENS", "1024")),
    )
    parser.add_argument("--log-level", default=os.getenv("MLX_ZONOS2_LOG_LEVEL", "info"))
    ns = parser.parse_args(argv)
    if ns.max_tokens_default <= 0:
        parser.error("--max-tokens-default must be positive")

    args = ServerArgs(
        server_host=ns.host,
        server_port=ns.port,
        model_path=ns.model,
        lazy_load=not ns.no_lazy,
        warm_on_startup=ns.warm,
        max_tokens_default=ns.max_tokens_default,
        log_level=ns.log_level,
    )
    object.__setattr__(args, "_model_config", _load_model_config(args.model_path))
    return args