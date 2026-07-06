from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MLX_MODEL_DIR = REPO_ROOT / "models" / "Zyphra-ZONOS2-mlx"
DEFAULT_PYTORCH_SOURCE = Path(
    os.getenv("ZONOS2_PYTORCH_MODEL", "/Users/vanch/ZONOS2/models/ZONOS2")
)


def resolve_default_model() -> str:
    override = os.getenv("MLX_ZONOS2_MODEL")
    if override:
        return override
    return str(DEFAULT_MLX_MODEL_DIR)


def mlx_model_ready(path: str | Path | None = None) -> bool:
    model_dir = Path(path or resolve_default_model())
    return model_dir.is_dir() and (model_dir / "config.json").is_file()