#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mlx_zonos2.paths import mlx_model_ready, resolve_default_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test mlx-audio ZONOS2 load().")
    parser.add_argument(
        "--model",
        default=resolve_default_model(),
        help="Local converted model directory (default: models/Zyphra-ZONOS2-mlx)",
    )
    parser.add_argument("--lazy", action="store_true", default=True)
    parser.add_argument("--no-lazy", action="store_false", dest="lazy")
    args = parser.parse_args()

    try:
        import mlx_audio.tts.models.zonos2  # noqa: F401
        from mlx_audio.tts import load
    except ImportError as exc:
        print(
            "FAIL: mlx-audio zonos2 module missing. Pin dependency to "
            "git+https://github.com/lucasnewman/mlx-audio.git@add-zonos2",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        return 1

    if not mlx_model_ready(args.model):
        print(
            f"FAIL: MLX model not found at {args.model}\n"
            "Run ./scripts/convert_local_model.sh first (uses local PyTorch weights).",
            file=sys.stderr,
        )
        return 1

    print(f"Loading {args.model} (lazy={args.lazy}) ...")
    model = load(args.model, lazy=args.lazy)
    print(f"OK: loaded model_type={getattr(model, 'model_type', 'unknown')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())