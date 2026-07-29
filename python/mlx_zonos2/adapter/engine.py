from __future__ import annotations

import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

from mlx_zonos2.server.args import ServerArgs

logger = logging.getLogger(__name__)

_DAC_REPOSITORY = "mlx-community/descript-audio-codec-44khz"


def resolve_local_dac_path(cache_root: str | os.PathLike[str] | None = None) -> str:
    """Require the ZONOS2 DAC to be fully cached before generation starts.

    ``mlx-audio`` otherwise downloads this dependency lazily on the first
    request, which leaves an API call appearing hung when network use is
    disabled or interrupted.
    """
    from pathlib import Path

    configured = os.environ.get("ZONOS2_DAC_MODEL_PATH", "").strip()
    if configured:
        candidates = [Path(configured).expanduser()]
    else:
        root = Path(
            cache_root or os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")
        ).expanduser()
        repo_cache = root / "models--mlx-community--descript-audio-codec-44khz"
        candidates = []
        ref = repo_cache / "refs" / "main"
        if ref.exists():
            candidates.append(repo_cache / "snapshots" / ref.read_text().strip())
        candidates.extend((repo_cache / "snapshots").glob("*"))

    checked: list[str] = []
    for candidate in candidates:
        if not candidate.exists() or str(candidate) in checked:
            continue
        checked.append(str(candidate))
        if (candidate / "config.json").is_file() and (
            candidate / "model.safetensors"
        ).is_file():
            return str(candidate.resolve())

    location = ", ".join(checked) or "no local DAC snapshot"
    raise Zonos2EngineError(
        f"Missing complete local {_DAC_REPOSITORY} dependency ({location}). "
        "Automatic download is disabled; set ZONOS2_DAC_MODEL_PATH to a complete snapshot."
    )


class Zonos2EngineError(RuntimeError):
    pass


@dataclass
class GenerationMetrics:
    e2e_ms: float
    audio_sec: float
    rtf: float


def apply_fade_out_pcm(
    audio_bytes: bytes, fade_out_ms: float, sample_rate: int = 44100
) -> bytes:
    if fade_out_ms <= 0 or not audio_bytes:
        return audio_bytes
    samples = np.frombuffer(audio_bytes, dtype=np.float32).copy()
    n = min(len(samples), int(sample_rate * fade_out_ms / 1000.0))
    if n <= 0:
        return audio_bytes
    fade = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    samples[-n:] *= fade
    return samples.tobytes()


def audio_array_to_pcm_bytes(audio: Any, sample_rate: int) -> bytes:
    if hasattr(audio, "__array_namespace__"):
        audio_np = np.asarray(mx.array(audio), dtype=np.float32)
    else:
        audio_np = np.asarray(audio, dtype=np.float32)
    audio_np = np.squeeze(audio_np)
    if audio_np.ndim != 1:
        raise Zonos2EngineError(f"Expected mono audio, got shape {audio_np.shape}")
    peak = float(np.max(np.abs(audio_np))) if audio_np.size else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak
    return audio_np.astype(np.float32, copy=False).tobytes()


class Zonos2Engine:
    def __init__(self, config: ServerArgs) -> None:
        self.config = config
        self._model: Any | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-tts")

    @property
    def model(self) -> Any:
        if self._model is None:
            raise Zonos2EngineError("Model is not loaded.")
        return self._model

    @property
    def sample_rate(self) -> int:
        return int(self.config.model_config.get("sample_rate", 44100))

    def load(self) -> None:
        resolve_local_dac_path()
        try:
            from mlx_audio.tts import load as load_tts
        except ImportError as exc:
            raise Zonos2EngineError(
                "mlx-audio is not installed. See README for the add-zonos2 dependency pin."
            ) from exc
        try:
            import mlx_audio.tts.models.zonos2  # noqa: F401
        except ImportError as exc:
            raise Zonos2EngineError(
                "mlx-audio zonos2 module is missing. Pin mlx-audio to "
                "git+https://github.com/lucasnewman/mlx-audio.git@add-zonos2"
            ) from exc

        logger.info("Loading MLX ZONOS2 model from %s", self.config.model_path)
        self._model = load_tts(self.config.model_path, lazy=self.config.lazy_load)
        logger.info("MLX ZONOS2 model loaded")

    def warmup(self) -> None:
        logger.info("Warmup generate (max_tokens=8)")
        self.generate_sync(
            text="Warmup.",
            max_tokens=8,
            text_normalization=False,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def generate_sync(self, **kwargs: Any) -> tuple[bytes, GenerationMetrics]:
        started = time.perf_counter()
        try:
            result = next(self.model.generate(stream=False, **kwargs))
        except StopIteration as exc:
            raise Zonos2EngineError("Model returned no audio.") from exc
        except Exception as exc:
            raise Zonos2EngineError(str(exc)) from exc

        sample_rate = int(getattr(result, "sample_rate", None) or self.sample_rate)
        pcm = audio_array_to_pcm_bytes(result.audio, sample_rate)
        elapsed = time.perf_counter() - started
        audio_sec = len(pcm) / (sample_rate * 4.0)
        rtf = elapsed / audio_sec if audio_sec > 0 else 0.0
        metrics = GenerationMetrics(
            e2e_ms=elapsed * 1000.0,
            audio_sec=audio_sec,
            rtf=rtf,
        )
        logger.info(
            "TTS completed: E2E=%.1fms audio=%.2fs RTF=%.2fx",
            metrics.e2e_ms,
            metrics.audio_sec,
            metrics.rtf,
        )
        return pcm, metrics

    async def generate(self, **kwargs: Any) -> tuple[bytes, GenerationMetrics]:
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self.generate_sync(**kwargs))
