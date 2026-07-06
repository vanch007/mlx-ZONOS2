from __future__ import annotations

import base64
import io
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np


def model_speaker_dim(model_config: dict[str, Any]) -> int:
    return int(model_config.get("speaker_embedding_dim", 2048))


def model_supports_speaker(model_config: dict[str, Any]) -> bool:
    return bool(model_config.get("speaker_enabled", False))


def decode_base64_blob(value: str, field_name: str) -> bytes:
    try:
        return base64.b64decode(value, validate=False)
    except Exception as exc:
        raise ValueError(f"Invalid base64 in {field_name}.") from exc


def decode_wav_bytes(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if sample_width != 2:
        raise ValueError("Reference audio must be 16-bit PCM WAV.")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, sample_rate


def load_embedding_from_bytes(
    blob: bytes,
    *,
    expected_dim: int,
    field_name: str,
) -> np.ndarray:
    if blob.startswith(b"\x93NUMPY"):
        embedding = np.load(io.BytesIO(blob), allow_pickle=False)
    else:
        try:
            import json

            payload = json.loads(blob.decode("utf-8"))
            embedding = np.asarray(payload, dtype=np.float32)
        except Exception:
            embedding = np.frombuffer(blob, dtype=np.float32)
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if embedding.shape[0] != expected_dim:
        raise ValueError(
            f"{field_name} has dim {embedding.shape[0]}, expected {expected_dim}."
        )
    return embedding


def resolve_ref_audio_path(
    *,
    speaker_audio_base64: str | None,
    legacy_speaker_wav_base64: str | None,
) -> str | None:
    payload = speaker_audio_base64 or legacy_speaker_wav_base64
    if not payload:
        return None
    audio_bytes = decode_base64_blob(payload, "speaker_audio_base64")
    suffix = ".wav"
    if audio_bytes[:4] == b"RIFF":
        suffix = ".wav"
    elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] in {b"\xff\xfb", b"\xff\xf3"}:
        suffix = ".mp3"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(audio_bytes)
    temp.flush()
    temp.close()
    return temp.name


def resolve_speaker_embedding_array(
    model: Any,
    model_config: dict[str, Any],
    *,
    speaker_embedding_base64: str | None,
    speaker_audio_base64: str | None,
    legacy_speaker_wav_base64: str | None,
) -> Any | None:
    if not model_supports_speaker(model_config):
        if any((speaker_embedding_base64, speaker_audio_base64, legacy_speaker_wav_base64)):
            raise ValueError("Current model does not support speaker embeddings.")
        return None

    expected_dim = model_speaker_dim(model_config)
    if speaker_embedding_base64:
        blob = decode_base64_blob(speaker_embedding_base64, "speaker_embedding_base64")
        embedding = load_embedding_from_bytes(
            blob,
            expected_dim=expected_dim,
            field_name="speaker_embedding_base64",
        )
        return embedding

    ref_path = resolve_ref_audio_path(
        speaker_audio_base64=speaker_audio_base64,
        legacy_speaker_wav_base64=legacy_speaker_wav_base64,
    )
    if ref_path is None:
        return None
    try:
        return model.extract_speaker_embedding(ref_path)
    finally:
        try:
            Path(ref_path).unlink(missing_ok=True)
        except OSError:
            pass