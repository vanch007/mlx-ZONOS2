from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal


def _sanitize_no_proxy_for_httpx() -> None:
    """httpx rejects bare IPv6 entries like ::1 in NO_PROXY."""
    for key in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(key)
        if not value:
            continue
        entries = [entry.strip() for entry in value.split(",")]
        entries = [entry for entry in entries if entry not in {"::1", "::1/128"}]
        os.environ[key] = ",".join(entries)


_sanitize_no_proxy_for_httpx()

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from mlx_zonos2.adapter.conditioning import (
    normalize_tts_request_language,
    resolve_quality_buckets,
    resolve_speaking_rate_bucket,
    resolve_tts_max_tokens,
)
from mlx_zonos2.adapter.engine import Zonos2Engine, Zonos2EngineError, apply_fade_out_pcm
from mlx_zonos2.adapter.speaker import (
    decode_base64_blob,
    load_embedding_from_bytes,
    model_speaker_dim,
    resolve_ref_audio_path,
    resolve_speaker_embedding_array,
)
from mlx_zonos2.adapter.speaker_cache import SpeakerCache
from mlx_zonos2.server.args import ServerArgs, parse_args

logger = logging.getLogger(__name__)

_ENGINE: Zonos2Engine | None = None
_SERVER_CONFIG: ServerArgs | None = None
_SPEAKER_CACHE: SpeakerCache | None = None
_DEFAULT_QUALITY_BUCKETS = {"trailing_silence_s": 3}


def configure_server(args: ServerArgs) -> None:
    global _SERVER_CONFIG
    _SERVER_CONFIG = args


def get_server_config() -> ServerArgs:
    if _SERVER_CONFIG is not None:
        return _SERVER_CONFIG
    return parse_args([])


class TTSGenerateRequest(BaseModel):
    text: str
    language: str = "en_us"
    text_normalization: bool = True
    temperature: float = 1.15
    topk: int = 106
    top_p: float = 0.0
    min_p: float = 0.18
    max_tokens: int | None = None
    fade_out_ms: float = 0.0
    repetition_window: int = 50
    repetition_penalty: float = 1.2
    repetition_codebooks: int = 8
    seed: int | None = None
    speaking_rate_enabled: bool = False
    speed: float | None = None
    speaking_rate: float | None = None
    speaking_rate_bucket: int | None = None
    quality_enabled: bool = True
    quality_buckets: Dict[str, int | None] | List[int | None] | None = Field(
        default_factory=lambda: dict(_DEFAULT_QUALITY_BUCKETS)
    )
    quality_values: Dict[str, float | None] | List[float | None] | None = None
    clean_speaker_background: bool = False
    accurate_mode: bool = True
    stream: bool = False
    speaker_audio_base64: str | None = None
    speaker_audio_name: str | None = None
    speaker_embedding_base64: str | None = None
    speaker_embedding_name: str | None = None
    speaker_embedding_id: str | None = None
    speaker_blend_embedding_id_a: str | None = None
    speaker_blend_embedding_id_b: str | None = None
    speaker_blend_t: float | None = None
    speaker_wav_base64: str | None = None


def get_engine() -> Zonos2Engine:
    if _ENGINE is None:
        raise HTTPException(status_code=503, detail="TTS engine is not initialized.")
    return _ENGINE


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _ENGINE, _SERVER_CONFIG, _SPEAKER_CACHE
    args = get_server_config()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    engine = Zonos2Engine(args)
    try:
        engine.load()
        if args.warm_on_startup:
            engine.warmup()
    except Zonos2EngineError as exc:
        logger.error("%s", exc)
        raise SystemExit(str(exc)) from exc
    _ENGINE = engine
    _SPEAKER_CACHE = SpeakerCache(ttl=3600.0)  # 1 hour default TTL
    yield
    engine.shutdown()
    _ENGINE = None
    _SPEAKER_CACHE = None


app = FastAPI(title="mlx-ZONOS2 API Server", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    engine = get_engine()
    return {
        "status": "ok",
        "backend": "mlx",
        "model": engine.config.model_path,
        "sample_rate": engine.sample_rate,
        "streaming": False,
    }


def _resolve_speaker_embedding(
    req: TTSGenerateRequest,
) -> tuple[Any | None, str | None]:
    """Resolve speaker embedding from cache, blend, or direct extraction.

    Returns:
        (embedding, cache_id): embedding to pass to generate(), or cache_id for caching.
        embedding can be None if no speaker conditioning provided.
        cache_id is set when speaker_embedding_id is used (for caching after extraction).
    """
    cache = _SPEAKER_CACHE
    engine = get_engine()
    model_config = engine.config.model_config

    # Case 1: Blend two cached embeddings
    if req.speaker_blend_embedding_id_a and req.speaker_blend_embedding_id_b:
        if not cache:
            raise ValueError("Speaker cache not initialized.")
        t = req.speaker_blend_t if req.speaker_blend_t is not None else 0.5
        blended = cache.blend(
            req.speaker_blend_embedding_id_a,
            req.speaker_blend_embedding_id_b,
            float(t),
        )
        if blended is None:
            raise ValueError(
                f"Blend failed: one or both IDs not found "
                f"(a={req.speaker_blend_embedding_id_a}, b={req.speaker_blend_embedding_id_b})."
            )
        return blended, req.speaker_blend_embedding_id_a  # use 'a' as cache reference

    # Case 2: Use cached embedding by ID
    if req.speaker_embedding_id and cache:
        cached = cache.get(req.speaker_embedding_id)
        if cached is not None:
            return cached, req.speaker_embedding_id

    # Case 3: Base64 embedding provided directly
    if req.speaker_embedding_base64:
        blob = decode_base64_blob(req.speaker_embedding_base64, "speaker_embedding_base64")
        embedding = load_embedding_from_bytes(
            blob,
            expected_dim=model_speaker_dim(model_config),
            field_name="speaker_embedding_base64",
        )
        return embedding, req.speaker_embedding_id

    # Case 4: Extract from audio (raw audio or ref_audio)
    ref_path = resolve_ref_audio_path(
        speaker_audio_base64=req.speaker_audio_base64,
        legacy_speaker_wav_base64=req.speaker_wav_base64 or req.speaker_wav_base64,
    )
    if ref_path is not None:
        try:
            embedding = engine.model.extract_speaker_embedding(ref_path)
            # Cache the extracted embedding if an ID was provided
            if req.speaker_embedding_id and cache:
                cache.set(req.speaker_embedding_id, embedding)
            return embedding, req.speaker_embedding_id
        finally:
            try:
                Path(ref_path).unlink(missing_ok=True)
            except OSError:
                pass
    elif any((req.speaker_audio_base64, req.speaker_wav_base64)):
        # Audio provided but could not be resolved to path — try as raw WAV
        audio_blob = req.speaker_audio_base64 or req.speaker_wav_base64
        if audio_blob:
            blob = decode_base64_blob(audio_blob, "speaker_audio_base64")
            if blob[:4] == b"RIFF":
                # Write to temp .wav
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.write(blob)
                tmp.close()
                try:
                    embedding = engine.model.extract_speaker_embedding(tmp.name)
                    if req.speaker_embedding_id and cache:
                        cache.set(req.speaker_embedding_id, embedding)
                    return embedding, req.speaker_embedding_id
                finally:
                    try:
                        Path(tmp.name).unlink(missing_ok=True)
                    except OSError:
                        pass
            else:
                Path(blob).unlink(missing_ok=True)

    return None, None


@app.post("/tts/generate")
async def tts_generate(req: TTSGenerateRequest) -> Response:
    if req.stream:
        raise HTTPException(
            status_code=400,
            detail="Streaming is unsupported in mlx-ZONOS2 v1. Use stream=false.",
        )

    engine = get_engine()
    model_config = engine.config.model_config

    try:
        language = normalize_tts_request_language(req.language)
        speaking_rate_bucket = resolve_speaking_rate_bucket(
            model_config,
            speaking_rate_bucket=req.speaking_rate_bucket,
            speaking_rate=req.speaking_rate,
            speed=req.speed,
            speaking_rate_enabled=req.speaking_rate_enabled,
        )
        quality_buckets = resolve_quality_buckets(
            model_config,
            quality_buckets=req.quality_buckets,
            quality_values=req.quality_values,
            quality_enabled=req.quality_enabled
            and int(model_config.get("quality_num_buckets", 0) or 0) > 0,
        )
        max_tokens = resolve_tts_max_tokens(
            model_config,
            engine.config.max_tokens_default,
            req.max_tokens,
        )
        speaker_embedding, _ = _resolve_speaker_embedding(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    generate_kwargs: dict[str, Any] = {
        "text": req.text,
        "lang_code": language,
        "text_normalization": req.text_normalization,
        "temperature": req.temperature,
        "top_k": req.topk,
        "top_p": req.top_p,
        "min_p": req.min_p,
        "max_tokens": max_tokens,
        "repetition_window": req.repetition_window,
        "repetition_penalty": req.repetition_penalty,
        "repetition_codebooks": req.repetition_codebooks,
        "seed": req.seed,
        "speaking_rate_bucket": speaking_rate_bucket,
        "quality_buckets": quality_buckets,
        "clean_speaker_background": req.clean_speaker_background,
        "accurate_mode": req.accurate_mode,
    }
    if speaker_embedding is not None:
        generate_kwargs["speaker_embedding"] = speaker_embedding

    try:
        pcm, metrics = await engine.generate(**generate_kwargs)
    except Zonos2EngineError as exc:
        message = str(exc)
        status = 503 if "memory" in message.lower() or "oom" in message.lower() else 500
        raise HTTPException(status_code=status, detail=message) from exc

    pcm = apply_fade_out_pcm(pcm, req.fade_out_ms, sample_rate=engine.sample_rate)
    return Response(
        content=pcm,
        media_type="audio/pcm",
        headers={
            "X-Audio-Sample-Rate": str(engine.sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Format": "float32",
            "X-Generation-E2E-Ms": f"{metrics.e2e_ms:.3f}",
            "X-Generation-Audio-Seconds": f"{metrics.audio_sec:.6f}",
            "X-Generation-RTF": f"{metrics.rtf:.6f}",
        },
    )


class OpenAISpeechRequest(BaseModel):
    model: str = "zonos2-mlx"
    input: str = ""
    voice: str | None = None
    response_format: Literal["pcm", "mp3", "wav", "flac"] = "pcm"
    speed: float = 1.0


@app.post("/v1/audio/speech")
async def openai_speech(req: OpenAISpeechRequest) -> Response:
    """OpenAI-compatible speech endpoint (non-streaming, v1)."""
    if req.speed <= 0:
        raise HTTPException(status_code=400, detail="Speed must be positive.")

    engine = get_engine()

    # Map speed to speaking_rate
    speaking_rate_bucket = None
    if abs(req.speed - 1.0) > 0.01:
        speaking_rate_bucket = resolve_speaking_rate_bucket(
            engine.config.model_config,
            speed=req.speed,
            speaking_rate_enabled=True,
        )

    try:
        pcm, metrics = await engine.generate(
            text=req.input,
            lang_code="en_us",
            text_normalization=True,
            max_tokens=resolve_tts_max_tokens(
                engine.config.model_config,
                engine.config.max_tokens_default,
                None,
            ),
            speaking_rate_bucket=speaking_rate_bucket,
            seed=None,
        )
    except Zonos2EngineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pcm = apply_fade_out_pcm(pcm, 0.0, sample_rate=engine.sample_rate)

    media_type = {
        "pcm": "audio/pcm",
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac",
    }.get(req.response_format, "audio/pcm")

    return Response(
        content=pcm,
        media_type=media_type,
        headers={
            "X-Audio-Sample-Rate": str(engine.sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Format": req.response_format,
            "X-Generation-E2E-Ms": f"{metrics.e2e_ms:.3f}",
            "X-Generation-Audio-Seconds": f"{metrics.audio_sec:.6f}",
            "X-Generation-RTF": f"{metrics.rtf:.6f}",
        },
    )


def main() -> None:
    args = parse_args()
    configure_server(args)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    uvicorn.run(
        app,
        host=args.server_host,
        port=args.server_port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
