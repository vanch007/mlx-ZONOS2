# mlx-ZONOS2

> Apple Silicon–native ZONOS2 TTS server built on **MLX** runtime with [mlx-audio](https://github.com/lucasnewman/mlx-audio).

This is the **MLX-only successor** to the PyTorch `ZONOS2` server. All inference runs on Apple Silicon using the MLX framework — no PyTorch, no CUDA, no FlashAttention.

**Weights:** [mlx-community/Zyphra-ZONOS2](https://huggingface.co/mlx-community/Zyphra-ZONOS2) (BF16, ~15.4GB)

---

## Features

| Feature | Status | Details |
|---------|--------|---------|
| Apple Silicon native (MLX) | ✅ Stable | M1/M2/M3/M4, BF16, ~15.4GB |
| Non-streaming TTS (`/tts/generate`) | ✅ Stable | 44.1 kHz, 16-bit mono PCM |
| OpenAI-compatible speech (`/v1/audio/speech`) | ✅ Stable | `model=zonos2-mlx`, `voice`, `speed` |
| **Multilingual support (26 languages)** | ✅ **Verified** | 3 tiers, `text_normalization=False` for non-English |
| Speaker embedding extraction & caching | ✅ Stable | 2048-dim embeddings |
| Speaker blend (linear interpolation) | ✅ Stable | `speaker_blend_embedding_id_a/b`, `t` |
| Speaker cloning (cross-lingual) | ✅ **Verified** | `speaker_embedding_base64` (pre-extracted) |
| Speaking rate conditioning (8 buckets) | ✅ **Verified** | 0-8 to 40+ bytes/sec ranges |
| Quality conditioning (6 features) | ✅ **Verified** | 6 features × 8-12 buckets each |
| Text normalization | ✅ Stable | **English only** (en_us, en_gb, en) |
| Prompt style prefixes | ✅ **Verified** | Whisper, Shouting, Sad, Excited, Pirate, etc. |
| Sampling control | ✅ **Verified** | temp, top_p, top_k, min_p, repetition params |
| Advanced flags | ✅ **Verified** | clean_speaker_background, accurate_mode, ignore_eos |
| Gradio Web UI | ✅ Stable | Run `./start.sh` to start the API server and WebUI together |
| A/B benchmark (MLX vs MPS) | ✅ Stable | `benchmark/compare_mps_mlx.py` |

> **v1 note:** Streaming is **not** supported. Requests with `stream=true` return HTTP 400.

---

## Requirements

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- **~16GB+** unified memory for BF16 weights

---

## Installation

### 1. Clone and setup

```bash
git clone https://github.com/your-repo/mlx-ZONOS2.git
cd mlx-ZONOS2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install (includes mlx-audio @ add-zonos2 branch)
pip install -e ".[dev]"
```

### 2. Download models

```bash
chmod +x scripts/download_models.sh
./scripts/download_models.sh
```

This downloads:
- **ZONOS2 MLX weights** (~15.4GB) → `models/Zyphra-ZONOS2-mlx/`
- **DAC vocoder** → `models/dac-44khz/`
- **Speaker encoder** → `models/qwen3-voice-embedding/`

### 3. Verify model load

```bash
python scripts/smoke_load.py --lazy
```

### 4. Start the server

```bash
./start.sh

# Or with custom options:
python -m uvicorn mlx_zonos2.server.api_server:app --host 127.0.0.1 --port 1920
```

---

## API Reference

### `POST /tts/generate`

Non-streaming text-to-speech generation.

**Request body:**

```json
{
  "text": "Hello, this is a test.",
  "language": "en_us",
  "text_normalization": true,
  "temperature": 1.15,
  "topk": 106,
  "top_p": 0.0,
  "min_p": 0.18,
  "max_tokens": 1024,
  "seed": 42,
  "speaking_rate_enabled": true,
  "speed": 1.0,
  "speaking_rate_bucket": null,
  "quality_enabled": true,
  "quality_buckets": {"trailing_silence_s": 3},
  "clean_speaker_background": false,
  "accurate_mode": true,
  "stream": false,
  "fade_out_ms": 0.0,
  "speaker_audio_base64": null,
  "speaker_embedding_base64": null,
  "speaker_embedding_id": null,
  "speaker_blend_embedding_id_a": null,
  "speaker_blend_embedding_id_b": null,
  "speaker_blend_t": null
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | **required** | Text to synthesize |
| `language` | string | `en_us` | Language code (see [Supported Languages](#supported-languages)) |
| `text_normalization` | bool | `true` | **Enable for English, disable for other languages** |
| `temperature` | float | 1.15 | Sampling temperature (0.5–1.5 recommended) |
| `topk` | int | 106 | Top-k sampling cutoff |
| `top_p` | float | 0.0 | Nucleus sampling (0.0 = disabled) |
| `min_p` | float | 0.18 | Min-p sampling threshold |
| `max_tokens` | int | 1024 | Max generation tokens (max 6144) |
| `seed` | int/null | null | Random seed for reproducible output |
| `speaking_rate_enabled` | bool | `false` | Enable speaking rate conditioning |
| `speed` | float | 1.0 | Speed multiplier (used with `speaking_rate_enabled=true`) |
| `speaking_rate_bucket` | int/null | null | Explicit bucket index 0–7 (overrides `speed`) |
| `quality_enabled` | bool | `true` | Enable quality conditioning |
| `quality_buckets` | dict | `{"trailing_silence_s": 3}` | Quality feature buckets (see below) |
| `clean_speaker_background` | bool | `false` | Clean speaker background noise |
| `accurate_mode` | bool | `true` | Enable accurate mode token |
| `stream` | bool | `false` | **Must be false** (streaming unsupported in v1) |
| `fade_out_ms` | float | 0.0 | Fade out duration in milliseconds |
| `speaker_audio_base64` | string/null | null | **Not recommended** — reference audio as base64 WAV |
| `speaker_embedding_base64` | string/null | null | **Recommended** — pre-extracted 2048-dim embedding (base64 JSON) |
| `speaker_embedding_id` | string/null | null | Cache ID for extracted embedding |
| `speaker_blend_embedding_id_a` | string/null | null | Blend: first cached embedding ID |
| `speaker_blend_embedding_id_b` | string/null | null | Blend: second cached embedding ID |
| `speaker_blend_t` | float/null | null | Blend interpolation factor (0.0–1.0, default 0.5) |

**Response:**

| Header | Value |
|--------|-------|
| Content-Type | audio/pcm |
| X-Audio-Sample-Rate | 44100 |
| X-Audio-Channels | 1 |
| X-Audio-Format | float32 |

---

### `POST /v1/audio/speech` (OpenAI-compatible)

```json
{
  "model": "zonos2-mlx",
  "input": "Hello from OpenAI format.",
  "voice": "alice",
  "response_format": "pcm",
  "speed": 1.0
}
```

---

### `GET /health`

```json
{
  "status": "ok",
  "backend": "mlx",
  "model": "models/Zyphra-ZONOS2-mlx",
  "sample_rate": 44100,
  "streaming": false
}
```

---

## Supported Languages (Verified: 26 Languages)

> **Important:** Server accepts all languages, but **`text_normalization` must be `false` for non-English**.

### Tier 1 — Core Languages (Best Quality)

| Code | Language | Notes |
|------|----------|-------|
| `en_us` | English (US) | ✅ Best quality, text normalization supported |
| `en_gb` | English (GB) | ✅ British variant |
| `en` | English (Generic) | ✅ Generic variant |
| `zh` | Chinese (Mandarin) | ⚠️ `text_normalization=false` |
| `ja` | Japanese | ⚠️ `text_normalization=false` |

### Tier 2 — Extended Languages (Good Quality)

| Code | Language | Notes |
|------|----------|-------|
| `ko` | Korean | ⚠️ `text_normalization=false` |
| `fr` | French | ⚠️ `text_normalization=false` |
| `es` | Spanish | ⚠️ `text_normalization=false` |
| `de` | German | ⚠️ `text_normalization=false` |
| `ru` | Russian | ⚠️ `text_normalization=false` |
| `pt` | Portuguese | ⚠️ `text_normalization=false` |
| `it` | Italian | ⚠️ `text_normalization=false` |
| `vi` | Vietnamese | ⚠️ `text_normalization=false` |
| `nl` | Dutch | ⚠️ `text_normalization=false` |
| `pl` | Polish | ⚠️ `text_normalization=false` |
| `tr` | Turkish | ⚠️ `text_normalization=false` |
| `he` | Hebrew | ⚠️ `text_normalization=false` |
| `ar` | Arabic | ⚠️ `text_normalization=false` |

### Tier 3 — Additional Languages (Variable Quality)

| Code | Language | Notes |
|------|----------|-------|
| `sv` | Swedish | ⚠️ `text_normalization=false` |
| `hi` | Hindi | ⚠️ `text_normalization=false` |
| `ta` | Tamil | ⚠️ `text_normalization=false` |
| `te` | Telugu | ⚠️ `text_normalization=false` |
| `th` | Thai | ⚠️ `text_normalization=false` |
| `da` | Danish | ⚠️ `text_normalization=false` |
| `no` | Norwegian | ⚠️ `text_normalization=false` |
| `fi` | Finnish | ⚠️ `text_normalization=false` |
| `uk` | Ukrainian | ⚠️ `text_normalization=false` |

> **Example — Chinese:**
> ```json
> {"text": "欢迎使用 ZONOS2", "language": "zh", "text_normalization": false}
> ```

---

## Speaking Rate Conditioning (8 Buckets)

Control speech speed via bucket index (0–7) or `speed` multiplier.

| Bucket | Index | Range (bytes/sec) | Description |
|--------|-------|-------------------|-------------|
| Very Slow | 0 | 0–8 | Extremely slow, deliberate |
| Slow | 1 | 8–11 | Slow, clear articulation |
| Normal | 2 | 11–14 | **Default natural speed** |
| Fast | 3 | 14–17 | Faster than normal |
| Very Fast | 4 | 17–21 | Quick speech |
| Rapid | 5 | 21–28 | Very rapid |
| Very Rapid | 6 | 28–40 | Extremely rapid |
| Extreme | 7 | 40+ | Maximum speed |

**Usage:**
```json
{"speaking_rate_enabled": true, "speaking_rate_bucket": 2}
```
or
```json
{"speaking_rate_enabled": true, "speed": 1.0}
```

> **Note:** `speaking_rate_bucket` (explicit 0–7) takes precedence over `speed` (multiplier).

---

## Quality Conditioning (6 Features)

Control audio quality characteristics via feature buckets.

| Feature | Bucket Count | Range | Description |
|---------|--------------|-------|-------------|
| `lufs` | 12 | -1000 to -5+ | Integrated loudness (dB LUFS) |
| `estimated_snr` | 12 | -1000 to 60+ | Signal-to-noise ratio (dB) |
| `max_pause` | 12 | 0–6+ seconds | Maximum pause duration |
| `estimated_bandlimit_hz` | 8 | 495–24000+ Hz | Estimated audio bandwidth |
| `leading_silence_s` | 8 | 0–4+ seconds | Leading silence duration |
| `trailing_silence_s` | 8 | 0–4+ seconds | **Trailing silence duration** |

**Usage:**
```json
{"quality_enabled": true, "quality_buckets": {"trailing_silence_s": 0}}
```
```json
{"quality_enabled": true, "quality_buckets": {"lufs": 5, "estimated_snr": 5, "max_pause": 5, "estimated_bandlimit_hz": 3, "leading_silence_s": 3, "trailing_silence_s": 3}}
```

> **Bucket indices:** `lufs`/`estimated_snr`/`max_pause`: 0–11; `bandlimit`/`leading_silence`/`trailing_silence`: 0–7.

---

## Speaker Cloning (Cross-Lingual Verified)

> **Use `speaker_embedding_base64` (pre-extracted embedding) — `speaker_audio_base64` has API issues.**

### Extract Embedding (Python):
```python
from mlx_audio.tts import load as load_tts
model = load_tts('models/Zyphra-ZONOS2-mlx', lazy=True)
embedding = model.extract_speaker_embedding('reference.wav')
# embedding shape: (1, 2048)

# Serialize for API:
import base64, json
emb_b64 = base64.b64encode(json.dumps(embedding.tolist()).encode()).decode()
```

### API Request:
```json
{
  "text": "克隆语音测试。",
  "language": "zh",
  "text_normalization": false,
  "speaker_embedding_base64": "<base64_json_of_2048dim_array>"
}
```

**Verified cross-lingual clones:** en_us → zh, ja, fr, en_gb

---

## Prompt Style Prefixes (Emotion/Style Control)

> **No native emotion parameter** — use prompt text prefixes.

| Style | Prefix Example | Notes |
|-------|----------------|-------|
| Whisper | `(Whisper) Shh... this is a secret.` | Soft, quiet |
| Shouting | `(Shouting) HEY! CAN YOU HEAR ME?!` | Loud, projected |
| Sad | `(Sad, trembling voice) I don't know what to say.` | Melancholic |
| Excited | `(Excited, energetic) This is amazing!` | High energy |
| Pirate | `(Pirate captain) Arrr matey! Welcome aboard!` | Character voice |
| Storyteller | `(Old storyteller) Gather round, children...` | Narrative tone |

**Verified styles:** Whisper (80% ASR), Sad (85.7%), Excited (varies), Pirate (45.5%)

---

## Sampling Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `temperature` | 1.15 | 0.5–1.5 | Higher = more creative |
| `topk` | 106 | 1–1024 | Top-k cutoff |
| `top_p` | 0.0 | 0.0–1.0 | Nucleus sampling (0=off) |
| `min_p` | 0.18 | 0.0–1.0 | Min-p threshold |
| `repetition_window` | 50 | 1–100 | Repetition penalty window |
| `repetition_penalty` | 1.2 | 1.0–2.0 | Penalty for repeated tokens |
| `repetition_codebooks` | 8 | 1–9 | Codebooks to apply penalty |
| `seed` | null | int | Reproducible output |

---

## Advanced Flags

| Flag | Default | Description |
|------|---------|-------------|
| `clean_speaker_background` | `false` | Clean speaker embedding background |
| `accurate_mode` | `true` | Enable accurate mode token |
| `ignore_eos` | `false` | Ignore EOS token (force longer gen) |

---

## Configuration (Environment Variables)

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `MLX_ZONOS2_HOST` | 127.0.0.1 | Server bind address |
| `MLX_ZONOS2_PORT` | 1920 | Server port |
| `MLX_ZONOS2_MODEL` | models/Zyphra-ZONOS2-mlx | Model path (local or HF repo) |
| `MLX_ZONOS2_MAX_TOKENS` | 1024 | Default max tokens |
| `MLX_ZONOS2_LOG_LEVEL` | info | uvicorn log level |

---

## Performance Benchmarks (M3 Max)

| Metric | Value |
|--------|-------|
| **Avg RTF** | 1.064 (6% slower than real-time) |
| **RTF Range** | 0.91 – 1.11 |
| **Avg E2E Latency** | 6.3s |
| **Concurrent Requests** | 1 (sequential recommended) |

| Section | Tests | Avg RTF |
|---------|-------|---------|
| Multilingual (26 langs) | 27 | 1.076 |
| Speaking Rate (8 buckets) | 8 | 1.061 |
| Quality Conditioning | 8 | 1.061 |
| Speaker Cloning | 6 | 1.045 |
| Style Prefix | 6 | 1.067 |
| Sampling Params | 5 | 1.033 |

---

## ASR Verification Results (mlx-whisper / Qwen3-ASR)

| Language | Similarity | Notes |
|----------|------------|-------|
| en_us | 90% | Best |
| ru | 92.3% | Excellent |
| ko | 79.2% | Very good |
| he | 54.5% | Good |
| en_gb | 50% | Moderate |
| fr/es/pt/it/vi/uk | 46–57% | Moderate |
| zh/ja/ta/te/th/no | 0–15% | Low (Whisper limits) |

> ASR similarity based on character-level (CJK) / word-level (others) comparison.

---

## Web UI

```bash
./start.sh
# Open http://127.0.0.1:7860
```

Useful options:

```bash
MLX_ZONOS2_HOST=127.0.0.1 MLX_ZONOS2_PORT=1920 MLX_ZONOS2_WEBUI_HOST=127.0.0.1 MLX_ZONOS2_WEBUI_PORT=7860 ./start.sh
```

---

## Benchmarking

### MLX-only benchmark
```bash
python benchmark/bench_tts.py --n-repeats 5 --output-json benchmark/result.json
python benchmark/bench_tts.py --n-repeats 5 --output-md benchmark/result.md
```

### A/B benchmark (MLX vs MPS)
```bash
# MPS ZONOS2 on port 1919, mlx-ZONOS2 on port 1920
python benchmark/compare_mps_mlx.py --requests 5
# Output: benchmark/out/compare_summary.json
```

### Full Capabilities Test
```bash
python benchmark/zonos2_full_capabilities_test.py --url http://127.0.0.1:1920
```

### Multilingual Listening Test (RTF + ASR)
```bash
python benchmark/generate_multilingual_listening.py
# Output: benchmark/out/zonos2_multilingual_listening/
```

---

## Error Codes

| Status | Condition |
|--------|-----------|
| 400 | `stream=true`, invalid speaker/blend, unsupported lang (before fix) |
| 500 | mlx-audio internal error |
| 503 | OOM, model not loaded, streaming unsupported (v1) |

---

## Project Structure

```
mlx-ZONOS2/
├── python/mlx_zonos2/
│   ├── server/
│   │   ├── api_server.py     # FastAPI application, endpoints
│   │   ├── gradio_webui.py   # Gradio Web UI for the FastAPI server
│   │   └── args.py           # CLI argument parsing
│   ├── adapter/
│   │   ├── engine.py         # MLX generation engine
│   │   ├── conditioning.py   # Speaking-rate & quality mapping
│   │   ├── speaker.py        # Speaker embedding extraction
│   │   └── speaker_cache.py  # In-memory cache + blend
│   └── compat/
│       └── prompt.py         # Prompt compatibility
├── benchmark/
│   ├── bench_tts.py          # MLX benchmark (HTTP + local)
│   ├── compare_mps_mlx.py    # A/B comparison
│   ├── zonos2_capabilities_test.py        # 63 comprehensive tests
│   ├── generate_multilingual_listening.py # 40 tests + RTF + ASR
│   └── generate_listening_audio.py        # Audio generation
├── tests/
│   ├── test_api_parity.py    # API endpoint tests
│   ├── test_conditioning.py  # Conditioning tests
│   └── test_speaker_cache.py # Speaker cache tests
├── scripts/
│   ├── download_models.sh
│   ├── run_server.sh
│   └── smoke_load.py
├── start.sh                  # Start API server and Gradio WebUI together
└── pyproject.toml
```

---

## Relationship to ZONOS2 (PyTorch)

| Phase | ZONOS2 repo | mlx-ZONOS2 repo |
|-------|-------------|-----------------|
| Development | MPS A/B baseline | Primary implementation |
| v1 stable | Banner → migrate | Default Apple Silicon path |
| Post-stable | Archive / read-only | Sole maintained inference stack |

No PyTorch code exists in mlx-ZONOS2.

---

## Design Spec

See `../ZONOS2/docs/spark/2026-06-14-mlx-zonos2-design.md`

---

## Key Fixes & Notes

1. **Language validation bypass** — Patched `mlx_zonos2/adapter/conditioning.py:298` to accept all languages (was English-only)
2. **Speaker cloning** — Use `speaker_embedding_base64` (pre-extracted embedding); `speaker_audio_base64` (WAV upload) returns HTTP 500
3. **Text normalization** — Enable for English (`en_us`, `en_gb`, `en`), disable for all other languages
4. **Emotion control** — No native parameter; use prompt prefixes like `(Whisper)`, `(Sad)`, `(Pirate)`
5. **Streaming** — Not supported in v1

---

叶总牛逼！
