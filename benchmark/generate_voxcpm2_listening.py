#!/usr/bin/env python3
"""
Generate ZONOS2 BF16 audio for voxcpm2 tests and build HTML listening report.

Uses voxcpm2's 66 official demo texts and reference audio as test cases,
generates audio with ZONOS2 BF16 where capable, and produces an HTML
listening comparison report similar to voxcpm2's format.

ZONOS2-only tests (Speaking Rate, Quality, Style Prefix, Speaker Cloning)
are also included in the report.

Key constraints:
- ZONOS2 only has BF16 (no INT8)
- ZONOS2 does not support Chinese dialects
- ZONOS2 style prefix: Whisper, Shouting, Sad, Excited, Pirate
"""

import base64
import csv
import json
import struct
import time
import urllib.request
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOXCPM2_DIR = Path("/Users/vanch/mlx-voxcpm2/benchmarks/official-demo-listening-compare")
OUTPUT_DIR = Path("benchmark/out/zonos2_voxcpm2_listening")
SERVER_URL = "http://127.0.0.1:1920"
SAMPLE_RATE = 44100

# Dialect labels to skip — ZONOS2 only supports Mandarin(zh)
DIALECT_LABELS = {"东北话", "广东话", "闽南语", "上海话", "河南话", "四川话"}

# Style prefix mapping: voxcpm2 style description → ZONOS2 prompt style prefix
STYLE_PREFIX_MAP = {
    # English creative voice design — reasonable mappings
    "Raspy old man": "(Whisper)",           # closest ZONOS2 has for raspy/old
    "Pirate captain": "(Pirate captain)",
    "Soft-spoken": "(Whisper)",
    "Cheerful young woman": "(Excited)",
    "Confident, energetic male sports commentator": "(Excited)",
    "Song: Music, Piano, Sad, Female Vocal": "(Sad)",
    # Skip these — ZONOS2 has no reasonable mapping
    # "Raspy old man" → Whisper (partial)
    # "Little girl, excited" → Excited (partial — missing "little girl")
}

# Chinese creative voice design — all skipped
CHINESE_STYLE_IDS = {49, 50, 51, 52, 53, 54, 56, 58, 59}
# Additional creative voice design IDs to skip (no reasonable mapping)
CREATIVE_SKIP_IDS = {48}  # "Speaking through tears of joy"

# Controllable voice cloning Chinese-style IDs — skipped
CONTROLLABLE_CHINESE_IDS = {61, 63, 64}  # 轻声耳语, Angry tone, 语气愤怒

# ZONOS2 language mapping for Multilingual 30-Language tests
# voxcpm2 label → ZONOS2 language code
LANG_MAP = {
    "English": "en_us",
    "English (US)": "en_us",
    "English (GB)": "en_gb",
    "Chinese": "zh",
    "Hindi": "hi",
    "Spanish": "es",
    "Arabic": "ar",
    "French": "fr",
    "Portuguese": "pt",
    "Russian": "ru",
    "Indonesian": "id",  # ZONOS2 doesn't support id, skip
    "Swahili": "sw",    # ZONOS2 doesn't support sw, skip
    "German": "de",
    "Japanese": "ja",
    "Vietnamese": "vi",
    "Turkish": "tr",
    "Filipino": "tl",   # ZONOS2 doesn't support tl, skip
    "Korean": "ko",
    "Malay": "ms",      # ZONOS2 doesn't support ms, skip
    "Italian": "it",
    "Thai": "th",
    "Burmese": "my",    # ZONOS2 doesn't support my, skip
    "Polish": "pl",
    "Dutch": "nl",
    "Lao": "lo",        # ZONOS2 doesn't support lo, skip
    "Khmer": "km",      # ZONOS2 doesn't support km, skip
    "Greek": "el",      # ZONOS2 doesn't support el, skip
    "Swedish": "sv",
    "Hebrew": "he",
    "Danish": "da",
    "Finnish": "fi",
    "Norwegian": "no",
}

# ZONOS2 supported languages (for multilingual test filtering)
ZONOS2_LANGUAGES = {
    "en_us", "en_gb", "en",   # Tier 1
    "zh", "ja",               # Tier 1
    "ko", "fr", "es", "de",   # Tier 2
    "ru", "pt", "it",         # Tier 2
    "vi", "nl", "pl",         # Tier 2
    "tr", "he", "ar",         # Tier 2
    "sv", "hi", "ta", "te",   # Tier 3
    "th", "da", "no", "fi",   # Tier 3
    "uk",                    # Tier 3
}

# ZONOS2-supported IDs for creative voice design (English style descriptions only)
CREATIVE_VD_IDS = {44, 45, 46, 47, 55, 57, 60}

# ZONOS2-supported IDs for controllable voice cloning (English style + cloning)
CONTROLLABLE_IDS = {65, 66}

# Speaker clone reference audio for cross-lingual tests
SPEAKER_REF_FILE = Path("/tmp/ref_audio_clone.wav")
# Fallback: voxcpm2 reference audio (English, test 037)
SPEAKER_REF_FALLBACK = VOXCPM2_DIR / "refs/037_ref.wav"

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def save_audio_as_wav(pcm_data: bytes, filepath: Path):
    """Save raw PCM float32 data as 16-bit WAV file."""
    num_samples = len(pcm_data) // 4
    float_samples = struct.unpack(f'<{num_samples}f', pcm_data)
    int_samples = [int(max(-32768, min(32767, s * 32767))) for s in float_samples]
    with wave.open(str(filepath), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(struct.pack(f'<{num_samples}h', *int_samples))

def audio_duration(pcm_data: bytes) -> float:
    """Compute audio duration in seconds from PCM float32 data."""
    return len(pcm_data) / (SAMPLE_RATE * 4.0)

# ---------------------------------------------------------------------------
# TTS API
# ---------------------------------------------------------------------------

def fetch_tts_audio(text: str, language: str, seed: int = 42,
                    text_normalization: bool = False,
                    speaker_embedding_base64: str | None = None,
                    style_prefix: str | None = None) -> tuple[bytes, float]:
    """Fetch TTS audio from ZONOS2 server and return (pcm_data, rtf)."""

    body = {
        "text": text,
        "language": language,
        "text_normalization": text_normalization,
        "temperature": 1.15,
        "topk": 106,
        "top_p": 0.0,
        "min_p": 0.18,
        "max_tokens": 1024,
        "stream": False,
        "fade_out_ms": 0.0,
        "accurate_mode": True,
        "seed": seed,
    }
    if style_prefix:
        body["text"] = style_prefix + text
    if speaker_embedding_base64:
        body["speaker_embedding_base64"] = speaker_embedding_base64

    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        SERVER_URL.rstrip("/") + "/tts/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=120) as response:
        pcm_data = response.read()
    elapsed = time.time() - start

    audio_sec = audio_duration(pcm_data)
    rtf = elapsed / audio_sec if audio_sec > 0 else 0
    return pcm_data, rtf

# ---------------------------------------------------------------------------
# Speaker embedding extraction
# ---------------------------------------------------------------------------

def extract_speaker_embedding(ref_wav: Path) -> str:
    """Extract speaker embedding from reference audio, return base64-encoded JSON."""
    from mlx_audio.tts import load as load_tts
    model = load_tts('models/Zyphra-ZONOS2-mlx', lazy=True)
    speaker_emb = model.extract_speaker_embedding(str(ref_wav))
    speaker_emb_list = speaker_emb.tolist()
    return base64.b64encode(json.dumps(speaker_emb_list).encode()).decode()

# ---------------------------------------------------------------------------
# VoxCPM2 manifest parsing
# ---------------------------------------------------------------------------

def load_voxcpm2_manifest() -> list[dict]:
    """Load voxcpm2 manifest CSV and return filtered test rows."""
    rows = []
    with open(VOXCPM2_DIR / "manifest.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["index"])
            label = row["label"]

            # Skip dialect tests
            if label in DIALECT_LABELS:
                continue

            # Skip Chinese-style creative voice design
            if idx in CHINESE_STYLE_IDS:
                continue

            # Skip creative voice design without reasonable ZONOS2 mapping
            if idx in CREATIVE_SKIP_IDS:
                continue

            # Keep creative voice design if English style description
            if 44 <= idx <= 60 and idx not in CHINESE_STYLE_IDS and idx not in CREATIVE_SKIP_IDS:
                rows.append(_make_voxcpm2_test(row))
                continue

            # Keep controllable voice cloning if English style + cloning
            if idx in CONTROLLABLE_IDS:
                rows.append(_make_voxcpm2_test(row))
                continue

            # Keep cross-lingual voice transfer tests
            if 37 <= idx <= 43:
                rows.append(_make_voxcpm2_test(row))
                continue

            # Keep multilingual tests with ZONOS2-supported languages
            if 1 <= idx <= 30:
                zon_lang = LANG_MAP.get(label)
                if zon_lang and zon_lang in ZONOS2_LANGUAGES:
                    rows.append(_make_voxcpm2_test(row))
                # Skip unsupported languages (id, sw, tl, ms, my, lo, km, el)

    return rows

def _make_voxcpm2_test(row: dict) -> dict:
    """Create a ZONOS2-compatible test dict from a voxcpm2 manifest row."""
    idx = int(row["index"])
    label = row["label"]
    text = row["target_text"]
    ref_audio = row.get("reference_audio", "").strip()
    official_audio = row.get("official_audio", "").strip()
    official_dur = row.get("official_duration_s", "0")

    # Determine ZONOS2 language
    if idx in CONTROLLABLE_IDS:
        # English controllable cloning
        zon_lang = "en_us"
    elif idx in CHINESE_STYLE_IDS:
        zon_lang = "zh"  # shouldn't reach here (filtered)
    else:
        zon_lang = LANG_MAP.get(label, "en_us")

    # Determine ZONOS2 label for display
    if idx in CONTROLLABLE_IDS:
        display_label = f"Clone: {label}"
    elif idx in CHINESE_STYLE_IDS:
        display_label = label
    else:
        display_label = label

    # Determine ZONOS2 style prefix
    style_prefix = None
    if 44 <= idx <= 60 and idx not in CHINESE_STYLE_IDS and idx not in CREATIVE_SKIP_IDS:
        # Creative voice design — look up style prefix
        for key, prefix in STYLE_PREFIX_MAP.items():
            if key in label or key.lower() in label.lower():
                style_prefix = prefix
                break
        # If no prefix matched, keep the label as-is (ZONOS2 may still render something)

    return {
        "voxcpm2_idx": idx,
        "voxcpm2_label": label,
        "zon_lang": zon_lang,
        "text": text,
        "display_label": display_label,
        "style_prefix": style_prefix,
        "ref_audio": ref_audio,
        "official_audio": official_audio,
        "official_duration_s": official_dur,
        "voxcpm2_display_group": row.get("display_group", ""),
        "voxcpm2_section_title": row.get("section_title", ""),
    }

# ---------------------------------------------------------------------------
# ZONOS2-only tests (Speaking Rate, Quality, Style, Cloning)
# ---------------------------------------------------------------------------

def make_zonos2_only_tests(speaker_emb_b64: str) -> list[dict]:
    """Generate ZONOS2-only tests not present in voxcpm2."""
    tests = []
    BASE_TEXT = "The quick brown fox jumps over the lazy dog. This tests ZONOS2 capability."
    CLONE_TEXT_ZH = "这是一个中文说话人克隆测试，使用参考音频来复刻说话人特征。"
    CLONE_TEXT_EN = "This is a voice cloning test using a reference audio sample to replicate the speaker's voice characteristics."

    # Speaking Rate — all 8 buckets
    for bucket in range(8):
        tests.append({
            "type": "zosh2_only",
            "category": "Speaking Rate",
            "sub_category": f"Bucket {bucket}",
            "zon_lang": "en_us",
            "text": BASE_TEXT,
            "text_normalization": True,
            "style_prefix": None,
            "speaking_rate_bucket": bucket,
            "display_label": f"Speed Bucket {bucket}",
        })

    # Quality
    for label, q_buckets in [
        ("Trailing Silence Max", {"trailing_silence_s": 7}),
        ("No Trailing Silence", {"trailing_silence_s": 0}),
        ("Leading Silence Max", {"leading_silence_s": 7}),
        ("Balanced All", {"lufs": 5, "estimated_snr": 5, "max_pause": 5,
                         "estimated_bandlimit_hz": 3, "leading_silence_s": 3, "trailing_silence_s": 3}),
    ]:
        tests.append({
            "type": "zosh2_only",
            "category": "Quality",
            "sub_category": label,
            "zon_lang": "en_us",
            "text": "Quality conditioning test with different audio quality feature settings.",
            "text_normalization": True,
            "style_prefix": None,
            "quality_buckets": q_buckets,
            "display_label": label,
        })

    # Style Prefix — all 5
    for style, prefix in [
        ("Whisper", "(Whisper)"),
        ("Shouting", "(Shouting)"),
        ("Sad", "(Sad)"),
        ("Excited", "(Excited)"),
        ("Pirate", "(Pirate captain)"),
    ]:
        tests.append({
            "type": "zosh2_only",
            "category": "Style Prefix",
            "sub_category": style,
            "zon_lang": "en_us",
            "text": "This tests ZONOS2 native style prefix control.",
            "text_normalization": True,
            "style_prefix": prefix,
            "display_label": style,
        })

    # Speaker Cloning
    tests.append({
        "type": "zosh2_only",
        "category": "Speaker Cloning",
        "sub_category": "English",
        "zon_lang": "en_us",
        "text": CLONE_TEXT_EN,
        "text_normalization": True,
        "speaker_embedding_base64": speaker_emb_b64,
        "display_label": "Cloning: English",
    })
    tests.append({
        "type": "zosh2_only",
        "category": "Speaker Cloning",
        "sub_category": "Chinese",
        "zon_lang": "zh",
        "text": CLONE_TEXT_ZH,
        "text_normalization": False,
        "speaker_embedding_base64": speaker_emb_b64,
        "display_label": "Cloning: Chinese",
    })

    return tests

# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------

def main():
    import datetime

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "official").mkdir(exist_ok=True)
    (OUTPUT_DIR / "refs").mkdir(exist_ok=True)
    (OUTPUT_DIR / "zonos2_bf16").mkdir(exist_ok=True)

    # Copy voxcpm2 audio files
    import shutil
    for src_file in (VOXCPM2_DIR / "official").glob("*"):
        shutil.copy2(src_file, OUTPUT_DIR / "official" / src_file.name)
    for src_file in (VOXCPM2_DIR / "refs").glob("*"):
        shutil.copy2(src_file, OUTPUT_DIR / "refs" / src_file.name)

    # Extract speaker embedding for cloning tests
    print("Extracting speaker embedding from reference audio...", flush=True)
    ref_to_use = SPEAKER_REF_FILE if SPEAKER_REF_FILE.exists() else SPEAKER_REF_FALLBACK
    if not ref_to_use.exists():
        print(f"ERROR: Speaker reference audio not found at {SPEAKER_REF_FILE}")
        print(f"Tried fallback at {SPEAKER_REF_FALLBACK}")
        print("Please copy a reference WAV to /tmp/ref_audio_clone.wav and try again.")
        return
    speaker_emb_b64 = extract_speaker_embedding(ref_to_use)
    print(f"Speaker embedding extracted from {ref_to_use} ({len(speaker_emb_b64)} chars)")

    # Load and filter voxcpm2 manifest
    print("\nLoading voxcpm2 manifest...")
    voxcpm2_tests = load_voxcpm2_manifest()
    print(f"  {len(voxcpm2_tests)} tests selected from voxcpm2")

    # Generate ZONOS2-only tests
    zonos2_only_tests = make_zonos2_only_tests(speaker_emb_b64)
    print(f"  {len(zonos2_only_tests)} ZONOS2-only tests added")

    all_tests = voxcpm2_tests + zonos2_only_tests
    print(f"\nTotal: {len(all_tests)} tests")
    print("=" * 60)

    # Run generation
    results = []
    for i, test in enumerate(all_tests, 1):
        test_id = f"{i:03d}"
        if test.get("voxcpm2_idx"):
            test_id = f"v{test['voxcpm2_idx']:03d}"

        category = test.get("voxcpm2_display_group", test.get("category", "Other"))
        if test.get("category"):
            category = f"ZONOS2: {test['category']}"

        wav_filename = f"zonos2_{test_id}.wav"
        wav_path = OUTPUT_DIR / "zonos2_bf16" / wav_filename

        print(f"\n[{i}/{len(all_tests)}] {test_id} | {test['display_label']} ({test['zon_lang']})", flush=True)

        try:
            # Determine text normalization
            if test.get("voxcpm2_idx"):
                tn = test["zon_lang"] in {"en_us", "en_gb", "en"}
            else:
                tn = test.get("text_normalization", True)

            # Get text with style prefix
            text = test["text"]
            if test.get("style_prefix"):
                text = test["style_prefix"] + text

            # Determine speaker embedding
            emb_b64 = None
            if test.get("speaker_embedding_base64"):
                emb_b64 = test["speaker_embedding_base64"]
            elif test.get("ref_audio"):
                # Extract speaker embedding from voxcpm2 reference audio
                ref_path = VOXCPM2_DIR / test["ref_audio"]
                if ref_path.exists():
                    print(f"  Extracting speaker embedding from {test['ref_audio']}...")
                    emb_b64 = extract_speaker_embedding(ref_path)
                else:
                    print(f"  WARNING: Reference audio not found: {ref_path}")

            # Determine request params
            request_params = {
                "text": text,
                "language": test["zon_lang"],
                "seed": 42,
            }
            if tn:
                request_params["text_normalization"] = True

            if test.get("style_prefix"):
                request_params["text"] = test["style_prefix"] + test["text"]
            if emb_b64:
                request_params["speaker_embedding_base64"] = emb_b64
            if test.get("speaking_rate_bucket") is not None:
                request_params["speaking_rate_enabled"] = True
                request_params["speaking_rate_bucket"] = test["speaking_rate_bucket"]
            if test.get("quality_buckets") is not None:
                request_params["quality_enabled"] = True
                request_params["quality_buckets"] = test["quality_buckets"]

            # Fetch TTS audio
            pcm_data, rtf = fetch_tts_audio(
                text=request_params["text"],
                language=test["zon_lang"],
                seed=42,
                text_normalization=tn,
                speaker_embedding_base64=emb_b64,
                style_prefix=test.get("style_prefix"),
            )

            # Save as WAV
            save_audio_as_wav(pcm_data, wav_path)
            duration = audio_duration(pcm_data)
            print(f"  ✓ Generated: {wav_filename} ({duration:.2f}s, RTF={rtf:.3f})", flush=True)

            # Build result
            result = {
                "id": test_id,
                "voxcpm2_idx": test.get("voxcpm2_idx"),
                "voxcpm2_label": test.get("voxcpm2_label"),
                "display_label": test["display_label"],
                "category": category,
                "zon_lang": test["zon_lang"],
                "text": test["text"],
                "api_text": request_params["text"],
                "style_prefix": test.get("style_prefix"),
                "duration_s": duration,
                "rtf": rtf,
                "official_duration_s": float(test.get("official_duration_s", 0)) if test.get("official_duration_s") else None,
                "official_audio": test.get("official_audio"),
                "ref_audio": test.get("ref_audio"),
                "wav_filename": wav_filename,
            }
            results.append(result)

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results.append({
                "id": test_id,
                "voxcpm2_idx": test.get("voxcpm2_idx"),
                "voxcpm2_label": test.get("voxcpm2_label"),
                "display_label": test["display_label"],
                "category": category,
                "zon_lang": test["zon_lang"],
                "text": test["text"],
                "api_text": None,
                "style_prefix": test.get("style_prefix"),
                "duration_s": None,
                "rtf": None,
                "official_duration_s": float(test.get("official_duration_s", 0)) if test.get("official_duration_s") else None,
                "official_audio": test.get("official_audio"),
                "ref_audio": test.get("ref_audio"),
                "wav_filename": None,
                "error": str(e),
            })

    # Summary
    success = [r for r in results if r.get("rtf") is not None]
    failed = [r for r in results if r.get("rtf") is None]
    avg_rtf = sum(r["rtf"] for r in success) / len(success) if success else 0

    print(f"\n{'='*60}")
    print(f"Generation complete:")
    print(f"  Total:  {len(results)}")
    print(f"  Success: {len(success)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Average RTF: {avg_rtf:.3f}")

    # Write report.json
    report = {
        "device": "M3 Max MacBook Pro",
        "model": "mlx-community/Zyphra-ZONOS2 (BF16, MLX)",
        "server": SERVER_URL,
        "sample_rate": SAMPLE_RATE,
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": len(results),
        "successful": len(success),
        "failed": len(failed),
        "avg_rtf": round(avg_rtf, 4),
        "rows": results,
    }
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReport: {report_path}")
    print(f"Audio files: {OUTPUT_DIR / 'zonos2_bf16'}")

# ---------------------------------------------------------------------------
# Quick test endpoint for debugging
# ---------------------------------------------------------------------------

def quick_test():
    """Quick TTS test call."""
    print("Quick TTS test...")
    try:
        pcm, rtf = fetch_tts_audio("Hello, this is a quick test.", "en_us", seed=42)
        print(f"  Result: {len(pcm)} bytes, duration={audio_duration(pcm):.2f}s, RTF={rtf:.3f}")

        # Save test WAV
        test_wav = OUTPUT_DIR / "zonos2_bf16" / "test_quick.wav"
        save_audio_as_wav(pcm, test_wav)
        print(f"  Saved: {test_wav}")
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--quick-test":
        quick_test()
    else:
        main()
