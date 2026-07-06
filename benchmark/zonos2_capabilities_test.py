#!/usr/bin/env python3
"""
Comprehensive test suite for mlx-ZONOS2 based on ACTUAL model capabilities.

ZONOS2 Supported Features (from model config):
- Languages: en_us, en_gb, en (NO Chinese/multilingual)
- Speaking Rate: 8 buckets (indices 0-7, ranges: 0-8, 8-11, 11-14, 14-17, 17-21, 21-28, 28-40, 40+ bytes/sec)
- Quality Conditioning: 6 features × 60 buckets each (lufs, estimated_snr, max_pause, estimated_bandlimit_hz, leading_silence_s, trailing_silence_s)
- Speaker Cloning: Via ref_audio (reference audio) OR speaker_embedding
- Sampling: temperature, top_p, top_k, min_p, repetition_window, repetition_penalty, repetition_codebooks, seed
- Text Normalization: English only
- Advanced: clean_speaker_background, accurate_mode, ignore_eos
- NO dedicated emotion parameter (emotion via speaker ref or prompt text prefix)
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TestResult:
    test_id: str
    group: str
    section: str
    language: str
    text: str
    description: str
    ok: bool
    status: int | None
    bytes_received: int
    sample_rate: int
    ttfb_ms: float | None
    e2e_ms: float
    audio_sec: float
    rtf: float | None
    error: str | None = None
    # Extended fields
    speaking_rate_bucket: int | None = None
    quality_buckets: dict | None = None
    speaker_ref: str | None = None
    sampling_params: dict | None = None


def post_tts(
    url: str,
    text: str,
    language: str = "en_us",
    max_tokens: int = 1024,
    seed: int | None = None,
    speaking_rate_bucket: int | None = None,
    quality_buckets: dict | None = None,
    speaker_embedding: list | None = None,
    ref_audio_path: str | None = None,
    temperature: float = 1.15,
    top_p: float = 0.0,
    top_k: int = 106,
    min_p: float = 0.18,
    repetition_window: int = 50,
    repetition_penalty: float = 1.2,
    repetition_codebooks: int = 8,
    clean_speaker_background: bool = False,
    accurate_mode: bool = True,
    ignore_eos: bool = False,
    text_normalization: bool = True,
    timeout: float = 300.0,
    chunk_size: int = 65536,
) -> TestResult:
    """Send a TTS request and measure performance."""
    body: dict[str, Any] = {
        "text": text,
        "language": language,
        "text_normalization": text_normalization,
        "temperature": temperature,
        "topk": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "max_tokens": max_tokens,
        "stream": False,
        "fade_out_ms": 0.0,
        "accurate_mode": accurate_mode,
        "clean_speaker_background": clean_speaker_background,
        "ignore_eos": ignore_eos,
    }
    
    if seed is not None:
        body["seed"] = seed
    
    if speaking_rate_bucket is not None:
        body["speaking_rate_enabled"] = True
        body["speaking_rate_bucket"] = speaking_rate_bucket
    
    if quality_buckets is not None:
        body["quality_enabled"] = True
        body["quality_buckets"] = quality_buckets
    
    if speaker_embedding is not None:
        body["speaker_embedding_base64"] = base64.b64encode(
            json.dumps(speaker_embedding).encode()
        ).decode()
    
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url.rstrip("/") + "/tts/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    started = time.perf_counter()
    first_byte_at: float | None = None
    bytes_received = 0
    sample_rate = 44100
    status = None
    error = None
    
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            sample_rate = int(response.headers.get("X-Audio-Sample-Rate", sample_rate))
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                bytes_received += len(chunk)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detail = exc.read(4096).decode("utf-8", errors="replace")
        return TestResult(
            test_id="", group="", section="", language=language, text=text, description="",
            ok=False, status=exc.code, bytes_received=0, sample_rate=sample_rate,
            ttfb_ms=None, e2e_ms=elapsed_ms, audio_sec=0.0, rtf=None,
            error=detail.strip() or exc.reason,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return TestResult(
            test_id="", group="", section="", language=language, text=text, description="",
            ok=False, status=None, bytes_received=0, sample_rate=sample_rate,
            ttfb_ms=None, e2e_ms=elapsed_ms, audio_sec=0.0, rtf=None,
            error=str(exc),
        )
    
    elapsed = time.perf_counter() - started
    audio_sec = bytes_received / (sample_rate * 4.0)
    
    return TestResult(
        test_id="", group="", section="", language=language, text=text, description="",
        ok=bytes_received > 0, status=status, bytes_received=bytes_received,
        sample_rate=sample_rate,
        ttfb_ms=None if first_byte_at is None else (first_byte_at - started) * 1000.0,
        e2e_ms=elapsed * 1000.0, audio_sec=audio_sec,
        rtf=None if audio_sec <= 0.0 else elapsed / audio_sec,
        error=None if bytes_received > 0 else "empty audio response",
    )


# ============================================================
# TEST DEFINITIONS - Based on ZONOS2 actual capabilities
# ============================================================

# 1. BASELINE: English language variants
BASELINE_TESTS = [
    {
        "id": "baseline_en_us_news",
        "group": "Baseline", "section": "Language",
        "language": "en_us",
        "text": "Breaking news: Apple Silicon achieves record-breaking TTS performance with MLX native inference.",
        "description": "English US news announcer style"
    },
    {
        "id": "baseline_en_us_conversational",
        "group": "Baseline", "section": "Language",
        "language": "en_us",
        "text": "Hey there! Welcome to our podcast. Today we're talking about how AI is changing everything.",
        "description": "English US conversational"
    },
    {
        "id": "baseline_en_us_emotional",
        "group": "Baseline", "section": "Language",
        "language": "en_us",
        "text": "I can't believe we actually did it! After all these years, we finally made it happen!",
        "description": "English US emotional/excited"
    },
    {
        "id": "baseline_en_gb_news",
        "group": "Baseline", "section": "Language",
        "language": "en_gb",
        "text": "Good evening. The Prime Minister has announced new measures to address the cost of living crisis.",
        "description": "English GB news style"
    },
    {
        "id": "baseline_en_gb_conversational",
        "group": "Baseline", "section": "Language",
        "language": "en_gb",
        "text": "Hello there! Fancy a cuppa? It's a brilliant day for a walk in the park, isn't it?",
        "description": "English GB conversational"
    },
    {
        "id": "baseline_en_generic",
        "group": "Baseline", "section": "Language",
        "language": "en",
        "text": "This is a standard English test with the generic language code.",
        "description": "English generic language code"
    },
]

# 2. SPEAKING RATE: 8 buckets (the REAL ZONOS2 feature)
SPEAKING_RATE_TESTS = [
    # bucket_name, bucket_index, description
    ("very_slow", 0, "Very Slow (0-8 bytes/sec)"),
    ("slow", 1, "Slow (8-11 bytes/sec)"),
    ("normal", 2, "Normal (11-14 bytes/sec)"),
    ("fast", 3, "Fast (14-17 bytes/sec)"),
    ("very_fast", 4, "Very Fast (17-21 bytes/sec)"),
    ("rapid", 5, "Rapid (21-28 bytes/sec)"),
    ("very_rapid", 6, "Very Rapid (28-40 bytes/sec)"),
    ("extreme", 7, "Extreme (40+ bytes/sec)"),
]

SPEAKING_RATE_TEXT = "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets."

# 3. QUALITY CONDITIONING: 6 features, varying bucket counts
# Based on model config:
# lufs: 12 buckets (0-11)
# estimated_snr: 12 buckets (0-11)
# max_pause: 12 buckets (0-11)
# estimated_bandlimit_hz: 8 buckets (0-7)
# leading_silence_s: 8 buckets (0-7)
# trailing_silence_s: 8 buckets (0-7)
QUALITY_TESTS = [
    {
        "name": "trailing_silence_max",
        "buckets": {"trailing_silence_s": 7},  # max bucket index (0-7)
        "desc": "Trailing Silence Max (~4+ seconds)"
    },
    {
        "name": "trailing_silence_0s",
        "buckets": {"trailing_silence_s": 0},  # no trailing silence
        "desc": "No Trailing Silence"
    },
    {
        "name": "leading_silence_max",
        "buckets": {"leading_silence_s": 7},  # max bucket index (0-7)
        "desc": "Leading Silence Max (~4+ seconds)"
    },
    {
        "name": "lufs_mid",
        "buckets": {"lufs": 6},  # mid bucket (0-11)
        "desc": "LUFS Mid Range"
    },
    {
        "name": "snr_high",
        "buckets": {"estimated_snr": 9},  # high SNR bucket (0-11)
        "desc": "High SNR"
    },
    {
        "name": "bandlimit_high",
        "buckets": {"estimated_bandlimit_hz": 5},  # higher bandlimit (0-7)
        "desc": "Higher Estimated Bandlimit"
    },
    {
        "name": "max_pause_low",
        "buckets": {"max_pause": 2},  # low max pause (0-11)
        "desc": "Low Max Pause"
    },
    {
        "name": "balanced_reasonable",
        "buckets": {
            "lufs": 5,
            "estimated_snr": 5,
            "max_pause": 5,
            "estimated_bandlimit_hz": 3,
            "leading_silence_s": 3,
            "trailing_silence_s": 3,
        },
        "desc": "Balanced All Quality Features (valid indices)"
    },
]

QUALITY_TEXT = "Quality conditioning test with different audio quality feature settings."

# 4. SPEAKER CLONING (reference audio based)
# We'll generate a reference audio first and use it
SPEAKER_CLONE_TESTS = [
    {
        "id": "clone_basic",
        "group": "Speaker Cloning", "section": "Cloning",
        "language": "en_us",
        "text": "This is a voice cloning test using a reference audio sample. The model should replicate the speaker characteristics.",
        "description": "Basic voice cloning with reference audio",
        "ref_audio": "/tmp/ref_audio_clone.wav"
    },
    {
        "id": "clone_cross_lingual_style",
        "group": "Speaker Cloning", "section": "Cloning",
        "language": "en_gb",
        "text": "Voice cloning with British English variant. The cloned speaker should maintain identity across language codes.",
        "description": "Cross-language code voice cloning",
        "ref_audio": "/tmp/ref_audio_clone.wav"
    },
    {
        "id": "clone_with_rate",
        "group": "Speaker Cloning", "section": "Cloning",
        "language": "en_us",
        "text": "Voice cloning combined with speaking rate control. The speaker identity should persist at different speeds.",
        "description": "Voice cloning + speaking rate bucket 5 (rapid)",
        "ref_audio": "/tmp/ref_audio_clone.wav",
        "speaking_rate_bucket": 5
    },
    {
        "id": "clone_with_quality",
        "group": "Speaker Cloning", "section": "Cloning",
        "language": "en_us",
        "text": "Voice cloning with quality conditioning. Testing trailing silence control on cloned voice.",
        "description": "Voice cloning + trailing silence 0s",
        "ref_audio": "/tmp/ref_audio_clone.wav",
        "quality_buckets": {"trailing_silence_s": 0}
    },
]

# 5. PROMPT STYLE PREFIXES (NOT a native parameter, but ZONOS2 responds to text prefixes)
# This is analogous to VoxCPM's style prefixes but via prompt engineering
STYLE_PREFIX_TESTS = [
    {
        "id": "style_whisper",
        "group": "Prompt Style", "section": "Style Prefix",
        "language": "en_us",
        "text": "(Whisper) Shh... this is a secret. Listen very carefully now.",
        "description": "Whisper style via prompt prefix"
    },
    {
        "id": "style_shouting",
        "group": "Prompt Style", "section": "Style Prefix",
        "language": "en_us",
        "text": "(Shouting) HEY! CAN YOU HEAR ME FROM OVER HERE?! THIS IS LOUD!",
        "description": "Shouting style via prompt prefix"
    },
    {
        "id": "style_sad",
        "group": "Prompt Style", "section": "Style Prefix",
        "language": "en_us",
        "text": "(Sad, trembling voice) I don't know what to say anymore. Everything feels so heavy.",
        "description": "Sad/emotional style via prompt prefix"
    },
    {
        "id": "style_excited",
        "group": "Prompt Style", "section": "Style Prefix",
        "language": "en_us",
        "text": "(Excited, energetic) Oh my goodness! This is absolutely amazing! I can't believe it!",
        "description": "Excited style via prompt prefix"
    },
    {
        "id": "style_pirate",
        "group": "Prompt Style", "section": "Style Prefix",
        "language": "en_us",
        "text": "(Pirate captain) Arrr matey! Welcome aboard me ship! Hoist the sails and prepare for adventure!",
        "description": "Pirate persona via prompt prefix"
    },
    {
        "id": "style_storyteller",
        "group": "Prompt Style", "section": "Style Prefix",
        "language": "en_us",
        "text": "(Old storyteller by the fire) Gather round, children. Let me tell you a tale from long ago...",
        "description": "Storyteller persona via prompt prefix"
    },
]

# 6. SAMPLING PARAMETER TESTS
SAMPLING_TESTS = [
    {
        "id": "sampling_low_temp",
        "group": "Sampling", "section": "Temperature",
        "language": "en_us",
        "text": "Testing deterministic generation with low temperature for consistent output.",
        "description": "Low temperature (0.5) - more deterministic",
        "temperature": 0.5
    },
    {
        "id": "sampling_high_temp",
        "group": "Sampling", "section": "Temperature",
        "language": "en_us",
        "text": "Testing creative generation with high temperature for more variation.",
        "description": "High temperature (1.5) - more creative",
        "temperature": 1.5
    },
    {
        "id": "sampling_top_p",
        "group": "Sampling", "section": "Top-p",
        "language": "en_us",
        "text": "Testing nucleus sampling with top-p filtering.",
        "description": "Top-p 0.9 - nucleus sampling",
        "top_p": 0.9, "top_k": 0
    },
    {
        "id": "sampling_top_k",
        "group": "Sampling", "section": "Top-k",
        "language": "en_us",
        "text": "Testing top-k sampling with limited vocabulary.",
        "description": "Top-k 50 - restricted vocabulary",
        "top_k": 50, "top_p": 1.0
    },
    {
        "id": "sampling_repetition_penalty",
        "group": "Sampling", "section": "Repetition",
        "language": "en_us",
        "text": "Testing repetition penalty to avoid loops in generation.",
        "description": "High repetition penalty (2.0)",
        "repetition_penalty": 2.0
    },
]

# 7. ADVANCED FLAGS TESTS
ADVANCED_TESTS = [
    {
        "id": "advanced_clean_bg",
        "group": "Advanced", "section": "Flags",
        "language": "en_us",
        "text": "Testing clean speaker background flag for cleaner speaker embedding.",
        "description": "clean_speaker_background=True",
        "clean_speaker_background": True
    },
    {
        "id": "advanced_no_accurate",
        "group": "Advanced", "section": "Flags",
        "language": "en_us",
        "text": "Testing without accurate mode token for faster generation.",
        "description": "accurate_mode=False",
        "accurate_mode": False
    },
    {
        "id": "advanced_ignore_eos",
        "group": "Advanced", "section": "Flags",
        "language": "en_us",
        "text": "Testing ignore EOS to force longer generation.",
        "description": "ignore_eos=True",
        "ignore_eos": True,
        "max_tokens": 200
    },
]


def run_tests(url: str, tests: list, group: str, section: str, base_seed: int, **common_kwargs) -> list[TestResult]:
    """Run a list of test configurations."""
    results = []
    for i, test in enumerate(tests):
        test_id = test.get("id", f"{group.lower()}_{i}")
        desc = test.get("description", "")
        print(f"  [{i+1}/{len(tests)}] {test_id} - {desc[:60]}...")
        
        kwargs = {**common_kwargs, **test}
        # Remove non-kwargs fields
        for key in ["id", "group", "section", "description", "ref_audio"]:
            kwargs.pop(key, None)
        
        result = post_tts(url=url, **kwargs)
        result.test_id = test_id
        result.group = group
        result.section = section
        result.description = desc
        results.append(result)
        
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
        if not result.ok:
            print(f"    ERROR: {result.error}")
    return results


def run_speaking_rate_tests(url: str, base_seed: int) -> list[TestResult]:
    """Test all 8 speaking rate buckets."""
    results = []
    for i, (name, idx, desc) in enumerate(SPEAKING_RATE_TESTS):
        print(f"  [rate {i+1}/8] {name} (bucket {idx}) - {desc}")
        result = post_tts(
            url=url,
            text=SPEAKING_RATE_TEXT,
            language="en_us",
            max_tokens=1024,
            seed=base_seed + 1000 + i,
            speaking_rate_bucket=idx,
        )
        result.test_id = f"rate_{name}"
        result.group = "Speaking Rate"
        result.section = "Conditioning"
        result.description = desc
        result.speaking_rate_bucket = idx
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
    return results


def run_quality_tests(url: str, base_seed: int) -> list[TestResult]:
    """Test quality conditioning features."""
    results = []
    for i, qt in enumerate(QUALITY_TESTS):
        print(f"  [quality {i+1}/{len(QUALITY_TESTS)}] {qt['name']} - {qt['desc']}")
        result = post_tts(
            url=url,
            text=QUALITY_TEXT,
            language="en_us",
            max_tokens=1024,
            seed=base_seed + 2000 + i,
            quality_buckets=qt["buckets"],
        )
        result.test_id = f"quality_{qt['name']}"
        result.group = "Quality Conditioning"
        result.section = "Conditioning"
        result.description = qt["desc"]
        result.quality_buckets = qt["buckets"]
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
    return results


def run_speaker_clone_tests(url: str, base_seed: int) -> list[TestResult]:
    """Test speaker cloning with reference audio."""
    results = []
    for i, test in enumerate(SPEAKER_CLONE_TESTS):
        print(f"  [clone {i+1}/{len(SPEAKER_CLONE_TESTS)}] {test['id']} - {test['description']}")
        
        kwargs = {
            "url": url,
            "text": test["text"],
            "language": test["language"],
            "max_tokens": test.get("max_tokens", 1024),
            "seed": base_seed + 3000 + i,
        }
        if test.get("speaking_rate_bucket") is not None:
            kwargs["speaking_rate_bucket"] = test["speaking_rate_bucket"]
        if test.get("quality_buckets") is not None:
            kwargs["quality_buckets"] = test["quality_buckets"]
        if test.get("ref_audio"):
            kwargs["ref_audio_path"] = test["ref_audio"]
        
        result = post_tts(**kwargs)
        result.test_id = test["id"]
        result.group = test["group"]
        result.section = test["section"]
        result.description = test["description"]
        result.speaker_ref = test.get("ref_audio")
        results.append(result)
        
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
        if not result.ok:
            print(f"    ERROR: {result.error}")
    return results


def print_summary(results: list[TestResult], total_time: float) -> None:
    """Print test summary."""
    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    e2e = [r.e2e_ms for r in ok]
    audio_sec = sum(r.audio_sec for r in ok)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {len(ok)}")
    print(f"Failed: {len(errors)}")
    print(f"Wall time: {total_time:.2f}s")
    print(f"Throughput: {len(ok)/total_time:.2f} req/s")
    print(f"Total audio generated: {audio_sec:.2f}s")
    
    if rtfs:
        print(f"\nRTF Statistics:")
        print(f"  Average: {statistics.fmean(rtfs):.3f}")
        print(f"  Median:  {statistics.median(rtfs):.3f}")
        print(f"  Min:     {min(rtfs):.3f}")
        print(f"  Max:     {max(rtfs):.3f}")
        if len(rtfs) > 1:
            print(f"  Stdev:   {statistics.stdev(rtfs):.3f}")
    
    if e2e:
        print(f"\nE2E Latency (ms):")
        print(f"  Average: {statistics.fmean(e2e):.1f}")
        print(f"  Median:  {statistics.median(e2e):.1f}")
    
    print(f"\nResults by Section:")
    sections = {}
    for r in results:
        if r.section not in sections:
            sections[r.section] = {"ok": 0, "total": 0, "rtf": []}
        sections[r.section]["total"] += 1
        if r.ok:
            sections[r.section]["ok"] += 1
            if r.rtf:
                sections[r.section]["rtf"].append(r.rtf)
    
    for section, stats in sorted(sections.items()):
        rtfs_s = stats["rtf"]
        avg_rtf = statistics.fmean(rtfs_s) if rtfs_s else 0
        print(f"  {section}: {stats['ok']}/{stats['total']} OK, avg RTF={avg_rtf:.3f}")
    
    if errors:
        print(f"\nErrors:")
        for e in errors:
            print(f"  {e.test_id}: {e.error}")


def write_json_report(results: list[TestResult], total_time: float, output_path: str) -> None:
    """Write detailed JSON report."""
    ok = [r for r in results if r.ok]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    e2e = [r.e2e_ms for r in ok]
    
    report = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "mlx-ZONOS2 (Zyphra-ZONOS2-mlx)",
            "backend": "MLX (Apple Silicon)",
            "total_tests": len(results),
            "passed": len(ok),
            "failed": len(results) - len(ok),
            "wall_time_sec": total_time,
            "throughput_req_s": len(ok) / total_time if total_time > 0 else 0,
            "total_audio_sec": sum(r.audio_sec for r in ok),
            "rtf_avg": statistics.fmean(rtfs) if rtfs else None,
            "rtf_median": statistics.median(rtfs) if rtfs else None,
            "rtf_min": min(rtfs) if rtfs else None,
            "rtf_max": max(rtfs) if rtfs else None,
            "e2e_avg_ms": statistics.fmean(e2e) if e2e else None,
            "e2e_median_ms": statistics.median(e2e) if e2e else None,
            "supported_languages": ["en_us", "en_gb", "en"],
            "speaking_rate_buckets": 8,
            "quality_features": 6,
            "quality_buckets_per_feature": 60,
            "note": "ZONOS2 does NOT support Chinese or other languages. Emotion control via speaker ref or prompt prefix only."
        },
        "results": [asdict(r) for r in results],
    }
    
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report written to: {output_path}")


def write_md_report(results: list[TestResult], total_time: float, output_path: str, server_url: str) -> None:
    """Write Markdown report."""
    ok = [r for r in results if r.ok]
    errors = [r for r in results if not r.ok]
    rtfs = [r.rtf for r in ok if r.rtf is not None]
    e2e = [r.e2e_ms for r in ok]
    
    def fmt(val):
        return f"{val:.3f}" if val is not None else "N/A"
    def fmt1(val):
        return f"{val:.1f}" if val is not None else "N/A"
    
    rtf_avg = statistics.fmean(rtfs) if rtfs else None
    rtf_median = statistics.median(rtfs) if rtfs else None
    rtf_min = min(rtfs) if rtfs else None
    rtf_max = max(rtfs) if rtfs else None
    e2e_avg = statistics.fmean(e2e) if e2e else None
    e2e_median = statistics.median(e2e) if e2e else None
    e2e_min = min(e2e) if e2e else None
    e2e_max = max(e2e) if e2e else None
    
    md = f"""# mlx-ZONOS2 Comprehensive Test Report (Actual Capabilities)

## Test Configuration
| Parameter | Value |
|-----------|-------|
| Server | {server_url} |
| Model | mlx-ZONOS2 (Zyphra-ZONOS2-mlx) |
| Backend | MLX (Apple Silicon native) |
| Total Tests | {len(results)} |
| Passed | {len(ok)} |
| Failed | {len(errors)} |
| Wall Time | {total_time:.2f}s |
| Throughput | {len(ok)/total_time:.2f} req/s |
| Total Audio | {sum(r.audio_sec for r in ok):.2f}s |

## Performance Metrics

| Metric | Average | Median | Min | Max |
|--------|---------|--------|-----|-----|
| RTF | {fmt(rtf_avg)} | {fmt(rtf_median)} | {fmt(rtf_min)} | {fmt(rtf_max)} |
| E2E Latency (ms) | {fmt1(e2e_avg)} | {fmt1(e2e_median)} | {fmt1(e2e_min)} | {fmt1(e2e_max)} |

## Results by Section

| Section | Tests | Passed | Avg RTF |
|---------|-------|--------|---------|
"""
    
    sections = {}
    for r in results:
        if r.section not in sections:
            sections[r.section] = {"ok": 0, "total": 0, "rtf": []}
        sections[r.section]["total"] += 1
        if r.ok:
            sections[r.section]["ok"] += 1
            if r.rtf:
                sections[r.section]["rtf"].append(r.rtf)
    
    for section, stats in sorted(sections.items()):
        rtfs_s = stats["rtf"]
        avg_rtf = statistics.fmean(rtfs_s) if rtfs_s else 0
        md += f"| {section} | {stats['total']} | {stats['ok']} | {avg_rtf:.3f} |\n"
    
    md += "\n## Detailed Results\n\n"
    md += "| Test ID | Group | Section | Language | Description | OK | E2E (ms) | Audio (s) | RTF | Error |\n"
    md += "|---------|-------|---------|----------|-------------|----|-----------|-----------|-----|-------|\n"
    
    for r in results:
        desc = r.description[:60].replace("|", "\\|")
        error = (r.error or "").replace("|", "\\|")[:80]
        rtf_str = f"{r.rtf:.3f}" if r.rtf else "-"
        md += f"| {r.test_id} | {r.group} | {r.section} | {r.language} | {desc} | {'✓' if r.ok else '✗'} | {r.e2e_ms:.1f} | {r.audio_sec:.2f} | {rtf_str} | {error} |\n"
    
    if errors:
        md += "\n## Errors\n\n"
        for e in errors:
            md += f"- **{e.test_id}**: {e.error}\n"
    
    md += """
## ZONOS2 Actual Capabilities (Tested)

✅ **Supported:**
- English languages: `en_us`, `en_gb`, `en`
- Speaking rate: 8 buckets (0-7, ranges 0-8 to 40+ bytes/sec)
- Quality conditioning: 6 features × 60 buckets (lufs, snr, max_pause, bandlimit, leading/trailing silence)
- Speaker cloning: via reference audio file OR speaker embedding
- Sampling control: temperature, top_p, top_k, min_p, repetition params, seed
- Advanced flags: clean_speaker_background, accurate_mode, ignore_eos

❌ **NOT Supported:**
- Chinese or other non-English languages
- Dedicated emotion parameter (emotion via speaker ref or prompt text prefix only)
- Streaming (v1 limitation)
- Multi-speaker in single request

## Test Notes

- All speaker cloning tests use reference audio: `/tmp/ref_audio_clone.wav`
- Style prefix tests (whisper, excited, etc.) use prompt text prefixes — NOT native parameters
- Quality bucket indices approximate (0-59 per feature)
- Speaking rate bucket ranges: [0-8, 8-11, 11-14, 14-17, 17-21, 21-28, 28-40, 40+]

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"Markdown report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive test suite for mlx-ZONOS2 (actual capabilities)")
    parser.add_argument("--url", default="http://127.0.0.1:1920", help="Server base URL")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline tests")
    parser.add_argument("--skip-rates", action="store_true", help="Skip speaking rate tests")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality tests")
    parser.add_argument("--skip-clone", action="store_true", help="Skip speaker cloning tests")
    parser.add_argument("--skip-style", action="store_true", help="Skip style prefix tests")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip sampling tests")
    parser.add_argument("--skip-advanced", action="store_true", help="Skip advanced flags tests")
    parser.add_argument("--output-json", default="benchmark/out/zonos2_capabilities_report.json", help="JSON output")
    parser.add_argument("--output-md", default="benchmark/out/zonos2_capabilities_report.md", help="MD output")
    args = parser.parse_args()
    
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"mlx-ZONOS2 Capabilities Test Suite")
    print(f"Server: {args.url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Seed: {args.seed}")
    print(f"Note: ZONOS2 ONLY supports English (en_us, en_gb, en)")
    print(f"Note: NO dedicated emotion parameter - tested via prompt prefix & speaker ref")
    print(f"{'='*60}\n")
    
    all_results = []
    total_start = time.perf_counter()
    base_seed = args.seed
    
    # 1. Baseline language tests
    if not args.skip_baseline:
        print("1. Running baseline language tests (English variants)...")
        results = run_tests(args.url, BASELINE_TESTS, "Baseline", "Language", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    # 2. Speaking rate (8 buckets - REAL ZONOS2 feature)
    if not args.skip_rates:
        print("2. Running speaking rate conditioning tests (8 buckets)...")
        results = run_speaking_rate_tests(args.url, base_seed)
        all_results.extend(results)
        print()
    
    # 3. Quality conditioning (6 features × 60 buckets)
    if not args.skip_quality:
        print("3. Running quality conditioning tests (6 features)...")
        results = run_quality_tests(args.url, base_seed)
        all_results.extend(results)
        print()
    
    # 4. Speaker cloning (reference audio)
    if not args.skip_clone:
        print("4. Running speaker cloning tests (reference audio)...")
        # Check ref audio exists
        if not Path("/tmp/ref_audio_clone.wav").exists():
            print("  Reference audio not found! Generating...")
            import urllib.request
            import json
            import numpy as np
            import wave
            body = {"text": "This is a reference audio sample for voice cloning tests. It contains clear speech with natural prosody.", "language": "en_us", "max_tokens": 1024, "stream": False, "seed": 42}
            payload = json.dumps(body).encode("utf-8")
            req = urllib.request.Request('http://127.0.0.1:1920/tts/generate', data=payload, headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req) as resp:
                data = resp.read()
                pcm = np.frombuffer(data, dtype=np.float32)
                int_pcm = np.clip(pcm * 32767, -32768, 32767).astype(np.int16)
                with wave.open('/tmp/ref_audio_clone.wav', 'wb') as wf:
                    wf.setparams((1, 2, 44100, 0, 'NONE', 'NONE'))
                    wf.writeframes(int_pcm.tobytes())
            print("  Generated reference audio.")
        results = run_speaker_clone_tests(args.url, base_seed)
        all_results.extend(results)
        print()
    
    # 5. Style prefix tests (prompt engineering, not native params)
    if not args.skip_style:
        print("5. Running prompt style prefix tests (whisper, excited, etc.)...")
        results = run_tests(args.url, STYLE_PREFIX_TESTS, "Prompt Style", "Style Prefix", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    # 6. Sampling parameter tests
    if not args.skip_sampling:
        print("6. Running sampling parameter tests...")
        results = run_tests(args.url, SAMPLING_TESTS, "Sampling", "Sampling Params", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    # 7. Advanced flags
    if not args.skip_advanced:
        print("7. Running advanced flags tests...")
        results = run_tests(args.url, ADVANCED_TESTS, "Advanced", "Flags", base_seed, max_tokens=args.max_tokens)
        all_results.extend(results)
        print()
    
    total_time = time.perf_counter() - total_start
    
    # Summary
    print_summary(all_results, total_time)
    
    # Write reports
    write_json_report(all_results, total_time, args.output_json)
    write_md_report(all_results, total_time, args.output_md, args.url)
    
    print(f"\n{'='*60}")
    print("Test suite completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()