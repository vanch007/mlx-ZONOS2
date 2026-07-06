#!/usr/bin/env python3
"""
Comprehensive test suite for mlx-ZONOS2 following mlx-voxcpm2 testing standards.

Tests:
- English language TTS (en_us, en_gb, en)
- Speaking rate conditioning (8 buckets)
- Quality conditioning (6 features)
- Speaker embedding extraction & caching
- Speaker blend (linear interpolation)
- RTF (Real-Time Factor) measurements
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


# Test cases adapted for mlx-ZONOS2 (English only)
TEST_CASES = [
    # English tests (en_us)
    {
        "id": "en_us_001_news",
        "group": "English (US)",
        "section": "Multilingual",
        "language": "en_us",
        "text": "Breaking news: Apple Silicon achieves record-breaking TTS performance with MLX native inference.",
        "description": "English US news announcer style"
    },
    {
        "id": "en_us_002_conversational",
        "group": "English (US)",
        "section": "Multilingual",
        "language": "en_us",
        "text": "Hey there! Welcome to our podcast. Today we're talking about how AI is changing everything.",
        "description": "English US conversational style"
    },
    {
        "id": "en_us_003_emotional",
        "group": "English (US)",
        "section": "Expressive",
        "language": "en_us",
        "text": "I can't believe we actually did it! After all these years, we finally made it happen!",
        "description": "English US emotional/excited"
    },
    {
        "id": "en_us_004_whisper",
        "group": "English (US)",
        "section": "Controllable",
        "language": "en_us",
        "text": "(Whisper) Shh... this is a secret. Listen very carefully now.",
        "description": "English US whisper style"
    },
    {
        "id": "en_us_005_formal",
        "group": "English (US)",
        "section": "Creative",
        "language": "en_us",
        "text": "(Raspy old man) The world has changed, son. It ain't what it used to be.",
        "description": "Creative: raspy old man"
    },
    {
        "id": "en_us_006_pirate",
        "group": "English (US)",
        "section": "Creative",
        "language": "en_us",
        "text": "(Pirate captain) All hands on deck! Secure the mainsail! We ride this storm or we die trying!",
        "description": "Creative: pirate captain"
    },
    {
        "id": "en_us_007_asmr",
        "group": "English (US)",
        "section": "Creative",
        "language": "en_us",
        "text": "(Soft-spoken, breathy female voice with ASMR quality) Close your eyes and imagine you're lying on a warm beach.",
        "description": "Creative: ASMR female"
    },
    {
        "id": "en_us_008_sports",
        "group": "English (US)",
        "section": "Creative",
        "language": "en_us",
        "text": "(Confident, energetic male sports commentator) He receives the ball in midfield, advances with speed, dribbles past one, past two — he's alone in front of the goalkeeper — he shoots! GOAL!",
        "description": "Creative: sports commentator"
    },
    {
        "id": "en_us_009_cheerful",
        "group": "English (US)",
        "section": "Controllable",
        "language": "en_us",
        "text": "(Cheerful and laughing) I just got the best news — you won't believe what happened today! Everything worked out perfectly!",
        "description": "Controlled: cheerful style"
    },
    {
        "id": "en_us_010_angry",
        "group": "English (US)",
        "section": "Controllable",
        "language": "en_us",
        "text": "(Angry tone, volume gradually increased) Today is the moment of our final confrontation. I will make you suffer here!",
        "description": "Controlled: angry style"
    },
    # English (GB) tests
    {
        "id": "en_gb_001_news",
        "group": "English (GB)",
        "section": "Multilingual",
        "language": "en_gb",
        "text": "Good evening. The Prime Minister has announced new measures to address the cost of living crisis.",
        "description": "English GB news style"
    },
    {
        "id": "en_gb_002_conversational",
        "group": "English (GB)",
        "section": "Multilingual",
        "language": "en_gb",
        "text": "Hello there! Fancy a cuppa? It's a brilliant day for a walk in the park, isn't it?",
        "description": "English GB conversational"
    },
    # English (generic) tests
    {
        "id": "en_001_standard",
        "group": "English (Generic)",
        "section": "Multilingual",
        "language": "en",
        "text": "This is a standard English test with the generic language code.",
        "description": "English generic language code"
    },
]

# Speaking rate buckets to test (8 buckets as per model config)
# Buckets: ['0-8', '8-11', '11-14', '14-17', '17-21', '21-28', '28-40', '40+']
SPEAKING_RATE_BUCKETS = [
    ("bucket_0_very_slow", 0),   # 0-8
    ("bucket_1_slow", 1),         # 8-11
    ("bucket_2_normal", 2),       # 11-14
    ("bucket_3_fast", 3),         # 14-17
    ("bucket_4_very_fast", 4),    # 17-21
    ("bucket_5_rapid", 5),        # 21-28
    ("bucket_6_very_rapid", 6),   # 28-40
    ("bucket_7_extreme", 7),      # 40+
]

# Quality conditioning features to test
QUALITY_FEATURES = [
    {"name": "trailing_silence_3s", "buckets": {"trailing_silence_s": 6}},  # 3s bucket index
    {"name": "trailing_silence_0s", "buckets": {"trailing_silence_s": 0}},  # 0s bucket index
    {"name": "balanced_quality", "buckets": {
        "lufs": 3,
        "estimated_snr": 3,
        "max_pause": 3,
        "estimated_bandlimit_hz": 3,
        "leading_silence_s": 3,
        "trailing_silence_s": 3,
    }},
]


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
    speaking_rate_bucket: str | None = None
    speaking_rate_index: int | None = None
    quality_feature: str | None = None
    speaker_embedding_id: str | None = None


def post_tts(
    url: str,
    text: str,
    language: str,
    max_tokens: int = 1024,
    seed: int | None = None,
    speaking_rate_enabled: bool = False,
    speaking_rate_bucket: int | None = None,
    quality_enabled: bool = False,
    quality_buckets: dict | None = None,
    timeout: float = 600.0,
    chunk_size: int = 65536,
) -> TestResult:
    """Send a TTS request and measure performance."""
    body: dict[str, Any] = {
        "text": text,
        "language": language,
        "text_normalization": True,
        "temperature": 1.15,
        "topk": 106,
        "top_p": 0.0,
        "min_p": 0.18,
        "max_tokens": max_tokens,
        "stream": False,
        "fade_out_ms": 0.0,
        "accurate_mode": True,
    }
    
    if seed is not None:
        body["seed"] = seed
    
    if speaking_rate_enabled and speaking_rate_bucket is not None:
        body["speaking_rate_enabled"] = True
        body["speaking_rate_bucket"] = speaking_rate_bucket
    
    if quality_enabled:
        body["quality_enabled"] = True
        body["quality_buckets"] = quality_buckets or {"trailing_silence_s": 3}
    
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
            test_id="",
            group="",
            section="",
            language=language,
            text=text,
            description="",
            ok=False,
            status=exc.code,
            bytes_received=0,
            sample_rate=sample_rate,
            ttfb_ms=None,
            e2e_ms=elapsed_ms,
            audio_sec=0.0,
            rtf=None,
            error=detail.strip() or exc.reason,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return TestResult(
            test_id="",
            group="",
            section="",
            language=language,
            text=text,
            description="",
            ok=False,
            status=None,
            bytes_received=0,
            sample_rate=sample_rate,
            ttfb_ms=None,
            e2e_ms=elapsed_ms,
            audio_sec=0.0,
            rtf=None,
            error=str(exc),
        )
    
    elapsed = time.perf_counter() - started
    audio_sec = bytes_received / (sample_rate * 4.0)  # float32 = 4 bytes per sample
    
    return TestResult(
        test_id="",
        group="",
        section="",
        language=language,
        text=text,
        description="",
        ok=bytes_received > 0,
        status=status,
        bytes_received=bytes_received,
        sample_rate=sample_rate,
        ttfb_ms=None if first_byte_at is None else (first_byte_at - started) * 1000.0,
        e2e_ms=elapsed * 1000.0,
        audio_sec=audio_sec,
        rtf=None if audio_sec <= 0.0 else elapsed / audio_sec,
        error=None if bytes_received > 0 else "empty audio response",
    )


def run_basic_language_tests(url: str, max_tokens: int, seed: int) -> list[TestResult]:
    """Run basic multi-language tests."""
    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"  [{i+1}/{len(TEST_CASES)}] {case['id']} ({case['language']}) - {case['description'][:50]}...")
        result = post_tts(
            url=url,
            text=case["text"],
            language=case["language"],
            max_tokens=max_tokens,
            seed=seed + i if seed else None,
        )
        result.test_id = case["id"]
        result.group = case["group"]
        result.section = case["section"]
        result.description = case["description"]
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
        if not result.ok:
            print(f"    ERROR: {result.error}")
    return results


def run_speaking_rate_tests(url: str, max_tokens: int, seed: int) -> list[TestResult]:
    """Test speaking rate conditioning across 8 buckets."""
    results = []
    test_text = "The quick brown fox jumps over the lazy dog. This tests speaking rate conditioning across eight buckets."
    
    for i, (bucket_name, bucket_idx) in enumerate(SPEAKING_RATE_BUCKETS):
        print(f"  [rate {i+1}/{len(SPEAKING_RATE_BUCKETS)}] {bucket_name} (index={bucket_idx})...")
        result = post_tts(
            url=url,
            text=test_text,
            language="en_us",
            max_tokens=max_tokens,
            seed=seed + 1000 + i if seed else None,
            speaking_rate_enabled=True,
            speaking_rate_bucket=bucket_idx,
        )
        result.test_id = f"rate_{bucket_name}"
        result.group = "Speaking Rate"
        result.section = "Conditioning"
        result.language = "en_us"
        result.description = f"Speaking rate: {bucket_name} (bucket index {bucket_idx})"
        result.speaking_rate_bucket = bucket_name
        result.speaking_rate_index = bucket_idx
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
    return results


def run_quality_tests(url: str, max_tokens: int, seed: int) -> list[TestResult]:
    """Test quality conditioning features."""
    results = []
    test_text = "Quality conditioning test with different trailing silence and quality feature settings."
    
    for i, qf in enumerate(QUALITY_FEATURES):
        print(f"  [quality {i+1}/{len(QUALITY_FEATURES)}] {qf['name']}...")
        result = post_tts(
            url=url,
            text=test_text,
            language="en_us",
            max_tokens=max_tokens,
            seed=seed + 2000 + i if seed else None,
            quality_enabled=True,
            quality_buckets=qf["buckets"],
        )
        result.test_id = f"quality_{qf['name']}"
        result.group = "Quality Conditioning"
        result.section = "Conditioning"
        result.language = "en_us"
        result.description = f"Quality: {qf['name']} {qf['buckets']}"
        result.quality_feature = qf["name"]
        results.append(result)
        marker = "✓" if result.ok else "✗"
        rtf_str = f" RTF={result.rtf:.3f}" if result.rtf else ""
        print(f"    {marker} e2e={result.e2e_ms:.1f}ms audio={result.audio_sec:.2f}s{rtf_str}")
    return results


def run_speaker_embedding_test(url: str) -> dict | None:
    """Test speaker embedding extraction (placeholder - needs reference audio)."""
    print("  Testing speaker embedding extraction...")
    print("    SKIPPED: No reference audio available in test environment")
    return None


def run_speaker_blend_test(url: str) -> dict | None:
    """Test speaker blend (placeholder - needs reference audio)."""
    print("  Testing speaker blend...")
    print("    SKIPPED: No reference audio available in test environment")
    return None


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
        print(f"  Min:     {min(e2e):.1f}")
        print(f"  Max:     {max(e2e):.1f}")
    
    # Group by section
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
        print(f"\nErrors (first 5):")
        for e in errors[:5]:
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
            "quality_buckets": 60,
        },
        "results": [asdict(r) for r in results],
    }
    
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nJSON report written to: {output_path}")


def write_md_report(results: list[TestResult], total_time: float, output_path: str, server_url: str) -> None:
    """Write Markdown report similar to mlx-voxcpm2 style."""
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
    
    md = f"""# mlx-ZONOS2 Comprehensive Test Report

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
    
    md += f"""
## Test Notes

- **Language Support**: mlx-ZONOS2 currently supports English only (`en_us`, `en_gb`, `en`). The original ZONOS2 model is primarily English-focused.
- **Speaking Rate**: 8 bucket conditioning supported (indices 0-7, ranges: 0-8, 8-11, 11-14, 14-17, 17-21, 21-28, 28-40, 40+ bytes/sec).
- **Quality Features**: 6 features with 60 buckets each (lufs, estimated_snr, max_pause, estimated_bandlimit_hz, leading_silence_s, trailing_silence_s).
- **Speaker Embedding/Blend**: Not tested (requires reference audio files).

Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    Path(output_path).write_text(md, encoding="utf-8")
    print(f"Markdown report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive test suite for mlx-ZONOS2")
    parser.add_argument("--url", default="http://127.0.0.1:1920", help="Server base URL")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--skip-languages", action="store_true", help="Skip multi-language tests")
    parser.add_argument("--skip-rates", action="store_true", help="Skip speaking rate tests")
    parser.add_argument("--skip-quality", action="store_true", help="Skip quality conditioning tests")
    parser.add_argument("--output-json", default="benchmark/out/comprehensive_report.json", help="JSON output path")
    parser.add_argument("--output-md", default="benchmark/out/comprehensive_report.md", help="Markdown output path")
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"mlx-ZONOS2 Comprehensive Test Suite")
    print(f"Server: {args.url}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Seed: {args.seed}")
    print(f"Supported languages: en_us, en_gb, en")
    print(f"Speaking rate buckets: 8")
    print(f"Quality features: 6 (lufs, estimated_snr, max_pause, estimated_bandlimit_hz, leading_silence_s, trailing_silence_s)")
    print(f"{'='*60}\n")
    
    all_results = []
    total_start = time.perf_counter()
    
    # 1. Multi-language tests (English only)
    if not args.skip_languages:
        print("1. Running multi-language tests (English variants)...")
        results = run_basic_language_tests(args.url, args.max_tokens, args.seed)
        all_results.extend(results)
        print()
    
    # 2. Speaking rate conditioning tests
    if not args.skip_rates:
        print("2. Running speaking rate conditioning tests (8 buckets)...")
        results = run_speaking_rate_tests(args.url, args.max_tokens, args.seed)
        all_results.extend(results)
        print()
    
    # 3. Quality conditioning tests
    if not args.skip_quality:
        print("3. Running quality conditioning tests...")
        results = run_quality_tests(args.url, args.max_tokens, args.seed)
        all_results.extend(results)
        print()
    
    # 4. Speaker embedding test (placeholder)
    print("4. Testing speaker embedding extraction...")
    run_speaker_embedding_test(args.url)
    print()
    
    # 5. Speaker blend test (placeholder)
    print("5. Testing speaker blend...")
    run_speaker_blend_test(args.url)
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