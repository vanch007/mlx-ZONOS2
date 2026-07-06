from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any


DEFAULT_TEXTS = [
    "Hello from Zonos two. This is a short benchmark request.",
    "Apple Silicon local inference should start with low concurrency.",
    "The benchmark measures time to first audio byte and end to end latency.",
]


@dataclass
class BenchResult:
    index: int
    ok: bool
    status: int | None
    bytes_received: int
    sample_rate: int
    ttfb_ms: float | None
    e2e_ms: float
    audio_sec: float
    rtf: float | None
    error: str | None = None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def post_tts(index: int, args: argparse.Namespace) -> BenchResult:
    text = args.text or DEFAULT_TEXTS[index % len(DEFAULT_TEXTS)]
    body: dict[str, Any] = {
        "text": text,
        "language": args.language,
        "text_normalization": args.text_normalization,
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "seed": None if args.seed is None else args.seed + index,
    }
    if args.fade_out_ms:
        body["fade_out_ms"] = args.fade_out_ms

    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        args.url.rstrip("/") + "/tts/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_byte_at: float | None = None
    bytes_received = 0
    sample_rate = 44100

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.getcode()
            sample_rate = int(response.headers.get("X-Audio-Sample-Rate", sample_rate))
            while True:
                chunk = response.read(args.chunk_size)
                if not chunk:
                    break
                if first_byte_at is None:
                    first_byte_at = time.perf_counter()
                bytes_received += len(chunk)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        detail = exc.read(4096).decode("utf-8", errors="replace")
        return BenchResult(
            index=index,
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
        return BenchResult(
            index=index,
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
    audio_sec = bytes_received / (sample_rate * 4.0)
    return BenchResult(
        index=index,
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


def print_summary(results: list[BenchResult], started: float, sample_rate: int = 44100) -> None:
    total_elapsed = time.perf_counter() - started
    ok = [item for item in results if item.ok]
    errors = [item for item in results if not item.ok]
    ttfb = [item.ttfb_ms for item in ok if item.ttfb_ms is not None]
    e2e = [item.e2e_ms for item in ok]
    rtfs = [item.rtf for item in ok if item.rtf is not None]
    audio_sec = sum(item.audio_sec for item in ok)

    print(f"requests={len(results)} ok={len(ok)} errors={len(errors)}")
    print(f"wall_time_sec={total_elapsed:.3f} throughput_req_s={len(ok) / total_elapsed:.3f}")
    if ok:
        if ttfb:
            print(
                "ttfb_ms "
                f"avg={statistics.fmean(ttfb):.1f} p50={percentile(ttfb, 0.50):.1f} "
                f"p90={percentile(ttfb, 0.90):.1f}"
            )
        print(
            "e2e_ms "
            f"avg={statistics.fmean(e2e):.1f} p50={percentile(e2e, 0.50):.1f} "
            f"p90={percentile(e2e, 0.90):.1f}"
        )
        print(
            f"audio_sec_total={audio_sec:.3f} "
            f"rtf_avg={statistics.fmean(rtfs):.3f}"
        )
    for item in errors[:3]:
        print(f"error[{item.index}] status={item.status} detail={item.error}")


def write_md_summary(results: list[BenchResult], started: float, server_url: str, output_md: str) -> None:
    """Write a Markdown-formatted summary."""
    total_elapsed = time.perf_counter() - started
    ok = [item for item in results if item.ok]
    errors = [item for item in results if not item.ok]
    e2e = [item.e2e_ms for item in ok]
    rtfs = [item.rtf for item in ok if item.rtf is not None]

    md = f"""# mlx-ZONOS2 TTS Benchmark

| Parameter | Value |
|-----------|-------|
| Server | {server_url} |
| Requests | {len(results)} |
| Success | {len(ok)} |
| Errors | {len(errors)} |
| Wall Time | {total_elapsed:.3f}s |

## Metrics

| Metric | Average | P50 | P90 |
|--------|---------|-----|-----|
"""
    if e2e:
        md += f"| E2E (ms) | {statistics.fmean(e2e):.1f} | {percentile(e2e, 0.50):.1f} | {percentile(e2e, 0.90):.1f} |\n"
    if rtfs:
        md += f"| RTF | {statistics.fmean(rtfs):.3f} | {percentile(rtfs, 0.50):.3f} | {percentile(rtfs, 0.90):.3f} |\n"

    md += f"\nThroughput: {len(ok) / total_elapsed:.3f} req/s\n"

    if errors:
        md += "\n## Errors\n\n"
        for item in errors[:5]:
            md += f"- Request {item.index}: status={item.status} — {item.error}\n"

    Path(output_md).write_text(md, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark mlx-ZONOS2 /tts/generate.")
    parser.add_argument("--url", default="http://127.0.0.1:1920", help="Server base URL.")
    parser.add_argument("--requests", type=int, default=3, help="Number of TTS requests.")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent request workers.")
    parser.add_argument("--max-tokens", type=int, default=64, help="Generation token cap per request.")
    parser.add_argument("--text", default=None, help="Use one fixed text for every request.")
    parser.add_argument("--language", default="en_us", help="TTS language code.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed; incremented per request.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument("--chunk-size", type=int, default=65536, help="Response read chunk size.")
    parser.add_argument("--fade-out-ms", type=float, default=0.0, help="Optional fade-out in ms.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text-normalization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-json", help="Write per-request results to this JSON file.")
    parser.add_argument("--output-md", help="Write summary in Markdown format.")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Assumed sample rate for duration calc.")
    args = parser.parse_args()
    if args.requests <= 0:
        parser.error("--requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    results: list[BenchResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(post_tts, index, args) for index in range(args.requests)]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            marker = "ok" if result.ok else "err"
            print(
                f"{marker}[{result.index}] status={result.status} "
                f"ttfb_ms={result.ttfb_ms if result.ttfb_ms is not None else '-'} "
                f"e2e_ms={result.e2e_ms:.1f} audio_sec={result.audio_sec:.3f} "
                f"bytes={result.bytes_received}"
            )

    results.sort(key=lambda item: item.index)
    print_summary(results, started, args.sample_rate)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as file:
            json.dump([asdict(item) for item in results], file, indent=2)
    if args.output_md:
        write_md_summary(results, started, args.url, args.output_md)


if __name__ == "__main__":
    main()