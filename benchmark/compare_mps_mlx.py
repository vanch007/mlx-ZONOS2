from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path


def run_bench(
    *,
    label: str,
    url: str,
    requests: int,
    max_tokens: int,
    output_json: Path,
) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("bench_tts.py")),
        "--url",
        url,
        "--requests",
        str(requests),
        "--max-tokens",
        str(max_tokens),
        "--no-stream",
        "--no-text-normalization",
        "--output-json",
        str(output_json),
    ]
    print(f"[{label}] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    rows = json.loads(output_json.read_text(encoding="utf-8"))
    ok = [row for row in rows if row.get("ok")]
    e2e = [float(row["e2e_ms"]) for row in ok]
    rtfs = [float(row["rtf"]) for row in ok if row.get("rtf") is not None]
    return {
        "label": label,
        "url": url,
        "requests": requests,
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "e2e_ms_avg": statistics.fmean(e2e) if e2e else None,
        "rtf_avg": statistics.fmean(rtfs) if rtfs else None,
        "output_json": str(output_json),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B benchmark MPS ZONOS2 vs mlx-ZONOS2.")
    parser.add_argument("--mps-url", default="http://127.0.0.1:1919")
    parser.add_argument("--mlx-url", default="http://127.0.0.1:1920")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out-dir", default="benchmark/out")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mps = run_bench(
        label="mps",
        url=args.mps_url,
        requests=args.requests,
        max_tokens=args.max_tokens,
        output_json=out_dir / "mps_result.json",
    )
    mlx = run_bench(
        label="mlx",
        url=args.mlx_url,
        requests=args.requests,
        max_tokens=args.max_tokens,
        output_json=out_dir / "mlx_result.json",
    )

    speedup = None
    if mps["rtf_avg"] and mlx["rtf_avg"] and mlx["rtf_avg"] > 0:
        speedup = mps["rtf_avg"] / mlx["rtf_avg"]

    summary = {"mps": mps, "mlx": mlx, "rtf_speedup_mlx_over_mps": speedup}
    summary_path = out_dir / "compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== A/B summary ===")
    print(f"mps rtf_avg={mps['rtf_avg']}")
    print(f"mlx rtf_avg={mlx['rtf_avg']}")
    print(f"speedup (higher is better for MLX)={speedup}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()