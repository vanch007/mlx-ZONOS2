#!/usr/bin/env bash
# Prefetch runtime dependencies only (DAC + optional speaker encoder).
# Main ZONOS2 MLX weights come from scripts/convert_local_model.sh — not HF mlx-community.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

DAC_MODEL="${MLX_ZONOS2_DAC_MODEL:-mlx-community/descript-audio-codec-44khz}"
SPEAKER_MODEL="${MLX_ZONOS2_SPEAKER_MODEL:-marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B}"
DOWNLOAD_SPEAKER="${MLX_ZONOS2_DOWNLOAD_SPEAKER:-1}"

export DAC_MODEL SPEAKER_MODEL DOWNLOAD_SPEAKER

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repos = [os.environ["DAC_MODEL"]]
if os.environ.get("DOWNLOAD_SPEAKER", "1") == "1":
    repos.append(os.environ["SPEAKER_MODEL"])

for repo in repos:
    print(f"Downloading {repo} ...")
    snapshot_download(repo_id=repo)
    print(f"Done: {repo}")
PY