#!/usr/bin/env bash
# Convert local PyTorch ZONOS2 weights to MLX safetensors (no mlx-community download).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE="${ZONOS2_PYTORCH_MODEL:-/Users/vanch/ZONOS2/models/ZONOS2}"
OUT="${MLX_ZONOS2_MODEL:-$ROOT/models/Zyphra-ZONOS2-mlx}"
DTYPE="${MLX_ZONOS2_DTYPE:-bfloat16}"
DAC_REPO="${MLX_ZONOS2_DAC_MODEL:-mlx-community/descript-audio-codec-44khz}"
SPEAKER_REPO="${MLX_ZONOS2_SPEAKER_MODEL:-marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B}"
SKIP_SPEAKER="${MLX_ZONOS2_SKIP_SPEAKER_ENCODER:-0}"

if [[ ! -f "$SOURCE/params.json" ]]; then
  echo "ERROR: params.json not found under $SOURCE" >&2
  echo "Set ZONOS2_PYTORCH_MODEL to your local PyTorch checkpoint directory." >&2
  exit 1
fi
if [[ ! -f "$SOURCE/model.pth" && ! -f "$SOURCE/model.pt" ]]; then
  echo "ERROR: model.pth/model.pt not found under $SOURCE" >&2
  exit 1
fi

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! python -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('mlx_audio.tts.models.zonos2.convert') else 1)"; then
  echo "ERROR: mlx-audio zonos2 convert module missing." >&2
  echo "Install with: pip install -e '.[convert]'" >&2
  exit 1
fi

echo "Source (PyTorch): $SOURCE"
echo "Output (MLX):     $OUT"
echo "dtype:            $DTYPE"

if [[ "$SKIP_SPEAKER" == "1" ]]; then
  python -m mlx_audio.tts.models.zonos2.convert \
    --hf-path "$SOURCE" \
    --mlx-path "$OUT" \
    --dtype "$DTYPE" \
    --dac-repo "$DAC_REPO" \
    --speaker-encoder-repo "$SPEAKER_REPO" \
    --skip-speaker-encoder
else
  python -m mlx_audio.tts.models.zonos2.convert \
    --hf-path "$SOURCE" \
    --mlx-path "$OUT" \
    --dtype "$DTYPE" \
    --dac-repo "$DAC_REPO" \
    --speaker-encoder-repo "$SPEAKER_REPO"
fi

echo "Done. Default model path: $OUT"
echo "Next: python scripts/smoke_load.py --lazy"