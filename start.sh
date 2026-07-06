#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONPATH="${ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

sanitize_no_proxy() {
  local key value sanitized
  for key in NO_PROXY no_proxy; do
    value="${!key:-}"
    [[ -z "$value" ]] && continue
    sanitized="$(python - "$value" <<'PY'
import sys

entries = [entry.strip() for entry in sys.argv[1].split(",")]
entries = [entry for entry in entries if entry not in {"::1", "::1/128"}]
print(",".join(entries))
PY
)"
    export "${key}=${sanitized}"
  done
}

sanitize_no_proxy

API_HOST="${MLX_ZONOS2_HOST:-127.0.0.1}"
API_PORT="${MLX_ZONOS2_PORT:-1920}"
WEBUI_HOST="${MLX_ZONOS2_WEBUI_HOST:-127.0.0.1}"
WEBUI_PORT="${MLX_ZONOS2_WEBUI_PORT:-7860}"
LOG_DIR="${MLX_ZONOS2_LOG_DIR:-${ROOT}/logs}"

mkdir -p "$LOG_DIR"

API_LOG="${LOG_DIR}/api_server.log"
WEBUI_LOG="${LOG_DIR}/gradio_webui.log"
API_PID=""
WEBUI_PID=""
API_MANAGED=0
WEBUI_MANAGED=0

api_healthy() {
  python - "$API_HOST" "$API_PORT" >/dev/null 2>&1 <<'PY'
import json
import sys
import urllib.request

host, port = sys.argv[1], int(sys.argv[2])
with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2) as response:
    data = json.loads(response.read().decode("utf-8"))
    if data.get("status") != "ok":
        raise SystemExit(1)
PY
}

webui_ready() {
  python - "$WEBUI_HOST" "$WEBUI_PORT" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], int(sys.argv[2])
with urllib.request.urlopen(f"http://{host}:{port}", timeout=2) as response:
    if response.status >= 400:
        raise SystemExit(1)
PY
}

cleanup() {
  local status=$?
  if [[ "$WEBUI_MANAGED" == "1" ]] && [[ -n "${WEBUI_PID}" ]] && kill -0 "$WEBUI_PID" 2>/dev/null; then
    kill "$WEBUI_PID" 2>/dev/null || true
  fi
  if [[ "$API_MANAGED" == "1" ]] && [[ -n "${API_PID}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
  exit "$status"
}

trap cleanup INT TERM EXIT

if api_healthy; then
  echo "Using existing mlx-ZONOS2 API on http://${API_HOST}:${API_PORT}"
else
  echo "Starting mlx-ZONOS2 API on http://${API_HOST}:${API_PORT}"
  python -m mlx_zonos2.server.api_server \
    --host "$API_HOST" \
    --port "$API_PORT" \
    >"$API_LOG" 2>&1 &
  API_PID=$!
  API_MANAGED=1
fi

echo "Waiting for API health check..."
for _ in $(seq 1 600); do
  if [[ "$API_MANAGED" == "1" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
    echo "API server exited during startup. Log:"
    tail -n 80 "$API_LOG" || true
    exit 1
  fi
  if api_healthy; then
    break
  fi
  sleep 1
done

if ! api_healthy; then
  echo "API health check timed out. Log:"
  if [[ "$API_MANAGED" == "1" ]]; then
    tail -n 80 "$API_LOG" || true
  fi
  exit 1
fi

if webui_ready; then
  echo "Using existing Gradio WebUI on http://${WEBUI_HOST}:${WEBUI_PORT}"
else
  echo "Starting Gradio WebUI on http://${WEBUI_HOST}:${WEBUI_PORT}"
  python -m mlx_zonos2.server.gradio_webui \
    --host "$WEBUI_HOST" \
    --port "$WEBUI_PORT" \
    --api-host "$API_HOST" \
    --api-port "$API_PORT" \
    >"$WEBUI_LOG" 2>&1 &
  WEBUI_PID=$!
  WEBUI_MANAGED=1
fi

echo "API log: ${API_LOG}"
echo "WebUI log: ${WEBUI_LOG}"
echo "Open: http://${WEBUI_HOST}:${WEBUI_PORT}"

while true; do
  if [[ "$API_MANAGED" == "1" ]]; then
    if ! kill -0 "$API_PID" 2>/dev/null; then
      echo "API server exited. Log:"
      tail -n 80 "$API_LOG" || true
      exit 1
    fi
  elif ! api_healthy; then
    echo "Existing API server is no longer healthy."
    exit 1
  fi
  if [[ "$WEBUI_MANAGED" == "1" ]]; then
    if ! kill -0 "$WEBUI_PID" 2>/dev/null; then
      echo "Gradio WebUI exited. Log:"
      tail -n 80 "$WEBUI_LOG" || true
      exit 1
    fi
  elif ! webui_ready; then
    echo "Existing Gradio WebUI is no longer reachable."
    exit 1
  fi
  sleep 2
done
