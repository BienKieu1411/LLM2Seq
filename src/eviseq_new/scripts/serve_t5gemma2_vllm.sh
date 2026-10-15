#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash serve_t5gemma2_vllm.sh LOCAL_MODEL_DIR [vLLM serve options...]

Environment:
  VLLM_BIN             vLLM executable (default: vllm)
  PYTHON_BIN           Python from the same environment (default: python3)
  HOST                 Bind address (default: 0.0.0.0)
  PORT                 HTTP port (default: 8000)
  SERVED_MODEL_NAME    API model name (default: t5gemma2)
  HF_HUB_OFFLINE       Defaults to 1; the checkpoint must be local
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

MODEL_DIR="$(cd -- "$1" 2>/dev/null && pwd)" || {
  echo "ERROR: local model directory does not exist: $1" >&2
  exit 2
}
shift
CONFIG_FILE="${MODEL_DIR}/config.json"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: missing config.json in local model directory: ${MODEL_DIR}" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! "${PYTHON_BIN}" - "${CONFIG_FILE}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
model_type = str(config.get("model_type", "")).lower()
architectures = config.get("architectures", [])
if model_type != "t5gemma2" and "T5Gemma2ForConditionalGeneration" not in architectures:
    raise SystemExit(
        "Expected a local T5Gemma2 checkpoint; "
        f"model_type={model_type!r}, architectures={architectures!r}"
    )
PY
then
  echo "ERROR: checkpoint config is not recognized as T5Gemma2." >&2
  exit 2
fi

VLLM_BIN="${VLLM_BIN:-vllm}"
if ! command -v "${VLLM_BIN}" >/dev/null 2>&1; then
  echo "ERROR: vLLM executable not found: ${VLLM_BIN}" >&2
  echo "Activate the vLLM environment and install the pinned plugin first." >&2
  exit 127
fi
if ! "${PYTHON_BIN}" -c 'import t5gemma2_vllm_plugin' >/dev/null 2>&1; then
  echo "ERROR: T5Gemma2 plugin is not importable from ${PYTHON_BIN}." >&2
  echo "Run scripts/install_t5gemma2_vllm_plugin.sh in the active vLLM environment." >&2
  exit 1
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

COMMAND=(
  "${VLLM_BIN}" serve "${MODEL_DIR}"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
  --served-model-name "${SERVED_MODEL_NAME:-t5gemma2}"
  --trust-remote-code
  --no-enable-chunked-prefill
)

echo "Starting T5Gemma2 vLLM server from local checkpoint: ${MODEL_DIR}"
echo "vLLM API model name: ${SERVED_MODEL_NAME:-t5gemma2}"
exec "${COMMAND[@]}" "$@"
