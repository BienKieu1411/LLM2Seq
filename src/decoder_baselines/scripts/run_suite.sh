#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="$(cd "${BASELINE_ROOT}/.." && pwd)"
cd "${SRC_ROOT}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
elif [[ -x "/Users/kieugiangbien/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-/Users/kieugiangbien/bienkieu_env/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
if [[ -z "${GPU_ID}" || "${GPU_ID}" == *,* || "${GPU_ID}" == *[[:space:]]* ]]; then
  echo "GPU_ID must identify exactly one GPU, for example GPU_ID=0." >&2
  exit 2
fi
export GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="${BASELINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG="${CONFIG:-${BASELINE_ROOT}/configs/suite.yaml}"
exec "${PYTHON_BIN}" -m decoder_baselines.suite --config "${CONFIG}" "$@"
