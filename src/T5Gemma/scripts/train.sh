#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/${ts}_train_lora.log"

echo "=== T5Gemma LoRA train ==="
echo "Config: ${CONFIG}"
echo "Log: ${log_file}"

"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/train_lora.py" \
  --config "${CONFIG}" \
  2>&1 | tee "${log_file}"

