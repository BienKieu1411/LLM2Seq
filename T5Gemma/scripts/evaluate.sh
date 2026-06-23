#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/${ts}_evaluate_full_test.log"

adapter_path="${ADAPTER_PATH:-${RUN_DIR}/best_adapter}"
test_file="${TEST_FILE:-${DATA_DIR}/test.jsonl}"

echo "=== T5Gemma full-test eval ==="
echo "Config: ${CONFIG}"
echo "Adapter: ${adapter_path}"
echo "Test file: ${test_file}"
echo "Output: ${EVAL_DIR}"
echo "Limit: ${EVAL_LIMIT}"
echo "Log: ${log_file}"

"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
  --config "${CONFIG}" \
  --adapter "${adapter_path}" \
  --test_file "${test_file}" \
  --output_dir "${EVAL_DIR}" \
  --limit "${EVAL_LIMIT}" \
  2>&1 | tee "${log_file}"

