#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_DIR}"
ts="$(date +%Y%m%d_%H%M%S)"
log_file="${LOG_DIR}/${ts}_evaluate_full_test.log"

checkpoint_path="${CHECKPOINT_PATH:-${RUN_DIR}/final_model}"
test_args=()
if [[ -n "${TEST_FILE:-}" ]]; then
  test_file="${TEST_FILE}"
  test_args=(--test_file "${test_file}")
else
  test_file="<from config data.test_file>"
fi

echo "=== T5Gemma full-test eval ==="
echo "Config: ${CONFIG}"
echo "Checkpoint: ${checkpoint_path}"
echo "Test file: ${test_file}"
echo "Output: ${EVAL_DIR}"
echo "Limit: ${EVAL_LIMIT}"
echo "Log: ${log_file}"

"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
  --config "${CONFIG}" \
  --checkpoint "${checkpoint_path}" \
  --output_dir "${EVAL_DIR}" \
  --limit "${EVAL_LIMIT}" \
  "${test_args[@]}" \
  2>&1 | tee "${log_file}"
