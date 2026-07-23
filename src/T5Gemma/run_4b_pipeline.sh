#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${PROJECT_ROOT}"

# Keep the 4B run completely separate from the 1B baseline. These explicit
# paths intentionally override values loaded from the shared env.txt.
CONFIG_4B="${T5GEMMA_4B_CONFIG:-T5Gemma/configs/wikilingua_full_4b_4b_3072.yaml}"
RUN_DIR_4B="${T5GEMMA_4B_RUN_DIR:-runs/t5gemma2_4b_4b_full_wikilingua}"
EVAL_DIR_4B="${T5GEMMA_4B_EVAL_DIR:-T5Gemma/eval_outputs/full_test_4b_4b}"
LOG_DIR_4B="${T5GEMMA_4B_LOG_DIR:-T5Gemma/logs/4b_4b}"

mkdir -p "${LOG_DIR_4B}"

echo "=== Prepare WikiLingua for T5Gemma 4B-4B ==="
"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/prepare_wikilingua_json.py" \
  --input_dir "${WIKI_DIR}" \
  --output_dir "${DATA_DIR}"

train_log="${LOG_DIR_4B}/$(date +%Y%m%d_%H%M%S)_train_full.log"
train_args=(--config "${CONFIG_4B}")
if [[ "${OVERWRITE_OUTPUT_DIR,,}" == "true" || "${OVERWRITE_OUTPUT_DIR}" == "1" || "${OVERWRITE_OUTPUT_DIR,,}" == "yes" ]]; then
  train_args+=(--overwrite-output-dir)
fi

echo "=== Full fine-tune T5Gemma 2 4B-4B ==="
echo "Config: ${CONFIG_4B}"
echo "Log: ${train_log}"
"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/train_full.py" \
  "${train_args[@]}" \
  2>&1 | tee "${train_log}"

if [[ "${FORCE_EVAL:-false}" =~ ^(true|1|yes)$ ]] || \
   [[ "${RUN_EVAL,,}" == "true" || "${RUN_EVAL}" == "1" || "${RUN_EVAL,,}" == "yes" ]]; then
  eval_log="${LOG_DIR_4B}/$(date +%Y%m%d_%H%M%S)_evaluate_full_test.log"
  echo "=== Evaluate T5Gemma 2 4B-4B on the full test set ==="
  "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
    --config "${CONFIG_4B}" \
    --checkpoint "${RUN_DIR_4B}/final_model" \
    --test_file "${TEST_FILE:-${DATA_DIR}/test.jsonl}" \
    --output_dir "${EVAL_DIR_4B}" \
    --limit "${EVAL_LIMIT}" \
    2>&1 | tee "${eval_log}"
fi
