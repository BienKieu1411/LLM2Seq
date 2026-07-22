#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${PROJECT_ROOT}"

export CONFIG="${LRSUM_CONFIG:-T5Gemma/configs/lrsum_full.yaml}"
export LRSUM_DATASET_NAME="${LRSUM_DATASET_NAME:-bltlab/lr-sum}"
export LRSUM_LANGUAGE="${LRSUM_LANGUAGE:-vie}"
export HF_REPO_ID="${LRSUM_HF_REPO_ID:-${HF_REPO_ID:-}}"
export DATA_DIR="${LRSUM_DATA_DIR:-T5Gemma/data/processed/lrsum}"
export RUN_DIR="${LRSUM_RUN_DIR:-runs/lrsum_t5gemma2_1b_1b_full}"
export EVAL_DIR="${LRSUM_EVAL_DIR:-T5Gemma/eval_outputs/lrsum/full_test}"
export TEST_FILE="${LRSUM_TEST_FILE:-${DATA_DIR}/test.jsonl}"
export LOG_DIR="${LRSUM_LOG_DIR:-${LOG_DIR:-T5Gemma/logs}}"

echo "=== Prepare Vietnamese LR-Sum (T5Gemma2) ==="
"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/prepare_lrsum_json.py" \
  --dataset_name "${LRSUM_DATASET_NAME}" \
  --language "${LRSUM_LANGUAGE}" \
  --output_dir "${DATA_DIR}"

echo "=== Full fine-tune T5Gemma2 on LR-Sum ==="
bash "${T5GEMMA_ROOT}/scripts/train.sh"

if [[ "${RUN_EVAL,,}" == "true" || "${RUN_EVAL}" == "1" || "${RUN_EVAL,,}" == "yes" ]]; then
  echo "=== Evaluate T5Gemma2 LR-Sum Full Test ==="
  bash "${T5GEMMA_ROOT}/scripts/evaluate.sh"
fi
