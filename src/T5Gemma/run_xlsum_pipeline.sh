#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${USE_BIENKIEU_ENV:-true}" == "true" || "${USE_BIENKIEU_ENV:-true}" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate bienkieu_env >/dev/null 2>&1 || true
  fi
fi

source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${PROJECT_ROOT}"

export CONFIG="${XLSUM_CONFIG:-T5Gemma/configs/xlsum_lora.yaml}"
export XLSUM_ARCHIVE="${XLSUM_ARCHIVE:-llm2seq/datasets/vietnamese_XLSum.tar.bz2}"
export DATA_DIR="${XLSUM_DATA_DIR:-T5Gemma/data/processed/xlsum}"
export RUN_DIR="${XLSUM_RUN_DIR:-runs/xlsum_t5gemma2_1b_1b_lora}"
export EVAL_DIR="${XLSUM_EVAL_DIR:-T5Gemma/eval_outputs/xlsum/full_test}"
export TEST_FILE="${XLSUM_TEST_FILE:-${DATA_DIR}/test.jsonl}"
export LOG_DIR="${XLSUM_LOG_DIR:-T5Gemma/logs}"
export MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-32}"

echo "=== Prepare Vietnamese XLSum (T5Gemma2) ==="
"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/prepare_xlsum_json.py" \
  --archive "${XLSUM_ARCHIVE}" \
  --output_dir "${DATA_DIR}" \
  --min_target_tokens "${MIN_TARGET_TOKENS}"

echo "=== Train T5Gemma2 LoRA on XLSum ==="
bash "${T5GEMMA_ROOT}/scripts/train.sh"

if [[ "${RUN_EVAL,,}" == "true" || "${RUN_EVAL}" == "1" || "${RUN_EVAL,,}" == "yes" ]]; then
  echo "=== Evaluate T5Gemma2 XLSum Full Test ==="
  bash "${T5GEMMA_ROOT}/scripts/evaluate.sh"
fi
