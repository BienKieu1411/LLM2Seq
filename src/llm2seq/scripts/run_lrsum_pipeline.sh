#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

MODEL_VARIANT="${1:-${MODEL_VARIANT:-llama}}"
LRSUM_DATASET_NAME="${LRSUM_DATASET_NAME:-bltlab/lr-sum}"
LRSUM_LANGUAGE="${LRSUM_LANGUAGE:-vie}"
DATA_DIR="${LRSUM_DATA_DIR:-${DATA_DIR:-llm2seq/data/processed/lrsum}}"
RUN_PHASE_EVAL="${RUN_PHASE_EVAL:-true}"
LOG_DIR="${LRSUM_LOG_DIR:-${LOG_DIR:-llm2seq/logs}}"
EVAL_ROOT="${LRSUM_EVAL_ROOT:-${EVAL_ROOT:-llm2seq/eval_outputs/lrsum}}"

case "${MODEL_VARIANT}" in
  llama)
    PHASE1_CONFIG="${LRSUM_PHASE1_CONFIG:-${PHASE1_CONFIG:-llm2seq/configs/lrsum_llama_phase1.yaml}}"
    PHASE2_CONFIG="${LRSUM_PHASE2_CONFIG:-${PHASE2_CONFIG:-llm2seq/configs/lrsum_llama_phase2.yaml}}"
    PHASE1_DIR="${LRSUM_PHASE1_DIR:-${PHASE1_DIR:-runs/lrsum_llama_phase1_warmup}}"
    PHASE2_DIR="${LRSUM_PHASE2_DIR:-${PHASE2_DIR:-runs/lrsum_llama_phase2_lora_encoder}}"
    ;;
  qwen)
    PHASE1_CONFIG="${LRSUM_PHASE1_CONFIG:-${PHASE1_CONFIG:-llm2seq/configs/lrsum_qwen_phase1.yaml}}"
    PHASE2_CONFIG="${LRSUM_PHASE2_CONFIG:-${PHASE2_CONFIG:-llm2seq/configs/lrsum_qwen_phase2.yaml}}"
    PHASE1_DIR="${LRSUM_PHASE1_DIR:-${PHASE1_DIR:-runs/lrsum_qwen_phase1_warmup}}"
    PHASE2_DIR="${LRSUM_PHASE2_DIR:-${PHASE2_DIR:-runs/lrsum_qwen_phase2_lora_encoder}}"
    ;;
  *)
    echo "Usage: bash llm2seq/scripts/run_lrsum_pipeline.sh [llama|qwen]" >&2
    exit 2
    ;;
esac

PHASE2_EVAL_DIR="${LRSUM_PHASE2_EVAL_DIR:-${EVAL_ROOT}/${MODEL_VARIANT}_phase2}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== Prepare Vietnamese LR-Sum (${MODEL_VARIANT}) ==="
"${PYTHON_BIN}" llm2seq/scripts/prepare_lrsum_json.py \
  --dataset_name "${LRSUM_DATASET_NAME}" \
  --language "${LRSUM_LANGUAGE}" \
  --output_dir "${DATA_DIR}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_prepare_lrsum_${MODEL_VARIANT}.log"

echo "=== Train LR-Sum Phase 1 (${MODEL_VARIANT}) ==="
bash llm2seq/scripts/train_phase1.sh "${PHASE1_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_lrsum_${MODEL_VARIANT}_phase1.log"

echo "=== Train LR-Sum Phase 2 (${MODEL_VARIANT}) ==="
bash llm2seq/scripts/train_phase2.sh "${PHASE1_DIR}/best.pt" "${PHASE2_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_lrsum_${MODEL_VARIANT}_phase2.log"

if [[ "${RUN_PHASE_EVAL}" == "true" || "${RUN_PHASE_EVAL}" == "1" ]]; then
  TEST_FILE="${DATA_DIR}/test.jsonl" \
  bash llm2seq/scripts/evaluate_phase.sh \
    "lrsum_${MODEL_VARIANT}_phase2" \
    "${PHASE2_CONFIG}" \
    "${PHASE2_DIR}/best.pt" \
    "${PHASE2_EVAL_DIR}" \
    autoregressive
fi
