#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${USE_BIENKIEU_ENV:-true}" == "true" || "${USE_BIENKIEU_ENV:-true}" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate bienkieu_env >/dev/null 2>&1 || true
  fi
fi

source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

MODEL_VARIANT="${1:-${MODEL_VARIANT:-llama}}"
XLSUM_ARCHIVE="${XLSUM_ARCHIVE:-llm2seq/datasets/vietnamese_XLSum.tar.bz2}"
DATA_DIR="${XLSUM_DATA_DIR:-llm2seq/data/processed/xlsum}"
MIN_TARGET_TOKENS="${MIN_TARGET_TOKENS:-32}"
RUN_PHASE_EVAL="${RUN_PHASE_EVAL:-true}"
LOG_DIR="${XLSUM_LOG_DIR:-llm2seq/logs}"
EVAL_ROOT="${XLSUM_EVAL_ROOT:-llm2seq/eval_outputs/xlsum}"

case "${MODEL_VARIANT}" in
  llama)
    PHASE1_CONFIG="${XLSUM_PHASE1_CONFIG:-llm2seq/configs/xlsum_llama_phase1.yaml}"
    PHASE2_CONFIG="${XLSUM_PHASE2_CONFIG:-llm2seq/configs/xlsum_llama_phase2.yaml}"
    PHASE1_DIR="${XLSUM_PHASE1_DIR:-runs/xlsum_llama_phase1_warmup}"
    PHASE2_DIR="${XLSUM_PHASE2_DIR:-runs/xlsum_llama_phase2_lora_encoder}"
    ;;
  qwen)
    PHASE1_CONFIG="${XLSUM_PHASE1_CONFIG:-llm2seq/configs/xlsum_qwen_phase1.yaml}"
    PHASE2_CONFIG="${XLSUM_PHASE2_CONFIG:-llm2seq/configs/xlsum_qwen_phase2.yaml}"
    PHASE1_DIR="${XLSUM_PHASE1_DIR:-runs/xlsum_qwen_phase1_warmup}"
    PHASE2_DIR="${XLSUM_PHASE2_DIR:-runs/xlsum_qwen_phase2_lora_encoder}"
    ;;
  *)
    echo "Usage: bash llm2seq/scripts/run_xlsum_pipeline.sh [llama|qwen]" >&2
    exit 2
    ;;
esac

PHASE1_EVAL_DIR="${XLSUM_PHASE1_EVAL_DIR:-${EVAL_ROOT}/${MODEL_VARIANT}_phase1}"
PHASE2_EVAL_DIR="${XLSUM_PHASE2_EVAL_DIR:-${EVAL_ROOT}/${MODEL_VARIANT}_phase2}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== Prepare Vietnamese XLSum (${MODEL_VARIANT}) ==="
"${PYTHON_BIN}" llm2seq/scripts/prepare_xlsum_json.py \
  --archive "${XLSUM_ARCHIVE}" \
  --output_dir "${DATA_DIR}" \
  --min_target_tokens "${MIN_TARGET_TOKENS}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_prepare_xlsum_${MODEL_VARIANT}.log"

echo "=== Train XLSum Phase 1 (${MODEL_VARIANT}) ==="
bash llm2seq/scripts/train_phase1.sh "${PHASE1_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_xlsum_${MODEL_VARIANT}_phase1.log"

# Phase 1 evaluation is skipped as requested
# if [[ "${RUN_PHASE_EVAL}" == "true" || "${RUN_PHASE_EVAL}" == "1" ]]; then
#   bash llm2seq/scripts/evaluate_phase.sh \
#     "xlsum_${MODEL_VARIANT}_phase1" \
#     "${PHASE1_CONFIG}" \
#     "${PHASE1_DIR}/best.pt" \
#     "${PHASE1_EVAL_DIR}" \
#     autoregressive
# fi

echo "=== Train XLSum Phase 2 (${MODEL_VARIANT}) ==="
bash llm2seq/scripts/train_phase2.sh "${PHASE1_DIR}/best.pt" "${PHASE2_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_xlsum_${MODEL_VARIANT}_phase2.log"

if [[ "${RUN_PHASE_EVAL}" == "true" || "${RUN_PHASE_EVAL}" == "1" ]]; then
  bash llm2seq/scripts/evaluate_phase.sh \
    "xlsum_${MODEL_VARIANT}_phase2" \
    "${PHASE2_CONFIG}" \
    "${PHASE2_DIR}/best.pt" \
    "${PHASE2_EVAL_DIR}" \
    autoregressive
fi
