#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

PHASE1_CONFIG="${PHASE1_CONFIG:-llm2seq/configs/wikilingua_phase1.yaml}"
PHASE2_CONFIG="${PHASE2_CONFIG:-llm2seq/configs/wikilingua_phase2.yaml}"
PHASE3_CONFIG="${PHASE3_CONFIG:-llm2seq/configs/wikilingua_phase3.yaml}"
PHASE1_DIR="${PHASE1_DIR:-runs/phase1_warmup}"
PHASE2_DIR="${PHASE2_DIR:-runs/phase2_lora_encoder}"
PHASE3_DIR="${PHASE3_DIR:-runs/phase3_mtp_self_distill}"
EVAL_ROOT="${EVAL_ROOT:-llm2seq/eval_outputs}"
PHASE1_EVAL_DIR="${PHASE1_EVAL_DIR:-${EVAL_ROOT}/full_test_phase1_main}"
PHASE2_EVAL_DIR="${PHASE2_EVAL_DIR:-${EVAL_ROOT}/full_test_phase2_main}"
PHASE3_EVAL_DIR="${PHASE3_EVAL_DIR:-${EVAL_DIR:-${EVAL_ROOT}/full_test_phase3_main}}"
PHASE3_MTP_EVAL_DIR="${PHASE3_MTP_EVAL_DIR:-${EVAL_ROOT}/full_test_phase3_mtp_verified}"
PHASE3_SPEED_COMPARE_DIR="${PHASE3_SPEED_COMPARE_DIR:-${EVAL_ROOT}/phase3_speed_comparison}"
RUN_PHASE_EVAL="${RUN_PHASE_EVAL:-true}"

bash llm2seq/scripts/train_phase1.sh "${PHASE1_CONFIG}"
if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  bash llm2seq/scripts/evaluate_phase.sh phase1_main "${PHASE1_CONFIG}" "${PHASE1_DIR}/best.pt" "${PHASE1_EVAL_DIR}" autoregressive
fi

bash llm2seq/scripts/train_phase2.sh "${PHASE1_DIR}/best.pt" "${PHASE2_CONFIG}"
if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  bash llm2seq/scripts/evaluate_phase.sh phase2_main "${PHASE2_CONFIG}" "${PHASE2_DIR}/best.pt" "${PHASE2_EVAL_DIR}" autoregressive
fi

bash llm2seq/scripts/train_phase3.sh "${PHASE2_DIR}/best.pt" "${PHASE3_CONFIG}"
if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  bash llm2seq/scripts/evaluate_phase.sh phase3_main "${PHASE3_CONFIG}" "${PHASE3_DIR}/best.pt" "${PHASE3_EVAL_DIR}" autoregressive "${PHASE2_DIR}/best.pt"
  bash llm2seq/scripts/evaluate_phase.sh phase3_mtp "${PHASE3_CONFIG}" "${PHASE3_DIR}/best.pt" "${PHASE3_MTP_EVAL_DIR}" mtp_verified "${PHASE2_DIR}/best.pt"
  "${PYTHON_BIN}" llm2seq/scripts/compare_speed_metrics.py \
    --main_metrics "${PHASE3_EVAL_DIR}/metrics.json" \
    --mtp_metrics "${PHASE3_MTP_EVAL_DIR}/metrics.json" \
    --output_dir "${PHASE3_SPEED_COMPARE_DIR}"
fi
