#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

REPO_ID="${HF_REPO_ID:-}"
DATA_DIR="${DATA_DIR:-llm2seq_final/data/processed}"
PHASE1_CONFIG="${PHASE1_CONFIG:-llm2seq_final/configs/phase1_warmup_4096.yaml}"
PHASE2_CONFIG="${PHASE2_CONFIG:-llm2seq_final/configs/phase2_lora_encoder_4096.yaml}"
PHASE3_CONFIG="${PHASE3_CONFIG:-llm2seq_final/configs/phase3_mtp_self_distill_4096.yaml}"
PHASE1_DIR="${PHASE1_DIR:-runs/h200_llm2seq_phase1_warmup}"
PHASE2_DIR="${PHASE2_DIR:-runs/h200_llm2seq_phase2_lora_encoder}"
PHASE3_DIR="${PHASE3_DIR:-runs/h200_llm2seq_phase3_mtp_self_distill}"
EVAL_ROOT="${EVAL_ROOT:-llm2seq_final/eval_outputs}"
PHASE1_EVAL_DIR="${PHASE1_EVAL_DIR:-${EVAL_ROOT}/full_test_phase1_main}"
PHASE2_EVAL_DIR="${PHASE2_EVAL_DIR:-${EVAL_ROOT}/full_test_phase2_main}"
PHASE3_EVAL_DIR="${PHASE3_EVAL_DIR:-${EVAL_DIR:-${EVAL_ROOT}/full_test_phase3_main}}"
PHASE3_MTP_EVAL_DIR="${PHASE3_MTP_EVAL_DIR:-${EVAL_ROOT}/full_test_phase3_mtp_verified}"
PHASE3_SPEED_COMPARE_DIR="${PHASE3_SPEED_COMPARE_DIR:-${EVAL_ROOT}/phase3_speed_comparison}"
RUN_PHASE_EVAL="${RUN_PHASE_EVAL:-true}"
LOG_DIR="${LOG_DIR:-llm2seq_final/logs}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== Resuming: Evaluate Phase 1 ==="
if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  bash llm2seq_final/scripts/evaluate_phase.sh \
    phase1_main \
    "${PHASE1_CONFIG}" \
    "${PHASE1_DIR}/best.pt" \
    "${PHASE1_EVAL_DIR}" \
    autoregressive
fi

echo "=== Train Phase 2 ==="
bash llm2seq_final/scripts/train_phase2.sh "${PHASE1_DIR}/best.pt" "${PHASE2_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_phase2.log"

if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  bash llm2seq_final/scripts/evaluate_phase.sh \
    phase2_main \
    "${PHASE2_CONFIG}" \
    "${PHASE2_DIR}/best.pt" \
    "${PHASE2_EVAL_DIR}" \
    autoregressive
fi

echo "=== Train Phase 3 ==="
bash llm2seq_final/scripts/train_phase3.sh "${PHASE2_DIR}/best.pt" "${PHASE3_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_phase3.log"

if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  bash llm2seq_final/scripts/evaluate_phase.sh \
    phase3_main \
    "${PHASE3_CONFIG}" \
    "${PHASE3_DIR}/best.pt" \
    "${PHASE3_EVAL_DIR}" \
    autoregressive \
    "${PHASE2_DIR}/best.pt"

  bash llm2seq_final/scripts/evaluate_phase.sh \
    phase3_mtp \
    "${PHASE3_CONFIG}" \
    "${PHASE3_DIR}/best.pt" \
    "${PHASE3_MTP_EVAL_DIR}" \
    mtp_verified \
    "${PHASE2_DIR}/best.pt"

  "${PYTHON_BIN}" llm2seq_final/scripts/compare_speed_metrics.py \
    --main_metrics "${PHASE3_EVAL_DIR}/metrics.json" \
    --mtp_metrics "${PHASE3_MTP_EVAL_DIR}/metrics.json" \
    --output_dir "${PHASE3_SPEED_COMPARE_DIR}" \
    2>&1 | tee "${LOG_DIR}/${STAMP}_speed_compare_phase3.log"
fi

if [[ -n "${REPO_ID}" ]]; then
  echo "=== Push to Hugging Face ==="
  PUSH_FOLDERS=(
    --folder "${PHASE1_DIR}"
    --folder "${PHASE2_DIR}"
    --folder "${PHASE3_DIR}"
  )
  if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
    PUSH_FOLDERS+=(
      --folder "${PHASE1_EVAL_DIR}"
      --folder "${PHASE2_EVAL_DIR}"
      --folder "${PHASE3_EVAL_DIR}"
      --folder "${PHASE3_MTP_EVAL_DIR}"
      --folder "${PHASE3_SPEED_COMPARE_DIR}"
    )
  fi
  PUSH_FOLDERS+=(--folder "${LOG_DIR}")

  "${PYTHON_BIN}" llm2seq_final/scripts/push_to_hf.py \
    --repo_id "${REPO_ID}" \
    "${PUSH_FOLDERS[@]}" \
    --path_in_repo_prefix "${STAMP}" \
    --commit_message "Upload H200 LLM2Seq run ${STAMP}"
else
  echo "HF_REPO_ID is empty; skipping Hugging Face upload."
fi
