#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

PHASE_NAME="${1:?Usage: bash llm2seq/scripts/evaluate_phase.sh PHASE_NAME CONFIG CHECKPOINT OUTPUT_DIR [DECODE_MODE] [BASE_CHECKPOINT]}"
CONFIG="${2:?Usage: bash llm2seq/scripts/evaluate_phase.sh PHASE_NAME CONFIG CHECKPOINT OUTPUT_DIR [DECODE_MODE] [BASE_CHECKPOINT]}"
CHECKPOINT="${3:?Usage: bash llm2seq/scripts/evaluate_phase.sh PHASE_NAME CONFIG CHECKPOINT OUTPUT_DIR [DECODE_MODE] [BASE_CHECKPOINT]}"
OUTPUT_DIR="${4:?Usage: bash llm2seq/scripts/evaluate_phase.sh PHASE_NAME CONFIG CHECKPOINT OUTPUT_DIR [DECODE_MODE] [BASE_CHECKPOINT]}"
DECODE_MODE="${5:-autoregressive}"
BASE_CHECKPOINT="${6:-}"

TEST_FILE="${TEST_FILE:-${DATA_DIR:-llm2seq/data/processed}/test.jsonl}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
LOG_DIR="${LOG_DIR:-llm2seq/logs}"
mkdir -p "${LOG_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}_eval_${PHASE_NAME}_${DECODE_MODE}.log"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Local checkpoint not found; trying Hugging Face fallback: ${CHECKPOINT}" >&2
  CHECKPOINT="$(resolve_hf_checkpoint_from_config "${CONFIG}" "${CHECKPOINT}")"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "Test file not found: ${TEST_FILE}" >&2
  exit 1
fi

echo "=== Evaluate ${PHASE_NAME} (${DECODE_MODE}) ==="
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
if [[ -n "${BASE_CHECKPOINT}" ]]; then
  if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Local base checkpoint not found; trying Hugging Face fallback: ${BASE_CHECKPOINT}" >&2
    BASE_PHASE="$(BASE_CHECKPOINT_VALUE="${BASE_CHECKPOINT}" "${PYTHON_BIN}" - <<'PY'
import os
path = os.environ["BASE_CHECKPOINT_VALUE"].lower()
if "phase1" in path or "warmup" in path:
    print("phase1")
elif "phase2" in path or "lora_encoder" in path:
    print("phase2")
elif "phase3" in path or "mtp_self_distill" in path:
    print("phase3")
else:
    print("phase2")
PY
)"
    BASE_CHECKPOINT="$(resolve_hf_checkpoint "${BASE_PHASE}" "${BASE_CHECKPOINT}")"
  fi
  if [[ ! -f "${BASE_CHECKPOINT}" ]]; then
    echo "Base checkpoint not found: ${BASE_CHECKPOINT}" >&2
    exit 1
  fi
  echo "Base checkpoint: ${BASE_CHECKPOINT}"
fi
echo "Test file: ${TEST_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Limit: ${EVAL_LIMIT}"
echo "Log: ${LOG_FILE}"

EXTRA_ARGS=()
if [[ -n "${BASE_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--base_checkpoint "${BASE_CHECKPOINT}")
fi

"${PYTHON_BIN}" llm2seq/scripts/evaluate_full_test.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --decode_mode "${DECODE_MODE}" \
  --batch_size 64 \
  --limit "${EVAL_LIMIT}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_FILE}"
