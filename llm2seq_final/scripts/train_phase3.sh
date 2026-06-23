#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

PHASE2_DIR="${PHASE2_DIR:-runs/h200_llm2seq_phase2_lora_encoder}"
PHASE2_CKPT="${1:-${PHASE2_DIR}/best.pt}"
CONFIG="${2:-${PHASE3_CONFIG:-llm2seq_final/configs/phase3_mtp_self_distill_4096.yaml}}"
LOG_DIR="${LOG_DIR:-llm2seq_final/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}_phase3.log"

if [[ ! -f "${PHASE2_CKPT}" ]]; then
  echo "Local Phase 2 checkpoint not found; trying Hugging Face fallback: ${PHASE2_CKPT}" >&2
  PHASE2_CKPT="$(resolve_hf_checkpoint phase2 "${PHASE2_CKPT}")"
fi

if [[ ! -f "${PHASE2_CKPT}" ]]; then
  echo "Phase 2 checkpoint not found: ${PHASE2_CKPT}" >&2
  echo "Usage: bash llm2seq_final/scripts/train_phase3.sh /path/to/phase2/best.pt [config]" >&2
  exit 1
fi

echo "=== H200 Phase 3: MTP-D self-distillation ==="
echo "Config: ${CONFIG}"
echo "Resume weights from: ${PHASE2_CKPT}"
echo "Log: ${LOG_FILE}"
"${PYTHON_BIN}" -m llm2seq.src.training.trainer --config "${CONFIG}" --resume "${PHASE2_CKPT}" 2>&1 | tee "${LOG_FILE}"
