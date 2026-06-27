#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

PHASE2_DIR="${PHASE2_DIR:-runs/phase2_lora_encoder}"
PHASE2_CKPT="${1:-${PHASE2_DIR}/best.pt}"
CONFIG="${2:-${PHASE3_CONFIG:-llm2seq/configs/wikilingua_phase3.yaml}}"
LOG_DIR="${LOG_DIR:-llm2seq/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}_phase3.log"

if [[ ! -f "${PHASE2_CKPT}" ]]; then
  echo "Local Phase 2 checkpoint not found; trying Hugging Face fallback: ${PHASE2_CKPT}" >&2
  PHASE2_CKPT="$(resolve_hf_checkpoint phase2 "${PHASE2_CKPT}")"
fi

if [[ ! -f "${PHASE2_CKPT}" ]]; then
  echo "Phase 2 checkpoint not found: ${PHASE2_CKPT}" >&2
  echo "Usage: bash llm2seq/scripts/train_phase3.sh /path/to/phase2/best.pt [config]" >&2
  exit 1
fi

echo "=== LLM2Seq Phase 3: MTP-D self-distillation ==="
echo "Config: ${CONFIG}"
echo "Resume weights from: ${PHASE2_CKPT}"
echo "Log: ${LOG_FILE}"
"${PYTHON_BIN}" -m src.training.trainer --config "${CONFIG}" --resume "${PHASE2_CKPT}" 2>&1 | tee "${LOG_FILE}"
