#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

PHASE1_DIR="${PHASE1_DIR:-runs/phase1_warmup}"
PHASE1_CKPT="${1:-${PHASE1_DIR}/best.pt}"
CONFIG="${2:-${PHASE2_CONFIG:-llm2seq/configs/wikilingua_phase2.yaml}}"
LOG_DIR="${LOG_DIR:-llm2seq/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}_phase2.log"

if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "Local Phase 1 checkpoint not found; trying Hugging Face fallback: ${PHASE1_CKPT}" >&2
  PHASE1_CKPT="$(resolve_hf_checkpoint phase1 "${PHASE1_CKPT}")"
fi

if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "Phase 1 checkpoint not found: ${PHASE1_CKPT}" >&2
  echo "Usage: bash llm2seq/scripts/train_phase2.sh /path/to/phase1/best.pt [config]" >&2
  exit 1
fi

echo "=== LLM2Seq Phase 2: LoRA encoder adaptation ==="
echo "Config: ${CONFIG}"
echo "Resume weights from: ${PHASE1_CKPT}"
echo "Log: ${LOG_FILE}"
"${PYTHON_BIN}" -m src.training.trainer --config "${CONFIG}" --resume "${PHASE1_CKPT}" 2>&1 | tee "${LOG_FILE}"
