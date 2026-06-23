#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_env.sh"
cd "${PROJECT_ROOT}"

CONFIG="${1:-${PHASE1_CONFIG:-llm2seq/configs/wikilingua_phase1.yaml}}"
LOG_DIR="${LOG_DIR:-llm2seq/logs}"
mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${STAMP}_phase1.log"

echo "=== LLM2Seq Phase 1: frozen encoder warmup ==="
echo "Config: ${CONFIG}"
echo "Log: ${LOG_FILE}"
"${PYTHON_BIN}" -m src.training.trainer --config "${CONFIG}" 2>&1 | tee "${LOG_FILE}"
