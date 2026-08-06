#!/usr/bin/env bash
set -Eeuo pipefail

# Resume test evaluation for the PubMed checkpoint produced after epoch 5.
# Run from any directory, for example:
#   CUDA_VISIBLE_DEVICES=0 bash eviseq/scripts/evaluate_pubmed_epoch5.sh

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${SRC_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:-eviseq/runs/eviseq/pubmed_continue_epoch5}"
CONFIG="${CONFIG:-${RUN_DIR}/resolved_config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_DIR}/last.pt}"
OUTPUT="${OUTPUT:-${RUN_DIR}/epoch5_test_predictions.jsonl}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
LOG_FILE="${LOG_FILE:-${RUN_DIR}/epoch5_evaluation.log}"

mkdir -p "${RUN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

for required in "${CONFIG}" "${CHECKPOINT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required file not found: ${required}" >&2
    exit 1
  fi
done

echo "=== EviSeq PubMed epoch-5 test evaluation ==="
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT}"
echo "Batch size: ${EVAL_BATCH_SIZE}"
echo "Log: ${LOG_FILE}"
echo "Resume: enabled"

bash eviseq/scripts/run.sh evaluate \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  "${OUTPUT}" \
  --split test \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --resume \
  "$@"

echo "=== EviSeq PubMed epoch-5 evaluation completed ==="
