#!/usr/bin/env bash
set -Eeuo pipefail

# Train and evaluate an isolated Qwen3-0.6B encoder -> Qwen3-0.6B decoder
# EviSeq run. Existing runs are protected unless OVERWRITE_OUTPUT_DIR=true.
# Run from any directory:
#   CUDA_VISIBLE_DEVICES=0 bash eviseq/scripts/run_qwen3_to_qwen3_pubmed.sh

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
cd "${SRC_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-eviseq/configs/models/qwen3_to_qwen3_pubmed.yaml}"
RUN_DIR="${RUN_DIR:-eviseq/runs/eviseq/pubmed_qwen3_to_qwen3}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

if [[ -e "${RUN_DIR}" && "${OVERWRITE_OUTPUT_DIR}" != "true" ]]; then
  echo "ERROR: run directory already exists: ${RUN_DIR}" >&2
  echo "Use another RUN_DIR or set OVERWRITE_OUTPUT_DIR=true intentionally." >&2
  exit 1
fi

TRAIN_ARGS=()
if [[ "${OVERWRITE_OUTPUT_DIR}" == "true" ]]; then
  TRAIN_ARGS+=(--overwrite-output-dir)
fi

echo "=== EviSeq Qwen3-0.6B encoder -> Qwen3-0.6B decoder (PubMed) ==="
echo "Config: ${CONFIG}"
echo "Run directory: ${RUN_DIR}"
echo "Evaluation batch size: ${EVAL_BATCH_SIZE}"

bash eviseq/scripts/run.sh train \
  "${CONFIG}" \
  --output-dir "${RUN_DIR}" \
  "${TRAIN_ARGS[@]}"

if [[ ! -f "${RUN_DIR}/last.pt" || ! -f "${RUN_DIR}/resolved_config.yaml" ]]; then
  echo "ERROR: training did not produce last.pt and resolved_config.yaml" >&2
  exit 1
fi

bash eviseq/scripts/run.sh evaluate \
  "${RUN_DIR}/resolved_config.yaml" \
  "${RUN_DIR}/last.pt" \
  "${RUN_DIR}/last_test_predictions.jsonl" \
  --split test \
  --batch-size "${EVAL_BATCH_SIZE}"

echo "=== Qwen3 -> Qwen3 PubMed run completed ==="
