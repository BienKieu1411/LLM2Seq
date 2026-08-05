#!/usr/bin/env bash
set -Eeuo pipefail

# Load EviSeq PubMed epoch_004, train one additional full-finetune epoch,
# then evaluate the resulting last.pt on the test split with batch size 96.
# The original run directory is never overwritten.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
RUNNER="${PROJECT_ROOT}/scripts/run.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -z "${PYTHON_BIN}" && -x "/workspace/storage-shared/nlp/dungdx4/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="/workspace/storage-shared/nlp/dungdx4/bienkieu_env/bin/python"
elif [[ -z "${PYTHON_BIN}" && -x "/Users/kieugiangbien/bienkieu_env/bin/python" ]]; then
  PYTHON_BIN="/Users/kieugiangbien/bienkieu_env/bin/python"
elif [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi
export PYTHON_BIN

CONFIG="${PROJECT_ROOT}/configs/tasks/pubmed_continue_epoch5.yaml"
CHECKPOINT="${EVISEQ_CHECKPOINT:-${PROJECT_ROOT}/runs/eviseq/pubmed_qwen3_evidence/epoch_004.pt}"
OUTPUT_DIR="${EVISEQ_CONTINUE_OUTPUT_DIR:-${PROJECT_ROOT}/runs/eviseq/pubmed_continue_epoch5}"
EVAL_BATCH_SIZE="${EVISEQ_EVAL_BATCH_SIZE:-96}"
PREDICTIONS="${OUTPUT_DIR}/epoch5_test_predictions.jsonl"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi

echo "=== EviSeq continuation: epoch_004 -> one full epoch ==="
echo "checkpoint: ${CHECKPOINT}"
echo "output:     ${OUTPUT_DIR}"
echo "eval batch: ${EVAL_BATCH_SIZE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  bash "${RUNNER}" train "${CONFIG}" \
    --init-checkpoint "${CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --overwrite-output-dir

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  bash "${RUNNER}" evaluate \
    "${OUTPUT_DIR}/resolved_config.yaml" \
    "${OUTPUT_DIR}/last.pt" \
    "${PREDICTIONS}" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}"

METRICS="${OUTPUT_DIR}/epoch5_test_predictions.metrics.json"
if [[ -f "${METRICS}" ]]; then
  echo "=== Evaluation metrics ==="
  "${PYTHON_BIN}" -m json.tool "${METRICS}"
fi
