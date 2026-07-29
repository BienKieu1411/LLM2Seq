#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate the already full-fine-tuned T5Gemma2-1B-1B checkpoint on the
# complete PubMed test split. This script never starts training.

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SRC_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

CONFIG="${CONFIG:-T5Gemma/configs/pubmed_full_1b_1b_4096.yaml}"
CHECKPOINT="${CHECKPOINT:-runs/t5gemma2_1b_1b_full_pubmed_4096/final_model}"
TEST_FILE="${TEST_FILE:-T5Gemma/data/processed/pubmed/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-T5Gemma/eval_outputs/pubmed/1b_1b}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: trained checkpoint not found: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${TEST_FILE}" ]]; then
  echo "ERROR: PubMed test file not found: ${TEST_FILE}" >&2
  exit 1
fi

echo "=== Evaluate trained T5Gemma2-1B-1B on PubMed test ==="
echo "GPU visibility: ${CUDA_VISIBLE_DEVICES:-not explicitly set}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Test file: ${TEST_FILE}"
echo "Output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" T5Gemma/scripts/evaluate_full_test.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --limit "${EVAL_LIMIT}" \
  "$@"

PREDICTIONS="${OUTPUT_DIR}/predictions.jsonl"
if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "ERROR: evaluation did not create ${PREDICTIONS}" >&2
  exit 1
fi

if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  echo "=== Calculate Perl ROUGE-1.5.5 ==="
  "${PYTHON_BIN}" rouge155/evaluate_rouge.py \
    "${PREDICTIONS}" \
    --output "${OUTPUT_DIR}/predictions.rouge155.json"
else
  echo "WARNING: PYROUGE_HOME_DIR is not set; Perl ROUGE-1.5.5 was skipped." >&2
fi

echo "=== PubMed evaluation completed ==="
