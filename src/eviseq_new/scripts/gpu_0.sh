#!/usr/bin/env bash
set -Eeuo pipefail

# Train the paper EviSeq recipe (PPLX encoder, sentence-aligned evidence CL)
# on PubMed, evaluate last.pt on the test split, then compute Perl ROUGE-1.5.5.
#
# Run from the repository src directory:
#   CUDA_VISIBLE_DEVICES=0 bash eviseq/scripts/gpu_0.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
RUN_DIR="runs/eviseq/pubmed_pplx_aligned_0_6b"
LAST_PREDICTIONS="${RUN_DIR}/last_test_predictions.jsonl"
BEST_PREDICTIONS="${RUN_DIR}/best_test_predictions.jsonl"
PROCESSED_DATA_DIR="datasets/pubmed"

if [[ ! -f "${PYROUGE_HOME_DIR}/ROUGE-1.5.5.pl" ]]; then
  echo "ERROR: ROUGE-1.5.5.pl not found under PYROUGE_HOME_DIR=${PYROUGE_HOME_DIR}" >&2
  echo "Export the correct PYROUGE_HOME_DIR before running this queue." >&2
  exit 1
fi

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_0_pubmed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== GPU 0: EviSeq PPLX-aligned PubMed train, test, and ROUGE-1.5.5 ==="
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== PYROUGE_HOME_DIR=${PYROUGE_HOME_DIR} ==="
echo "=== Log: ${LOG_FILE} ==="

if [[ -f "${PROCESSED_DATA_DIR}/train.jsonl" \
   && -f "${PROCESSED_DATA_DIR}/validation.jsonl" \
   && -f "${PROCESSED_DATA_DIR}/test.jsonl" ]]; then
  echo "=== PubMed data already prepared; skipping copy/conversion ==="
else
  echo "=== Preparing PubMed from ${PUBMED_SOURCE_DIR} ==="
  for required in train.label.jsonl val.label.jsonl test.label.jsonl; do
    if [[ ! -f "${PUBMED_SOURCE_DIR}/${required}" ]]; then
      echo "ERROR: missing ${PUBMED_SOURCE_DIR}/${required}" >&2
      exit 1
    fi
  done
  bash scripts/run.sh prepare-pubmed "${PUBMED_SOURCE_DIR}"
fi

bash scripts/run.sh pplx-pubmed-aligned --overwrite-output-dir

bash scripts/run.sh evaluate-pplx-pubmed-aligned-test \
  --batch-size "${EVAL_BATCH_SIZE}"

bash scripts/run.sh rouge155 \
  "${PROJECT_ROOT}/${LAST_PREDICTIONS}" \
  --details

if [[ -f "${RUN_DIR}/best.pt" ]]; then
  bash scripts/run.sh evaluate \
    "${RUN_DIR}/resolved_config.yaml" \
    "${RUN_DIR}/best.pt" \
    "${BEST_PREDICTIONS}" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}"

  bash scripts/run.sh rouge155 \
    "${PROJECT_ROOT}/${BEST_PREDICTIONS}" \
    --details
else
  echo "=== best.pt is disabled or absent; skipping best-checkpoint evaluation ==="
fi

echo "=== GPU 0 PubMed queue completed successfully ==="
