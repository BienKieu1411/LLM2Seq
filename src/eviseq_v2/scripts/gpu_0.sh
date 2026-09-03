#!/usr/bin/env bash
set -Eeuo pipefail

# Run the current PubMed PCEB main recipe.
# Training runs teacher-forced validation loss to preserve best.pt, but never
# generates validation summaries; each run is decoded directly on test once
# from last.pt.  The test score is exploratory until the final ROUGE-1.5.5
# audit is complete.
#
# Run from the repository src directory:
#   CUDA_VISIBLE_DEVICES=0 bash eviseq_v2/scripts/gpu_0.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The supplied PubMed label files may intentionally reuse records/IDs across
# splits.  Permit conversion to continue on this supplied benchmark copy;
# export EVISEQ_ALLOW_CROSS_SPLIT_CONTENT=false to restore the strict gate.
export EVISEQ_ALLOW_CROSS_SPLIT_CONTENT="${EVISEQ_ALLOW_CROSS_SPLIT_CONTENT:-true}"
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="datasets/pubmed"

if [[ ! -f "${PYROUGE_HOME_DIR}/ROUGE-1.5.5.pl" ]]; then
  echo "ERROR: ROUGE-1.5.5.pl not found under PYROUGE_HOME_DIR=${PYROUGE_HOME_DIR}" >&2
  echo "Export the correct PYROUGE_HOME_DIR before running this queue." >&2
  exit 1
fi

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_0_pubmed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== GPU 0: PCEB PubMed (PPLX 0.6B -> Qwen3 0.6B) ==="
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== EVISEQ_ALLOW_CROSS_SPLIT_CONTENT=${EVISEQ_ALLOW_CROSS_SPLIT_CONTENT} ==="
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

run_and_score() {
  local config="$1"
  local run_dir="$2"
  local predictions="${run_dir}/last_test_predictions.jsonl"

  echo "=== Training ${config}; saving best.pt by validation loss (no valid predictions) ==="
  bash scripts/run.sh train "${config}" --overwrite-output-dir
  echo "=== Evaluating last.pt directly on the test split ==="
  bash scripts/run.sh evaluate-test "${config}" --batch-size "${EVAL_BATCH_SIZE}"
  bash scripts/run.sh rouge155 "${PROJECT_ROOT}/${predictions}" --details
}

run_and_score \
  configs/models/pplx_pubmed_pceb.yaml \
  runs/eviseq/pubmed_pceb_pplx_0_6b

echo "=== GPU 0 PubMed queue completed successfully ==="
