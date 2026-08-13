#!/usr/bin/env bash
set -Eeuo pipefail

# Run a corrected PubMed baseline, then two isolated architecture variants:
# identity-initialized PPLX->Qwen bridge projection, then the same bridge
# with cross-gate 0.20.  Each run evaluates its own last.pt with Perl
# ROUGE-1.5.5.  The corrected baseline has a separate output directory, so
# historical legacy artifacts are retained for reference.
#
# Run from the repository src directory:
#   CUDA_VISIBLE_DEVICES=0 bash eviseq_new/scripts/gpu_0.sh

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
PROCESSED_DATA_DIR="datasets/pubmed"

if [[ ! -f "${PYROUGE_HOME_DIR}/ROUGE-1.5.5.pl" ]]; then
  echo "ERROR: ROUGE-1.5.5.pl not found under PYROUGE_HOME_DIR=${PYROUGE_HOME_DIR}" >&2
  echo "Export the correct PYROUGE_HOME_DIR before running this queue." >&2
  exit 1
fi

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_0_pubmed_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== GPU 0: corrected PPLX baseline -> bridge projection -> gate 0.20 ==="
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

run_and_score() {
  local train_mode="$1"
  local eval_mode="$2"
  local run_dir="$3"
  local predictions="${run_dir}/last_test_predictions.jsonl"

  echo "=== Starting ${train_mode} ==="
  bash scripts/run.sh "${train_mode}" --overwrite-output-dir
  bash scripts/run.sh "${eval_mode}" --batch-size "${EVAL_BATCH_SIZE}"
  bash scripts/run.sh rouge155 "${PROJECT_ROOT}/${predictions}" --details
}

run_and_score \
  pplx-pubmed-aligned-corrected \
  evaluate-pplx-pubmed-aligned-corrected-test \
  runs/eviseq/pubmed_pplx_aligned_corrected_0_6b

run_and_score \
  pplx-pubmed-aligned-bridgeproj \
  evaluate-pplx-pubmed-aligned-bridgeproj-test \
  runs/eviseq/pubmed_pplx_aligned_bridgeproj_0_6b

run_and_score \
  pplx-pubmed-aligned-bridgeproj-gate20 \
  evaluate-pplx-pubmed-aligned-bridgeproj-gate20-test \
  runs/eviseq/pubmed_pplx_aligned_bridgeproj_gate20_0_6b

echo "=== GPU 0 PubMed queue completed successfully ==="
