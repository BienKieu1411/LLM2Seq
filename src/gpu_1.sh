#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 1 queue:
#   1. T5Gemma2 1B-1B on PubMed
#   2. EviSeq on CNN/DailyMail
#
# Run from anywhere:
#   CUDA_VISIBLE_DEVICES=1 bash gpu_1.sh
# Intentional rerun:
#   OVERWRITE_OUTPUTS=true CUDA_VISIBLE_DEVICES=1 bash gpu_1.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
CNNDM_SOURCE_DIR="${CNNDM_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/cnndm}"
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-false}"

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

overwrite_args=()
if [[ "${OVERWRITE_OUTPUTS,,}" =~ ^(true|1|yes)$ ]]; then
  overwrite_args+=(--overwrite-output-dir)
fi

ensure_cnndm() {
  local data_dir="eviseq/data/cnndm"
  if [[ -s "${data_dir}/train.jsonl" \
    && -s "${data_dir}/validation.jsonl" \
    && -s "${data_dir}/test.jsonl" ]]; then
    echo "=== EviSeq CNN/DailyMail data already prepared; skipping copy ==="
    return
  fi
  echo "=== Prepare EviSeq CNN/DailyMail from ${CNNDM_SOURCE_DIR} ==="
  bash eviseq/run.sh prepare-cnndm "${CNNDM_SOURCE_DIR}"
}

run_rouge155_if_available() {
  local predictions="$1"
  if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
    bash eviseq/run.sh rouge155 "${predictions}" --details
  else
    echo "=== Perl ROUGE-1.5.5 skipped: PYROUGE_HOME_DIR is not set ==="
  fi
}

echo "=== GPU 1 queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Log: ${LOG_FILE} ==="

echo "=== [GPU 1 / 1 of 2] Train T5Gemma2 1B-1B on PubMed ==="
PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR}" \
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUTS}" \
bash run.sh t5gemma-pubmed-1b

ensure_cnndm

echo "=== [GPU 1 / 2 of 2] Train EviSeq on CNN/DailyMail ==="
bash eviseq/run.sh cnndm "${overwrite_args[@]}"
bash eviseq/run.sh paper-test-cnndm
run_rouge155_if_available \
  "runs/eviseq/cnndm_qwen3_evidence/last_test_predictions.jsonl"

echo "=== GPU 1 queue completed successfully ==="
