#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 0 queue:
#   1. EviSeq on PubMed
#   2. EviSeq on WikiLingua
#
# Run from anywhere:
#   CUDA_VISIBLE_DEVICES=0 bash gpu_0.sh
# Intentional rerun:
#   OVERWRITE_OUTPUTS=true CUDA_VISIBLE_DEVICES=0 bash gpu_0.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-false}"

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_0_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

overwrite_args=()
if [[ "${OVERWRITE_OUTPUTS,,}" =~ ^(true|1|yes)$ ]]; then
  overwrite_args+=(--overwrite-output-dir)
fi

ensure_pubmed() {
  local data_dir="eviseq/data/pubmed"
  if [[ -s "${data_dir}/train.jsonl" \
    && -s "${data_dir}/validation.jsonl" \
    && -s "${data_dir}/test.jsonl" ]]; then
    echo "=== EviSeq PubMed data already prepared; skipping copy ==="
    return
  fi
  echo "=== Prepare EviSeq PubMed from ${PUBMED_SOURCE_DIR} ==="
  bash eviseq/run.sh prepare-pubmed "${PUBMED_SOURCE_DIR}"
}

run_rouge155_if_available() {
  local predictions="$1"
  if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
    bash eviseq/run.sh rouge155 "${predictions}" --details
  else
    echo "=== Perl ROUGE-1.5.5 skipped: PYROUGE_HOME_DIR is not set ==="
  fi
}

echo "=== GPU 0 queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Log: ${LOG_FILE} ==="

ensure_pubmed

echo "=== [GPU 0 / 1 of 2] Train EviSeq on PubMed ==="
bash eviseq/run.sh pubmed "${overwrite_args[@]}"
bash eviseq/run.sh paper-test-pubmed
run_rouge155_if_available \
  "runs/eviseq/pubmed_qwen3_evidence/last_test_predictions.jsonl"

echo "=== [GPU 0 / 2 of 2] Train EviSeq on WikiLingua ==="
bash eviseq/run.sh wiki "${overwrite_args[@]}"
bash eviseq/run.sh paper-test-wiki
run_rouge155_if_available \
  "runs/eviseq/wikilingua_qwen3_evidence/last_test_predictions.jsonl"

echo "=== GPU 0 queue completed successfully ==="
