#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 1 queue:
#   1. EviSeq-v2 PPLX-Embed 0.6B on PubMed
#   2. LLM2Seq-v2 on PubMed
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
OVERWRITE_OUTPUTS="${OVERWRITE_OUTPUTS:-false}"

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_1_$(date +%Y%m%d_%H%M%S).log"
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
    echo "=== EviSeq-v2 PubMed data already prepared; skipping copy ==="
    return
  fi
  echo "=== Prepare EviSeq-v2 PubMed from ${PUBMED_SOURCE_DIR} ==="
  bash eviseq_v2/run.sh prepare-pubmed "${PUBMED_SOURCE_DIR}"
}

echo "=== GPU 1 queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Log: ${LOG_FILE} ==="

ensure_pubmed

echo "=== [GPU 1 / 1 of 2] Train EviSeq-v2 PPLX-Embed 0.6B on PubMed ==="
bash eviseq_v2/run.sh pplx-pubmed "${overwrite_args[@]}"
bash eviseq_v2/run.sh paper-test-pplx-pubmed
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 \
    runs/eviseq_v2/pubmed_pplx_0_6b/ranking/last_test_predictions.jsonl --details
else
  echo "Perl ROUGE skipped for PPLX PubMed: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== [GPU 1 / 2 of 2] Train LLM2Seq-v2 on PubMed ==="
if [[ "${OVERWRITE_OUTPUTS,,}" =~ ^(true|1|yes)$ ]]; then
  PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR}" \
  bash llm2seq_v2/run.sh pubmed --overwrite-output-dir
else
  PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR}" \
  bash llm2seq_v2/run.sh pubmed
fi

echo "=== GPU 1 queue completed successfully ==="
