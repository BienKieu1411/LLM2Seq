#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 1 queue:
#   1. T5Gemma2 1B-1B on PubMed
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

echo "=== GPU 1 queue started on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Log: ${LOG_FILE} ==="

echo "=== [GPU 1 / 1 of 2] Train T5Gemma2 1B-1B on PubMed ==="
PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR}" \
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUTS}" \
bash run.sh t5gemma-pubmed-1b

echo "=== [GPU 1 / 2 of 2] Train LLM2Seq-v2 on PubMed ==="
if [[ "${OVERWRITE_OUTPUTS,,}" =~ ^(true|1|yes)$ ]]; then
  PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR}" \
  bash llm2seq_v2/run.sh pubmed --overwrite-output-dir
else
  PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR}" \
  bash llm2seq_v2/run.sh pubmed
fi

echo "=== GPU 1 queue completed successfully ==="
