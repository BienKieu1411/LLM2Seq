#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 0 queue:
#   1. Evaluate Qwen PubMed last.pt.
#   2. Train PPLX-Embed EviSeq on WikiLingua, then test.
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/gpu_0.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PUBMED_EVAL_BATCH_SIZE="${PUBMED_EVAL_BATCH_SIZE:-128}"
WIKI_EVAL_BATCH_SIZE="${WIKI_EVAL_BATCH_SIZE:-8}"

RUN_DIR="runs/eviseq/pubmed_qwen3_evidence"
CONFIG="${RUN_DIR}/resolved_config.yaml"
CHECKPOINT="${RUN_DIR}/last.pt"

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_0_qwen_phase2_eval_then_pplx_wiki_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== GPU 0: evaluate Qwen PubMed Phase 2, then train PPLX WikiLingua ==="
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Log: ${LOG_FILE} ==="

[[ -f "${CONFIG}" ]] || {
  echo "ERROR: missing ${CONFIG}" >&2
  exit 1
}
[[ -f "${CHECKPOINT}" ]] || {
  echo "ERROR: missing checkpoint ${CHECKPOINT}" >&2
  exit 1
}

echo "=== Evaluate Qwen PubMed last.pt on the test split ==="
bash scripts/run.sh evaluate-pubmed-test \
  --batch-size "${PUBMED_EVAL_BATCH_SIZE}"

PUBMED_PREDICTIONS="${RUN_DIR}/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash scripts/run.sh rouge155 "${PROJECT_ROOT}/${PUBMED_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for Qwen PubMed: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== Train PPLX-Embed EviSeq on WikiLingua ==="
bash scripts/run.sh pplx

echo "=== Evaluate PPLX WikiLingua last.pt on the test split ==="
bash scripts/run.sh evaluate-test \
  configs/models/pplx_wikilingua.yaml \
  --batch-size "${WIKI_EVAL_BATCH_SIZE}"

WIKI_PREDICTIONS="runs/eviseq/encoders/pplx_0_6b/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash scripts/run.sh rouge155 "${PROJECT_ROOT}/${WIKI_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for PPLX WikiLingua: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== GPU 0 queue completed successfully ==="
