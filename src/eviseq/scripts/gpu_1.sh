#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 1 queue:
#   1. Evaluate the completed PPLX-Embed PubMed checkpoint.
#   2. Train EviSeq Qwen3-Embedding on WikiLingua.
# Run:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/gpu_1.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PUBMED_EVAL_BATCH_SIZE="${PUBMED_EVAL_BATCH_SIZE:-128}"
WIKI_EVAL_BATCH_SIZE="${WIKI_EVAL_BATCH_SIZE:-8}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUN_DIR="runs/eviseq/pubmed_pplx_0_6b"
CONFIG="${RUN_DIR}/resolved_config.yaml"
CHECKPOINT="${RUN_DIR}/last.pt"

mkdir -p logs/gpu_queues
LOG_FILE="logs/gpu_queues/gpu_1_phase2_eval_then_qwen_wiki_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== GPU 1: evaluate PPLX PubMed Phase 2, then train Qwen3-Embedding WikiLingua ==="
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

echo "=== Evaluate PPLX PubMed last.pt on the test split ==="
bash scripts/run.sh evaluate-pplx-pubmed-test \
  --batch-size "${PUBMED_EVAL_BATCH_SIZE}"

PREDICTIONS="${RUN_DIR}/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash scripts/run.sh rouge155 "${PROJECT_ROOT}/${PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== Train Qwen3-Embedding EviSeq on WikiLingua ==="
bash scripts/run.sh wiki

echo "=== Evaluate Qwen3-Embedding WikiLingua last.pt on the test split ==="
bash scripts/run.sh evaluate-wiki-test \
  --batch-size "${WIKI_EVAL_BATCH_SIZE}"

WIKI_PREDICTIONS="runs/eviseq/wikilingua_qwen3_evidence/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash scripts/run.sh rouge155 "${PROJECT_ROOT}/${WIKI_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for Qwen3-Embedding WikiLingua: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== GPU 1 queue completed successfully ==="
