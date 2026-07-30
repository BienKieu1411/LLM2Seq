#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 0 queue:
#   1. Discard incomplete Qwen PubMed Phase 3 and evaluate Phase-2 last.pt.
#   2. Train PPLX-Embed EviSeq-v2 on WikiLingua (Phase 1-2 only), then test.
#
# Run:
#   CUDA_VISIBLE_DEVICES=0 bash gpu_0.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PUBMED_EVAL_BATCH_SIZE="${PUBMED_EVAL_BATCH_SIZE:-1}"
WIKI_EVAL_BATCH_SIZE="${WIKI_EVAL_BATCH_SIZE:-8}"

RUN_DIR="runs/eviseq_v2/pubmed_qwen3_evidence"
CONFIG="${RUN_DIR}/resolved_config.yaml"
PHASE2_CHECKPOINT="${RUN_DIR}/last.pt"
RANKING_DIR="${RUN_DIR}/ranking"

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
[[ -f "${PHASE2_CHECKPOINT}" ]] || {
  echo "ERROR: missing Phase-2 checkpoint ${PHASE2_CHECKPOINT}" >&2
  exit 1
}

# A killed Phase 3 leaves PHASE2_COMPLETE and RUNNING. Restore the committed
# Phase-2 state so the standard evaluator selects root/last.pt.
rm -f "${RUN_DIR}/RUNNING"
if [[ -f "${RUN_DIR}/PHASE2_COMPLETE" ]]; then
  mv -f "${RUN_DIR}/PHASE2_COMPLETE" "${RUN_DIR}/COMPLETE"
fi
[[ -f "${RUN_DIR}/COMPLETE" ]] || {
  echo "ERROR: neither ${RUN_DIR}/COMPLETE nor PHASE2_COMPLETE exists." >&2
  exit 1
}

# Delete only incomplete Phase-3 artifacts, never Phase-2 last.pt.
rm -rf "${RANKING_DIR}"

echo "=== Evaluate Qwen PubMed Phase-2 last.pt on the test split ==="
EVISEQ_EVAL_BATCH_SIZE="${PUBMED_EVAL_BATCH_SIZE}" \
  bash eviseq_v2/run.sh paper-test-pubmed

PUBMED_PREDICTIONS="${RUN_DIR}/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 "${PUBMED_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for Qwen PubMed: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== Train PPLX-Embed EviSeq-v2 on WikiLingua (Phase 1-2 only) ==="
bash eviseq_v2/run.sh pplx

echo "=== Evaluate PPLX WikiLingua Phase-2 last.pt on the test split ==="
EVISEQ_EVAL_BATCH_SIZE="${WIKI_EVAL_BATCH_SIZE}" \
  bash eviseq_v2/run.sh paper-test \
  eviseq_v2/configs/encoders/pplx_0_6b.yaml

WIKI_PREDICTIONS="runs/eviseq_v2/encoders/pplx_0_6b/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 "${WIKI_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for PPLX WikiLingua: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== GPU 0 queue completed successfully ==="
