#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 1 queue:
#   1. Evaluate the completed Phase-2 PPLX-Embed PubMed checkpoint.
#   2. Train EviSeq-v2 Qwen3-Embedding on WikiLingua (Phase 1-2 only).
#
# The incomplete PubMed Phase 3 is intentionally discarded. Phase-2 last.pt
# is preserved. Do NOT pass --overwrite-output-dir.
# Run:
#   CUDA_VISIBLE_DEVICES=1 bash gpu_1.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/eviseq_v2${PYTHONPATH:+:${PYTHONPATH}}"

PUBMED_EVAL_BATCH_SIZE="${PUBMED_EVAL_BATCH_SIZE:-64}"
WIKI_EVAL_BATCH_SIZE="${WIKI_EVAL_BATCH_SIZE:-8}"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV}/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

RUN_DIR="runs/eviseq_v2/pubmed_pplx_0_6b"
CONFIG="${RUN_DIR}/resolved_config.yaml"
PHASE2_CHECKPOINT="${RUN_DIR}/last.pt"
RANKING_DIR="${RUN_DIR}/ranking"

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
[[ -f "${PHASE2_CHECKPOINT}" ]] || {
  echo "ERROR: missing Phase-2 checkpoint ${PHASE2_CHECKPOINT}" >&2
  exit 1
}

# A killed/failed Phase 3 leaves PHASE2_COMPLETE and RUNNING behind. Restore
# the committed Phase-2 state so the standard evaluator selects root/last.pt.
rm -f "${RUN_DIR}/RUNNING"
if [[ -f "${RUN_DIR}/PHASE2_COMPLETE" ]]; then
  mv -f "${RUN_DIR}/PHASE2_COMPLETE" "${RUN_DIR}/COMPLETE"
fi
[[ -f "${RUN_DIR}/COMPLETE" ]] || {
  echo "ERROR: neither ${RUN_DIR}/COMPLETE nor PHASE2_COMPLETE exists." >&2
  exit 1
}

# Delete only incomplete/generated ranking artifacts, never Phase-2 last.pt.
rm -rf "${RANKING_DIR}"

echo "=== Evaluate PPLX PubMed Phase-2 last.pt on the test split ==="
EVISEQ_EVAL_BATCH_SIZE="${PUBMED_EVAL_BATCH_SIZE}" \
  bash eviseq_v2/run.sh paper-test-pplx-pubmed

PREDICTIONS="${RUN_DIR}/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 "${PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== Train Qwen3-Embedding EviSeq-v2 on WikiLingua (Phase 1-2 only) ==="
bash eviseq_v2/run.sh wiki

echo "=== Evaluate Qwen3-Embedding WikiLingua Phase-2 last.pt on the test split ==="
EVISEQ_EVAL_BATCH_SIZE="${WIKI_EVAL_BATCH_SIZE}" \
  bash eviseq_v2/run.sh paper-test-wiki

WIKI_PREDICTIONS="runs/eviseq_v2/wikilingua_qwen3_evidence/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 "${WIKI_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for Qwen3-Embedding WikiLingua: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== GPU 1 queue completed successfully ==="
