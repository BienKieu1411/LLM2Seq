#!/usr/bin/env bash
set -Eeuo pipefail

# GPU 1 queue:
#   1. Resume Phase 3 for EviSeq-v2 PPLX-Embed on PubMed.
#   2. Train EviSeq-v2 PPLX-Embed on WikiLingua, including Phase 3 ranking.
#
# Phase-2 last.pt is preserved. Do NOT pass --overwrite-output-dir.
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
LOG_FILE="logs/gpu_queues/gpu_1_phase3_pplx_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=== GPU 1: resume EviSeq-v2 PPLX PubMed Phase 3 ==="
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

if [[ -f "${RANKING_DIR}/COMPLETE" && -f "${RANKING_DIR}/last.pt" ]]; then
  echo "=== Phase 3 is already complete; skipping ranking training ==="
else
  # A failed Phase 3 leaves PHASE2_COMPLETE and RUNNING behind. Restore the
  # Phase-2 completion marker expected by the atomic Phase-3 pipeline.
  rm -f "${RUN_DIR}/RUNNING"
  if [[ -f "${RUN_DIR}/PHASE2_COMPLETE" ]]; then
    mv -f "${RUN_DIR}/PHASE2_COMPLETE" "${RUN_DIR}/COMPLETE"
  fi
  [[ -f "${RUN_DIR}/COMPLETE" ]] || {
    echo "ERROR: neither ${RUN_DIR}/COMPLETE nor PHASE2_COMPLETE exists." >&2
    exit 1
  }

  # Remove only an incomplete Phase-3 directory. Never touch Phase-2 last.pt.
  rm -rf "${RANKING_DIR}"

  RUN_DIR="${RUN_DIR}" "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

import eviseq_v2.training as training

run_dir = Path(os.environ["RUN_DIR"])
config_path = run_dir / "resolved_config.yaml"
phase2_checkpoint = run_dir / "last.pt"

assert config_path.is_file(), config_path
assert phase2_checkpoint.is_file(), phase2_checkpoint

# Skip Phase 1-2 and hand the existing canonical checkpoint to the unchanged
# Phase-3 implementation.
training.stable.train = lambda config, overwrite: phase2_checkpoint

ranked_checkpoint = training.train(
    str(config_path),
    overwrite_output_dir=False,
)
print(f"PHASE 3 COMPLETE: {ranked_checkpoint}", flush=True)
PY
fi

echo "=== Evaluate ranked PPLX PubMed checkpoint on the test split ==="
bash eviseq_v2/run.sh paper-test-pplx-pubmed

PREDICTIONS="${RANKING_DIR}/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 "${PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== Train PPLX-Embed EviSeq-v2 on WikiLingua, including Phase 3 ==="
bash eviseq_v2/run.sh pplx

echo "=== Evaluate ranked PPLX WikiLingua checkpoint on the test split ==="
bash eviseq_v2/run.sh paper-test \
  eviseq_v2/configs/encoders/pplx_0_6b.yaml

WIKI_PREDICTIONS="runs/eviseq_v2/encoders/pplx_0_6b/ranking/last_test_predictions.jsonl"
if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  bash eviseq_v2/run.sh rouge155 "${WIKI_PREDICTIONS}" --details
else
  echo "Perl ROUGE skipped for PPLX WikiLingua: PYROUGE_HOME_DIR is not set." >&2
fi

echo "=== GPU 1 queue completed successfully ==="
