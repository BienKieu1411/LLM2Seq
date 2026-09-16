#!/usr/bin/env bash
set -euo pipefail

CALLER_CWD="$(pwd)"
REQUESTED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-}"
REQUESTED_OVERWRITE="${OVERWRITE_OUTPUT_DIR:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${PROJECT_ROOT}"

MODE="${1:-all}"
if [[ -n "${REQUESTED_SOURCE_DIR}" ]]; then
  PUBMED_SOURCE_DIR="${REQUESTED_SOURCE_DIR}"
fi
if [[ -n "${REQUESTED_OVERWRITE}" ]]; then
  OVERWRITE_OUTPUT_DIR="${REQUESTED_OVERWRITE}"
fi
# T5Gemma now reads the canonical EviSeq files directly.  Keep all paths
# relative to PROJECT_ROOT after load_env.sh has changed into that directory.
PUBMED_DATA_DIR="${PUBMED_DATA_DIR:-eviseq_new/datasets/pubmed}"
PUBMED_RAW_DIR="${PUBMED_RAW_DIR:-eviseq_new/datasets/raw/pubmed}"
PUBMED_LOG_DIR="${PUBMED_LOG_DIR:-T5Gemma/logs/pubmed}"
mkdir -p "${PUBMED_LOG_DIR}"

pubmed_data_ready() {
  [[ -s "${PUBMED_DATA_DIR}/train.jsonl" \
    && -s "${PUBMED_DATA_DIR}/validation.jsonl" \
    && -s "${PUBMED_DATA_DIR}/test.jsonl" ]]
}

prepare_pubmed() {
  if [[ -z "${PUBMED_SOURCE_DIR:-}" ]]; then
    echo "ERROR: set PUBMED_SOURCE_DIR to the folder containing train.label.jsonl, val.label.jsonl, and test.label.jsonl." >&2
    exit 2
  fi
  if [[ "${PUBMED_SOURCE_DIR}" != /* ]]; then
    PUBMED_SOURCE_DIR="${CALLER_CWD}/${PUBMED_SOURCE_DIR}"
  fi
  echo "=== Prepare canonical EviSeq PubMed data ==="
  echo "Source: ${PUBMED_SOURCE_DIR}"
  echo "Processed: ${PUBMED_DATA_DIR}"
  echo "Raw copy: ${PUBMED_RAW_DIR}"
  prepare_args=(
    --dataset pubmed
    --input-dir "${PUBMED_SOURCE_DIR}"
    --output-dir "${PUBMED_DATA_DIR}"
    --raw-copy-dir "${PUBMED_RAW_DIR}"
  )
  if [[ "${ALLOW_CROSS_SPLIT_CONTENT:-false}" =~ ^(true|1|yes)$ ]]; then
    prepare_args+=(--allow-cross-split-content)
  fi
  bash "${PROJECT_ROOT}/eviseq_new/scripts/prepare_afmr.sh" "${prepare_args[@]}"
}

if [[ "${FORCE_PREPARE_PUBMED:-false}" =~ ^(true|1|yes)$ ]]; then
  prepare_pubmed
elif pubmed_data_ready; then
  echo "PubMed processed data already exists; skipping copy/conversion."
else
  prepare_pubmed
fi

run_one() {
  local scale="$1"
  local config="$2"
  local run_dir="$3"
  local eval_dir="$4"
  local train_log
  local eval_log

  train_log="${PUBMED_LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${scale}_train.log"
  echo "=== Full fine-tune T5Gemma ${scale} on PubMed (4096 source tokens) ==="
  echo "Config: ${config}"
  echo "Log: ${train_log}"
  # Route training through train.sh so CUDA_VISIBLE_DEVICES=0,1 launches
  # torchrun/DDP.  The config is passed explicitly because this wrapper runs
  # two different PubMed recipes in one invocation.
  bash "${T5GEMMA_ROOT}/scripts/train.sh" \
    --config "${config}" \
    2>&1 | tee "${train_log}"

  eval_log="${PUBMED_LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${scale}_eval.log"
  echo "=== Evaluate T5Gemma ${scale} on PubMed test ==="
  "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
    --config "${config}" \
    --checkpoint "${run_dir}/final_model" \
    --test_file "${PUBMED_DATA_DIR}/test.jsonl" \
    --output_dir "${eval_dir}" \
    --limit "${EVAL_LIMIT}" \
    2>&1 | tee "${eval_log}"

  if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
    "${PYTHON_BIN}" "${PROJECT_ROOT}/rouge155/evaluate_rouge.py" \
      "${eval_dir}/predictions.jsonl" \
      --output "${eval_dir}/predictions.rouge155.json"
  else
    echo "Perl ROUGE skipped: export PYROUGE_HOME_DIR to calculate the paper score automatically." >&2
  fi
}

run_1b() {
  run_one \
    "1B-1B" \
    "T5Gemma/configs/pubmed_full_1b_1b_4096.yaml" \
    "runs/t5gemma2_1b_1b_full_pubmed_4096" \
    "T5Gemma/eval_outputs/pubmed/1b_1b"
}

run_4b() {
  run_one \
    "4B-4B" \
    "T5Gemma/configs/pubmed_full_4b_4b_4096.yaml" \
    "runs/t5gemma2_4b_4b_full_pubmed_4096" \
    "T5Gemma/eval_outputs/pubmed/4b_4b"
}

case "${MODE}" in
  1b)
    run_1b
    ;;
  4b)
    run_4b
    ;;
  all)
    run_1b
    run_4b
    ;;
  *)
    echo "Usage: PUBMED_SOURCE_DIR=/path/to/pubmed bash run_pubmed_pipeline.sh {1b|4b|all}" >&2
    echo "Canonical data is read from ${PUBMED_DATA_DIR}; set FORCE_PREPARE_PUBMED=true to prepare it again." >&2
    exit 2
    ;;
esac
