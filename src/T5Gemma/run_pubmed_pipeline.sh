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
if [[ -z "${PUBMED_SOURCE_DIR:-}" ]]; then
  echo "ERROR: set PUBMED_SOURCE_DIR to the folder containing train.label.jsonl, val.label.jsonl, and test.label.jsonl." >&2
  exit 2
fi
if [[ "${PUBMED_SOURCE_DIR}" != /* ]]; then
  PUBMED_SOURCE_DIR="${CALLER_CWD}/${PUBMED_SOURCE_DIR}"
fi

PUBMED_RAW_DIR="${PUBMED_RAW_DIR:-T5Gemma/datasets/pubmed}"
PUBMED_DATA_DIR="${PUBMED_DATA_DIR:-T5Gemma/data/processed/pubmed}"
PUBMED_LOG_DIR="${PUBMED_LOG_DIR:-T5Gemma/logs/pubmed}"
mkdir -p "${PUBMED_LOG_DIR}"

echo "=== Copy and prepare PubMed ==="
echo "Source: ${PUBMED_SOURCE_DIR}"
echo "Raw copy: ${PUBMED_RAW_DIR}"
"${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/prepare_pubmed_json.py" \
  --input_dir "${PUBMED_SOURCE_DIR}" \
  --raw_copy_dir "${PUBMED_RAW_DIR}" \
  --output_dir "${PUBMED_DATA_DIR}"

run_one() {
  local scale="$1"
  local config="$2"
  local run_dir="$3"
  local eval_dir="$4"
  local -a train_args
  local train_log
  local eval_log

  train_args=(--config "${config}")
  if [[ "${OVERWRITE_OUTPUT_DIR,,}" == "true" || "${OVERWRITE_OUTPUT_DIR}" == "1" || "${OVERWRITE_OUTPUT_DIR,,}" == "yes" ]]; then
    train_args+=(--overwrite-output-dir)
  fi

  train_log="${PUBMED_LOG_DIR}/$(date +%Y%m%d_%H%M%S)_${scale}_train.log"
  echo "=== Full fine-tune T5Gemma ${scale} on PubMed (4096 source tokens) ==="
  echo "Config: ${config}"
  echo "Log: ${train_log}"
  "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/train_full.py" \
    "${train_args[@]}" \
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
    echo "Usage: PUBMED_SOURCE_DIR=/path/to/pubmet bash run_pubmed_pipeline.sh {1b|4b|all}" >&2
    exit 2
    ;;
esac
