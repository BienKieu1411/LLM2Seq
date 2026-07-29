#!/usr/bin/env bash

# A GPU chosen by the caller must take precedence over a stale value in
# T5Gemma/env.txt. This is essential when independent queues use GPU 0 and 1.
CALLER_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
CALLER_SET_CUDA_VISIBLE_DEVICES=false
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
  CALLER_SET_CUDA_VISIBLE_DEVICES=true
fi

LOAD_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export T5GEMMA_ROOT="$(cd "${LOAD_ENV_SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT="$(cd "${T5GEMMA_ROOT}/.." && pwd)"

if [[ -z "${ENV_FILE:-}" ]]; then
  if [[ -f "${T5GEMMA_ROOT}/env.txt" ]]; then
    ENV_FILE="${T5GEMMA_ROOT}/env.txt"
  elif [[ -f "${T5GEMMA_ROOT}/.env" ]]; then
    ENV_FILE="${T5GEMMA_ROOT}/.env"
  else
    ENV_FILE="${T5GEMMA_ROOT}/env.txt"
  fi
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

if [[ "${CALLER_SET_CUDA_VISIBLE_DEVICES}" == "true" ]]; then
  export CUDA_VISIBLE_DEVICES="${CALLER_CUDA_VISIBLE_DEVICES}"
fi

BIENKIEU_PYTHON="/Users/kieugiangbien/bienkieu_env/bin/python"
if [[ -z "${PYTHON_BIN:-}" && -x "${BIENKIEU_PYTHON}" ]]; then
  export PYTHON_BIN="${BIENKIEU_PYTHON}"
else
  export PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
fi
export CONFIG="${CONFIG:-T5Gemma/configs/wikilingua_full_3072.yaml}"
export WIKI_DIR="${WIKI_DIR:-T5Gemma/datasets/wikilingua}"
export DATA_DIR="${DATA_DIR:-T5Gemma/data/processed}"
export RUN_DIR="${RUN_DIR:-runs/t5gemma2_1b_1b_full_wikilingua}"
export EVAL_DIR="${EVAL_DIR:-T5Gemma/eval_outputs/full_test}"
export LOG_DIR="${LOG_DIR:-T5Gemma/logs}"
export EVAL_LIMIT="${EVAL_LIMIT:--1}"
export RUN_EVAL="${RUN_EVAL:-true}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${T5GEMMA_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${T5GEMMA_ROOT}:${PROJECT_ROOT}"
fi
