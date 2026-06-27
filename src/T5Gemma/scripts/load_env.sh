#!/usr/bin/env bash

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

export PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
export CONFIG="${CONFIG:-T5Gemma/configs/wikilingua_lora_3072.yaml}"
export WIKI_DIR="${WIKI_DIR:-T5Gemma/wikilingua}"
export DATA_DIR="${DATA_DIR:-T5Gemma/data/processed}"
export RUN_DIR="${RUN_DIR:-runs/t5gemma2_1b_1b_lora_wikilingua}"
export EVAL_DIR="${EVAL_DIR:-T5Gemma/eval_outputs/full_test}"
export LOG_DIR="${LOG_DIR:-T5Gemma/logs}"
export EVAL_LIMIT="${EVAL_LIMIT:--1}"
export RUN_EVAL="${RUN_EVAL:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${T5GEMMA_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${T5GEMMA_ROOT}:${PROJECT_ROOT}"
fi
