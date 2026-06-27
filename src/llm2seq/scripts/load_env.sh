#!/usr/bin/env bash

LOAD_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLM2SEQ_ROOT="$(cd "${LOAD_ENV_SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT="$(cd "${LLM2SEQ_ROOT}/.." && pwd)"
export LLM2SEQ_DIR_NAME="$(basename "${LLM2SEQ_ROOT}")"

if [[ -z "${ENV_FILE:-}" ]]; then
  if [[ -f "${LLM2SEQ_ROOT}/env.txt" ]]; then
    ENV_FILE="${LLM2SEQ_ROOT}/env.txt"
  elif [[ -f "${LLM2SEQ_ROOT}/.env" ]]; then
    ENV_FILE="${LLM2SEQ_ROOT}/.env"
  else
    ENV_FILE="${LLM2SEQ_ROOT}/env.txt"
  fi
fi

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_AUTO_DOWNLOAD_CHECKPOINTS="${HF_AUTO_DOWNLOAD_CHECKPOINTS:-true}"
export HF_CHECKPOINT_CACHE="${HF_CHECKPOINT_CACHE:-runs/hf_checkpoints}"

# Make this folder self-contained: expose the bundled llm2seq package inside
# llm2seq/llm2seq when the parent repo is not uploaded.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${LLM2SEQ_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${LLM2SEQ_ROOT}:${PROJECT_ROOT}"
fi

# Most configs intentionally use llm2seq/... paths. If the uploaded folder
# was renamed, create a best-effort compatibility symlink in the parent folder.
if [[ "${LLM2SEQ_DIR_NAME}" != "llm2seq" && ! -e "${PROJECT_ROOT}/llm2seq" ]]; then
  ln -s "${LLM2SEQ_ROOT}" "${PROJECT_ROOT}/llm2seq" 2>/dev/null || true
fi

resolve_hf_checkpoint() {
  local phase="$1"
  local local_path="$2"
  if [[ -f "${local_path}" ]]; then
    printf '%s\n' "${local_path}"
    return 0
  fi
  "${PYTHON_BIN}" "${LLM2SEQ_ROOT}/scripts/hf_checkpoint.py" resolve \
    --phase "${phase}" \
    --local "${local_path}"
}

resolve_hf_checkpoint_from_config() {
  local config_path="$1"
  local local_path="$2"
  if [[ -f "${local_path}" ]]; then
    printf '%s\n' "${local_path}"
    return 0
  fi
  "${PYTHON_BIN}" "${LLM2SEQ_ROOT}/scripts/hf_checkpoint.py" resolve \
    --config "${config_path}" \
    --local "${local_path}"
}
