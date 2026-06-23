#!/usr/bin/env bash

LOAD_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export H200_ROOT="$(cd "${LOAD_ENV_SCRIPT_DIR}/.." && pwd)"
export PROJECT_ROOT="$(cd "${H200_ROOT}/.." && pwd)"
export H200_DIR_NAME="$(basename "${H200_ROOT}")"

if [[ -z "${ENV_FILE:-}" ]]; then
  if [[ -f "${H200_ROOT}/env.txt" ]]; then
    ENV_FILE="${H200_ROOT}/env.txt"
  elif [[ -f "${H200_ROOT}/.env" ]]; then
    ENV_FILE="${H200_ROOT}/.env"
  else
    ENV_FILE="${H200_ROOT}/env.txt"
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
# llm2seq_final/llm2seq when the parent repo is not uploaded.
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${H200_ROOT}:${PROJECT_ROOT}:${PYTHONPATH}"
else
  export PYTHONPATH="${H200_ROOT}:${PROJECT_ROOT}"
fi

# Most configs intentionally use llm2seq_final/... paths. If the uploaded folder
# was renamed, create a best-effort compatibility symlink in the parent folder.
if [[ "${H200_DIR_NAME}" != "llm2seq_final" && ! -e "${PROJECT_ROOT}/llm2seq_final" ]]; then
  ln -s "${H200_ROOT}" "${PROJECT_ROOT}/llm2seq_final" 2>/dev/null || true
fi

resolve_hf_checkpoint() {
  local phase="$1"
  local local_path="$2"
  if [[ -f "${local_path}" ]]; then
    printf '%s\n' "${local_path}"
    return 0
  fi
  "${PYTHON_BIN}" "${H200_ROOT}/scripts/hf_checkpoint.py" resolve \
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
  "${PYTHON_BIN}" "${H200_ROOT}/scripts/hf_checkpoint.py" resolve \
    --config "${config_path}" \
    --local "${local_path}"
}
