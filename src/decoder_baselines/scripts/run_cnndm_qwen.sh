#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || {
  echo "Use exactly one GPU, e.g. GPU_ID=0" >&2
  exit 2
}
export GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export QWEN3_0_6B_PATH="${QWEN3_0_6B_PATH:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-0.6B}"
export DECODER_EVAL_BACKEND="${DECODER_EVAL_BACKEND:-local}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec bash "${BASELINE_ROOT}/scripts/run_suite.sh" --models qwen3_0_6b --datasets cnndm "$@"
