#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EV_ROOT="$(cd "${BASELINE_ROOT}/../eviseq_new" && pwd)"
MODEL="${1:-qwen3_0_6b}"
if [[ "$#" -gt 1 ]]; then
  echo "Usage: GPU_ID=0 bash scripts/run_cnndm_qwen.sh [qwen3_0_6b|qwen3_1_7b|qwen3_4b]" >&2
  exit 2
fi
case "${MODEL}" in
  qwen3_0_6b)
    export QWEN3_0_6B_PATH="${QWEN3_0_6B_PATH:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-0.6B}"
    ;;
  qwen3_1_7b)
    export QWEN3_1_7B_PATH="${QWEN3_1_7B_PATH:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-1.7B}"
    ;;
  qwen3_4b)
    export QWEN3_4B_PATH="${QWEN3_4B_PATH:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-4B}"
    ;;
  *)
    echo "Unknown model '${MODEL}'. Choose qwen3_0_6b, qwen3_1_7b or qwen3_4b." >&2
    exit 2
    ;;
esac
GPU_ID="${GPU_ID:-0}"
[[ "${GPU_ID}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
  echo "Use one or more GPUs, e.g. GPU_ID=0 or GPU_ID=0,1" >&2
  exit 2
}
export GPU_ID
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CNNDM_DATA_DIR="${CNNDM_DATA_DIR:-${EV_ROOT}/datasets/cnndm}"
export CNNDM_SOURCE_DIR="${CNNDM_SOURCE_DIR:-/workspace/storage-shared/nlp/users/dungdx4/datasets/cnndm}"
export DECODER_EVAL_BACKEND="${DECODER_EVAL_BACKEND:-vllm}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

bash "${EV_ROOT}/scripts/ensure_cnndm_prepared.sh"

exec bash "${BASELINE_ROOT}/scripts/run_suite.sh" --models "${MODEL}" --datasets cnndm
