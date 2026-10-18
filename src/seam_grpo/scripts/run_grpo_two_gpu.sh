#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_GPU="${TRAIN_GPU:-0}"
ROLLOUT_GPU="${ROLLOUT_GPU:-1}"
if [[ "$TRAIN_GPU" == "$ROLLOUT_GPU" || "$TRAIN_GPU" == *,* || "$ROLLOUT_GPU" == *,* ]]; then
  echo 'TRAIN_GPU and ROLLOUT_GPU must each name one different GPU.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$TRAIN_GPU"
export NPROC_PER_NODE=1
echo "SEAM GRPO: train GPU=$TRAIN_GPU; dedicated vLLM GPU=$ROLLOUT_GPU; one training process"
exec bash "$ROOT/scripts/run_grpo.sh" "$@" --vllm-mode dedicated --rollout-gpu "$ROLLOUT_GPU"
