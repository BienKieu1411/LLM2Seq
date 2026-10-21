#!/usr/bin/env bash
set -euo pipefail
SI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${GPU_IDS:-}" ]]; then
  GPU_LIST="$GPU_IDS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPU_LIST="$CUDA_VISIBLE_DEVICES"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Set GPU_IDS or CUDA_VISIBLE_DEVICES to the allocated GPU IDs." >&2
    exit 2
  fi
  mapfile -t VISIBLE_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if ((${#VISIBLE_GPU_IDS[@]} == 0)); then
    echo "No visible GPUs found; set GPU_IDS to the allocated GPU IDs." >&2
    exit 2
  fi
  GPU_LIST="$(IFS=,; printf '%s' "${VISIBLE_GPU_IDS[*]}")"
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${PYTHON_BIN:-python3}" "$SI_ROOT/prepare_si.py" \
  --gpus "$GPU_LIST" --start-server "$@"
