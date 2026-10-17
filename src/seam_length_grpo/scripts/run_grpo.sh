#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${PYTHON_BIN:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NPROC_PER_NODE must be a positive integer' >&2
  exit 2
fi
if (( NPROC_PER_NODE > 1 )); then
  exec "$PYTHON_BIN" -m torch.distributed.run --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29517}" \
    "$ROOT/run_grpo.py" "$@"
fi
exec "$PYTHON_BIN" "$ROOT/run_grpo.py" "$@"
