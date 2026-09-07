#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
processes="${NPROC_PER_NODE:-1}"
if ! [[ "$processes" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi
if [[ "${1:-}" == "train" ]]; then
  if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
    set -- "$@" --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
  fi
  if (( processes > 1 )); then
    exec "${PYTHON:-python3}" -m torch.distributed.run --standalone --nnodes=1 --local-addr=127.0.0.1 \
      --nproc_per_node="$processes" "$ROOT/run_afmr.py" "$@"
  fi
fi
exec "${PYTHON:-python3}" "$ROOT/run_afmr.py" "$@"
