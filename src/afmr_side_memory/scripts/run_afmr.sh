#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON:-python3}"

# Training uses one process per visible GPU. Evaluation remains a single
# process; run explicit evaluation shards when multi-GPU decoding is desired.
NPROC="${AFMR_NPROC:-}"
if [[ -z "${NPROC}" ]]; then
  IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES:-0}"
  NPROC="${#visible_devices[@]}"
fi
if [[ "${1:-}" == train && "${NPROC}" -gt 1 ]]; then
  exec "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" "$ROOT/run_afmr.py" "$@"
fi
exec "${PYTHON_BIN}" "$ROOT/run_afmr.py" "$@"
