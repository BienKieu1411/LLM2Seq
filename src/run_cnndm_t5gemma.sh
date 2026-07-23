#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SOURCE_DIR="${1:-${CNNDM_SOURCE_DIR:-}}"
GPU_ID="${2:-${CNNDM_GPU_ID:-1}}"

if [[ -z "${SOURCE_DIR}" ]]; then
  cat >&2 <<'EOF'
Usage:
  bash run_cnndm_t5gemma.sh /absolute/path/to/cnndm [gpu_id]

The source folder must contain train.txt, val.txt, and test.txt.
The default GPU id is 1. T5Gemma 1B-1B runs first, followed by 4B-4B.
EOF
  exit 2
fi
if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "ERROR: CNN/DailyMail source folder does not exist: ${SOURCE_DIR}" >&2
  exit 2
fi
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: gpu_id must be a non-negative integer, got: ${GPU_ID}" >&2
  exit 2
fi

export CNNDM_SOURCE_DIR="${SOURCE_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "=== Dedicated CNN/DailyMail T5Gemma queue ==="
echo "Physical GPU: ${GPU_ID}"
echo "Dataset source: ${CNNDM_SOURCE_DIR}"
echo "Order: T5Gemma 1B-1B -> T5Gemma 4B-4B"

bash T5Gemma/run_cnndm_pipeline.sh all
