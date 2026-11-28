#!/usr/bin/env bash
set -Eeuo pipefail

EV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${CNNDM_SOURCE_DIR:-/workspace/storage-shared/nlp/users/dungdx4/datasets/cnndm}"
DATA_DIR="${CNNDM_DATA_DIR:-${EV_ROOT}/datasets/cnndm}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="python3"
fi

DATA_PARENT="$(dirname "${DATA_DIR}")"
mkdir -p "${DATA_PARENT}"
DATA_PARENT="$(cd "${DATA_PARENT}" && pwd)"
DATA_DIR="${DATA_PARENT}/$(basename "${DATA_DIR}")"
LOCK_DIR="${DATA_DIR}.prepare.lock"

is_prepared() {
  [[ -s "${DATA_DIR}/train.jsonl" \
    && -s "${DATA_DIR}/validation.jsonl" \
    && -s "${DATA_DIR}/test.jsonl" \
    && -s "${DATA_DIR}/preparation_report.json" ]]
}

if is_prepared; then
  echo "CNN/DM canonical splits already exist: ${DATA_DIR}"
  exit 0
fi

waited=0
until mkdir "${LOCK_DIR}" 2>/dev/null; do
  if is_prepared; then
    echo "CNN/DM preparation finished in another process: ${DATA_DIR}"
    exit 0
  fi
  (( waited < 3600 )) || die "Timed out waiting for CNN/DM preparation lock: ${LOCK_DIR}"
  sleep 5
  waited=$((waited + 5))
done
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

if is_prepared; then
  echo "CNN/DM canonical splits already exist: ${DATA_DIR}"
  exit 0
fi
[[ -d "${SOURCE_DIR}" ]] || die "Raw CNN/DM directory not found: ${SOURCE_DIR}; set CNNDM_SOURCE_DIR"

echo "Preparing raw CNN/DM data from ${SOURCE_DIR}"
PYTHON="${PYTHON_BIN}" bash "${EV_ROOT}/scripts/prepare_afmr.sh" \
  --dataset cnndm \
  --input-dir "${SOURCE_DIR}" \
  --output-dir "${DATA_DIR}" \
  --raw-copy-dir "${SOURCE_DIR}" \
  --allow-cross-split-content

is_prepared || die "Preparation did not create all canonical splits and preparation_report.json in ${DATA_DIR}"
echo "CNN/DM data ready: ${DATA_DIR}"
