#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  cat >&2 <<'USAGE'
Usage: retrain_gated.sh CONFIG TRAIN_JSONL VALIDATION_JSONL [RUN_DIR]

The configured target must be the local base google/t5gemma-2-1b-1b checkpoint.
The script first checks target cache/block parity and stops on any mismatch.
USAGE
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"
TRAIN_FILE="$(cd -- "$(dirname -- "$2")" && pwd)/$(basename -- "$2")"
VALIDATION_FILE="$(cd -- "$(dirname -- "$3")" && pwd)/$(basename -- "$3")"
RUN_DIR="${4:-$PWD/runs/encdec_drafter_margin/$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$(mkdir -p -- "$RUN_DIR" && cd -- "$RUN_DIR" && pwd)"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
TRAIN_STEPS="${TRAIN_STEPS:-5000}"
BENCHMARK_LIMIT="${BENCHMARK_LIMIT:-128}"

export PYTHONPATH="$PACKAGE_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "[gate] checking native, cached, full-prefix, and block verifier parity"
"$PYTHON" -m encdec_drafter diagnose-verifier \
  --config "$CONFIG" \
  --input-file "$VALIDATION_FILE" \
  --limit 8 \
  --prefix-tokens 32 \
  --block-sizes 2 4 8 \
  --output "$RUN_DIR/verifier_gate.json"

echo "[traces] collecting target-greedy PubMed train traces (limit=$TRAIN_LIMIT)"
"$PYTHON" -m encdec_drafter collect-traces \
  --config "$CONFIG" \
  --input-file "$TRAIN_FILE" \
  --output "$RUN_DIR/teacher_traces.jsonl" \
  --limit "$TRAIN_LIMIT" \
  --overwrite

echo "[train] greedy top-1 margin accepted-length surrogate updates (steps=$TRAIN_STEPS)"
"$PYTHON" -m encdec_drafter train \
  --config "$CONFIG" \
  --traces "$RUN_DIR/teacher_traces.jsonl" \
  --output "$RUN_DIR/drafter.pt" \
  --steps "$TRAIN_STEPS"

echo "[benchmark] paired native/speculative validation (limit=$BENCHMARK_LIMIT)"
"$PYTHON" -m encdec_drafter benchmark \
  --config "$CONFIG" \
  --input-file "$VALIDATION_FILE" \
  --draft-checkpoint "$RUN_DIR/drafter.pt" \
  --output-dir "$RUN_DIR/benchmark" \
  --limit "$BENCHMARK_LIMIT" \
  --repeats 3 \
  --warmup 4 \
  --require-exact

echo "Run artifacts: $RUN_DIR"
