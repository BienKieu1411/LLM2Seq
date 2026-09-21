#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_train_only.sh CONFIG TRAIN_JSONL [TRAIN OPTIONS]

Example:
  bash scripts/run_train_only.sh \
    configs/8192_avg.yaml /data/my_dataset/train.jsonl \
    --device cuda:0 --overwrite-output-dir
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

CONFIG="$1"
TRAIN_FILE="$2"
shift 2

[[ -f "$CONFIG" ]] || { echo "Config not found: $CONFIG" >&2; exit 1; }
[[ -f "$TRAIN_FILE" ]] || { echo "Training JSONL not found: $TRAIN_FILE" >&2; exit 1; }

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON_BIN" "$ROOT/run_afmr.py" train "$CONFIG" \
  --train-file "$TRAIN_FILE" \
  --train-only \
  "$@"
