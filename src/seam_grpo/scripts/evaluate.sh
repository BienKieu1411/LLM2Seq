#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR=""
CHECKPOINT="best.pt"
SPLIT="test"
EXTRA=()
while (($#)); do
  case "$1" in
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
if [[ -z "$RUN_DIR" ]]; then
  echo 'Usage: evaluate.sh --run-dir /path/to/grpo_run [--checkpoint best.pt] [--split test] [native eval options]' >&2
  exit 2
fi
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
if [[ "$CHECKPOINT" = /* ]]; then CKPT="$CHECKPOINT"; else CKPT="$RUN_DIR/$CHECKPOINT"; fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${PYTHON_BIN:-python3}" "$ROOT/../eviseq_new/run_afmr.py" evaluate \
  "$RUN_DIR/resolved_config.yaml" "$CKPT" \
  "$RUN_DIR/${SPLIT}_$(basename "$CHECKPOINT" .pt)_predictions.jsonl" --split "$SPLIT" "${EXTRA[@]}"
