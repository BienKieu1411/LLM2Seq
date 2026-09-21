#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CONFIG=""
CHECKPOINT=""
DATASET_DIR="$ROOT/datasets/wikilingua"
OUTPUT_DIR=""
ADDITIONAL_EPOCHS=5
DEVICE=""
EVAL_BATCH_SIZE=""
MAX_EVAL_EXAMPLES=0
OVERWRITE=0

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/continue_wikilingua.sh \
    --config CONFIG \
    --checkpoint CHECKPOINT \
    [--dataset-dir DATASET_DIR] \
    [--output-dir OUTPUT_DIR] \
    [--additional-epochs N] \
    [--device DEVICE] \
    [--eval-batch-size N] \
    [--max-eval-examples N] \
    [--overwrite-output-dir]

The checkpoint's model and optimizer are resumed, then the remaining run is
continued on WikiLingua. The generated configuration is saved in OUTPUT_DIR.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --dataset-dir) DATASET_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --additional-epochs) ADDITIONAL_EPOCHS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --max-eval-examples) MAX_EVAL_EXAMPLES="$2"; shift 2 ;;
    --overwrite-output-dir) OVERWRITE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$CONFIG" || -z "$CHECKPOINT" ]]; then
  usage
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "WikiLingua directory not found: $DATASET_DIR" >&2
  exit 1
fi
for split in train validation test; do
  if [[ ! -f "$DATASET_DIR/$split.jsonl" ]]; then
    echo "Missing WikiLingua split: $DATASET_DIR/$split.jsonl" >&2
    exit 1
  fi
done
if ! [[ "$ADDITIONAL_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--additional-epochs must be a positive integer" >&2
  exit 2
fi

CHECKPOINT="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
DATASET_DIR="$(cd "$DATASET_DIR" && pwd)"
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$ROOT/runs/afmr/wikilingua_from_$(basename "$(dirname "$CHECKPOINT")")"
fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ "$OUTPUT_DIR" == "$(dirname "$CHECKPOINT")" ]]; then
  echo "OUTPUT_DIR must differ from the checkpoint directory" >&2
  exit 2
fi

GENERATED_CONFIG="$OUTPUT_DIR/wikilingua_resume.yaml"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" - "$CONFIG" "$CHECKPOINT" "$DATASET_DIR" "$OUTPUT_DIR" "$ADDITIONAL_EPOCHS" "$GENERATED_CONFIG" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

from eviseq_afmr.config import load_config

config_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
dataset_dir = Path(sys.argv[3])
output_dir = Path(sys.argv[4])
additional_epochs = int(sys.argv[5])
generated_path = Path(sys.argv[6])
config = load_config(config_path)
state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
stage = state.get("stage")
stage_epoch = int(state.get("stage_epoch") or 0)
if stage not in {"interface_warmup", "full_finetune"}:
    raise SystemExit(
        f"Checkpoint stage must be interface_warmup or full_finetune; got {stage!r}. "
        "Use the last.pt produced by AFMR training."
    )

data = config["data"]
data["train_file"] = str(dataset_dir / "train.jsonl")
data["validation_file"] = str(dataset_dir / "validation.jsonl")
data["test_file"] = str(dataset_dir / "test.jsonl")
config["experiment"]["output_dir"] = str(output_dir)
config["training"]["resume_checkpoint"] = ""

# The trainer interprets stage epoch counts as totals when resuming. Extend
# the active stage by the requested number of WikiLingua epochs so a completed
# train-only run does not silently perform zero updates.
if stage == "full_finetune":
    config["training"]["full_finetune_epochs"] = stage_epoch + additional_epochs
else:
    config["training"]["full_finetune_epochs"] = additional_epochs

config.pop("_meta", None)
generated_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"checkpoint_stage={stage}")
print(f"checkpoint_stage_epoch={stage_epoch}")
print(f"wikilingua_epochs={additional_epochs}")
print(f"generated_config={generated_path}")
PY

TRAIN_ARGS=("$ROOT/run_afmr.py" train "$GENERATED_CONFIG" --resume-checkpoint "$CHECKPOINT")
if [[ -n "$DEVICE" ]]; then TRAIN_ARGS+=(--device "$DEVICE"); fi
if [[ "$OVERWRITE" == 1 ]]; then TRAIN_ARGS+=(--overwrite-output-dir); fi
"$PYTHON_BIN" "${TRAIN_ARGS[@]}"

PREDICTIONS="$OUTPUT_DIR/test_predictions.jsonl"
EVAL_ARGS=("$ROOT/run_afmr.py" evaluate "$GENERATED_CONFIG" "$OUTPUT_DIR/last.pt" "$PREDICTIONS" --split test --max-examples "$MAX_EVAL_EXAMPLES")
if [[ -n "$DEVICE" ]]; then EVAL_ARGS+=(--device "$DEVICE"); fi
if [[ -n "$EVAL_BATCH_SIZE" ]]; then EVAL_ARGS+=(--batch-size "$EVAL_BATCH_SIZE"); fi
"$PYTHON_BIN" "${EVAL_ARGS[@]}"

echo "Training and evaluation complete. Predictions: $PREDICTIONS"
