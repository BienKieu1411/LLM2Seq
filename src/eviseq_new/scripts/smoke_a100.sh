#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/afmr_pubmed.yaml}"
TRAIN_EXAMPLES="${AFMR_SMOKE_TRAIN_EXAMPLES:-100}"
VALIDATION_EXAMPLES="${AFMR_SMOKE_VALIDATION_EXAMPLES:-20}"
EVAL_EXAMPLES="${AFMR_SMOKE_EVAL_EXAMPLES:-4}"
EVAL_BATCH_SIZE="${AFMR_SMOKE_EVAL_BATCH_SIZE:-2}"
PYTHON_BIN="${PYTHON:-python3}"
DEVICE="${AFMR_SMOKE_DEVICE:-cuda}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi
CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"

mkdir -p "$ROOT/runs/smoke"
OUTPUT_DIR="$(mktemp -d "$ROOT/runs/smoke/run_XXXXXXXX")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
"$PYTHON_BIN" - "$CONFIG" "$ROOT" "$OUTPUT_DIR" <<'PY'
import sys
from pathlib import Path
import yaml

sys.path.insert(0, sys.argv[2])
from eviseq_afmr.config import load_config, resolve_path

config = load_config(Path(sys.argv[1]))
for key in ("train_file", "validation_file", "test_file"):
    config["data"][key] = str(resolve_path(config["data"][key], config))
config.pop("_meta", None)
config["experiment"]["output_dir"] = sys.argv[3]
config["training"].update(interface_warmup_epochs=1, full_finetune_epochs=1,
    batch_size=2, gradient_accumulation_steps=2, validation_batch_size=2,
    resume_checkpoint="", num_workers=0, validation_num_workers=0, log_every_steps=1)
config["model"]["gradient_checkpointing"] = True
with open(Path(sys.argv[3]) / "smoke_config.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
CONFIG="$OUTPUT_DIR/smoke_config.yaml"
echo "Isolated smoke output: $OUTPUT_DIR"

bash "$ROOT/scripts/run_afmr.sh" train "$CONFIG" \
  --device "$DEVICE" \
  --max-train-examples "$TRAIN_EXAMPLES" \
  --max-validation-examples "$VALIDATION_EXAMPLES"

bash "$ROOT/scripts/run_afmr.sh" evaluate "$CONFIG" "$OUTPUT_DIR/last.pt" \
  "$OUTPUT_DIR/smoke_test_predictions.jsonl" \
  --split test \
  --device "$DEVICE" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-examples "$EVAL_EXAMPLES"
