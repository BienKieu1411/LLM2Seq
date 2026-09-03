#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
cd "$SRC_ROOT"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$VIRTUAL_ENV/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="$PACKAGE_ROOT:$SRC_ROOT/llm2seq_v2:$SRC_ROOT/eviseq_v2${PYTHONPATH:+:$PYTHONPATH}"
# These are mandatory, not optional hints: from_pretrained also receives
# local_files_only=true in Python, so the runner cannot fetch a missing model.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

CONFIG="$PACKAGE_ROOT/configs/wikilingua.yaml"

config_value() {
  "$PYTHON_BIN" - "$CONFIG" "$1" <<'PY'
import sys
from direct_qwen.config import load_config
value = load_config(sys.argv[1])
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

OUTPUT_DIR="$(config_value experiment.output_dir)"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$MODE" in
  test)
    "$PYTHON_BIN" -m pytest -q "$PACKAGE_ROOT/tests"
    ;;
  inspect)
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import json, sys
from direct_qwen.config import load_config
print(json.dumps(load_config(sys.argv[1]), ensure_ascii=False, indent=2))
PY
    ;;
  train)
    "$PYTHON_BIN" -m direct_qwen.training --config "$CONFIG" "$@"
    ;;
  eval-validation)
    "$PYTHON_BIN" -m direct_qwen.evaluate \
      --config "$OUTPUT_DIR/resolved_config.yaml" \
      --checkpoint "$OUTPUT_DIR/last.pt" \
      --output "$OUTPUT_DIR/last_validation_predictions.jsonl" \
      --split validation "$@"
    ;;
  wiki)
    "$PYTHON_BIN" -m direct_qwen.training --config "$CONFIG" "$@"
    "$PYTHON_BIN" -m direct_qwen.evaluate \
      --config "$OUTPUT_DIR/resolved_config.yaml" \
      --checkpoint "$OUTPUT_DIR/last.pt" \
      --output "$OUTPUT_DIR/last_validation_predictions.jsonl" \
      --split validation
    ;;
  paper-test)
    "$PYTHON_BIN" -m direct_qwen.evaluate \
      --config "$OUTPUT_DIR/resolved_config.yaml" \
      --checkpoint "$OUTPUT_DIR/last.pt" \
      --output "$OUTPUT_DIR/last_test_predictions.jsonl" \
      --split test --paper-test
    ;;
  rouge155)
    PREDICTIONS="${1:?Pass the complete paper-test predictions.jsonl}"
    "$PYTHON_BIN" "$SRC_ROOT/rouge155/evaluate_rouge.py" "$PREDICTIONS"
    ;;
  *)
    cat <<'EOF'
Direct Qwen3-0.6B full-finetune control (offline and last-only)

  # Optional: use an already-present local model directory.
  export DIRECT_QWEN_MODEL_PATH=/absolute/local/path/to/Qwen3-0.6B

  bash direct_qwen/run.sh test
  bash direct_qwen/run.sh inspect
  bash direct_qwen/run.sh wiki --overwrite-output-dir
  bash direct_qwen/run.sh paper-test
  bash direct_qwen/run.sh rouge155 runs/direct_qwen/qwen3_0_6b_full_wikilingua/last_test_predictions.jsonl
EOF
    ;;
esac
