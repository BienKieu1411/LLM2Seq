#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
DEFAULT_ROOT="/workspace/storage-shared/nlp/dungdx4/nemo_restore/datasets/raw_data/finetune/all_datasets_stratified/32768_aggregated_small"
INPUT_JSONL="${INPUT_JSONL:-${DEFAULT_ROOT}/train_balanced.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${ROOT}/datasets/8192_avg/train.jsonl}"
SOURCE_FIELD="${SOURCE_FIELD:-input}"
TARGET_FIELD="${TARGET_FIELD:-output}"
ID_FIELD="${ID_FIELD:-id}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_8192_avg_train.sh [options]

Converts a balanced JSONL with input/output fields into the canonical format:
  {"id": ..., "text": ..., "summary": ...}

Source and summary text are detokenized during conversion.

Options:
  --input PATH          Balanced input JSONL
  --output PATH         Canonical output JSONL
  --source-field FIELD  Input source field/path (default: input)
  --target-field FIELD  Input target field/path (default: output)
  --id-field FIELD      Input ID field/path (default: id)
  --python PATH         Python executable (default: python3)
  -h, --help            Show this help
EOF
}

while (($#)); do
  case "$1" in
    --input) INPUT_JSONL="$2"; shift 2 ;;
    --output) OUTPUT_JSONL="$2"; shift 2 ;;
    --source-field) SOURCE_FIELD="$2"; shift 2 ;;
    --target-field) TARGET_FIELD="$2"; shift 2 ;;
    --id-field) ID_FIELD="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$INPUT_JSONL" ]] || { echo "Input JSONL not found: $INPUT_JSONL" >&2; exit 1; }
[[ "$INPUT_JSONL" != "$OUTPUT_JSONL" ]] || { echo "Output must differ from input" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT_JSONL")"
cd "$ROOT"
exec "$PYTHON_BIN" "$ROOT/run_afmr.py" prepare \
  "$INPUT_JSONL" "$OUTPUT_JSONL" \
  --source-field "$SOURCE_FIELD" \
  --target-field "$TARGET_FIELD" \
  --id-field "$ID_FIELD" \
  --detokenize
