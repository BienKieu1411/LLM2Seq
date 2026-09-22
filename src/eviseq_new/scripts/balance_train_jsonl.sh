#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python3}}"
DEFAULT_ROOT="/workspace/storage-shared/nlp/dungdx4/nemo_restore/datasets/raw_data/finetune/all_datasets_stratified/32768_aggregated_small"
INPUT_JSONL="${INPUT_JSONL:-${DEFAULT_ROOT}/train.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${DEFAULT_ROOT}/train_balanced.jsonl}"
REPORT_JSON="${REPORT_JSON:-${DEFAULT_ROOT}/train_balanced.report.json}"
TARGET_TOKENIZER="${TARGET_TOKENIZER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
SOURCE_FIELD="${SOURCE_FIELD:-input}"
TARGET_FIELD="${TARGET_FIELD:-output}"
MAX_SOURCE_TOKENS="${MAX_SOURCE_TOKENS:-2048}"
MAX_TARGET_TOKENS="${MAX_TARGET_TOKENS:-512}"
TARGET_BELOW="${TARGET_BELOW:-}"
SOURCE_BINS="${SOURCE_BINS:-50}"
MAX_PER_SOURCE_BIN="${MAX_PER_SOURCE_BIN:-20000}"
SELECTION="${SELECTION:-random}"
SEED="${SEED:-17}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/balance_train_jsonl.sh [options]

Defaults: source <= 2048 tokens, target <= 512 tokens, 50 source-length bins,
20,000 samples/bin, and detokenized source/target output.

Options:
  --input PATH                 Input train JSONL
  --output PATH                Balanced output JSONL
  --report PATH                JSON report path
  --target-tokenizer PATH      Local tokenizer for target lengths
  --source-field FIELD         Source field/path (default: input)
  --target-field FIELD         Target field/path (default: output)
  --max-source-tokens N        Inclusive source limit <= N (default: 2048)
  --max-target-tokens N        Inclusive target limit <= N (default: 512)
  --target-below N             Strict target limit < N
  --source-bins N              Number of source-length bins (default: 50)
  --max-per-source-bin N       Cap per source bin (default: 20000)
  --selection {first,random}   Sampling policy (default: random)
  --seed N                     Random seed (default: 17)
  --python PATH                Python executable (default: python3)
  -h, --help                   Show this help
EOF
}

while (($#)); do
  case "$1" in
    --input) INPUT_JSONL="$2"; shift 2 ;;
    --output) OUTPUT_JSONL="$2"; shift 2 ;;
    --report) REPORT_JSON="$2"; shift 2 ;;
    --target-tokenizer) TARGET_TOKENIZER="$2"; shift 2 ;;
    --source-field) SOURCE_FIELD="$2"; shift 2 ;;
    --target-field) TARGET_FIELD="$2"; shift 2 ;;
    --max-source-tokens) MAX_SOURCE_TOKENS="$2"; shift 2 ;;
    --max-target-tokens) MAX_TARGET_TOKENS="$2"; TARGET_BELOW=""; shift 2 ;;
    --target-below) TARGET_BELOW="$2"; MAX_TARGET_TOKENS=""; shift 2 ;;
    --source-bins) SOURCE_BINS="$2"; shift 2 ;;
    --max-per-source-bin) MAX_PER_SOURCE_BIN="$2"; shift 2 ;;
    --selection) SELECTION="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$INPUT_JSONL" ]] || { echo "Input JSONL not found: $INPUT_JSONL" >&2; exit 1; }
[[ -d "$TARGET_TOKENIZER" ]] || { echo "Target tokenizer not found: $TARGET_TOKENIZER" >&2; exit 1; }
[[ "$OUTPUT_JSONL" != "$INPUT_JSONL" ]] || { echo "Output must differ from input" >&2; exit 1; }

mkdir -p "$(dirname "$OUTPUT_JSONL")" "$(dirname "$REPORT_JSON")"

ARGS=(
  "$SCRIPT_DIR/filter_jsonl_by_target_length.py"
  "$INPUT_JSONL" "$OUTPUT_JSONL"
  --source-field "$SOURCE_FIELD"
  --target-field "$TARGET_FIELD"
  --target-tokenizer "$TARGET_TOKENIZER"
  --max-source-tokens "$MAX_SOURCE_TOKENS"
  --detokenize
  --balance-source-bins "$SOURCE_BINS"
  --max-per-source-bin "$MAX_PER_SOURCE_BIN"
  --selection "$SELECTION"
  --seed "$SEED"
  --report "$REPORT_JSON"
)
if [[ -n "$TARGET_BELOW" ]]; then
  ARGS+=(--target-below "$TARGET_BELOW")
else
  ARGS+=(--max-target-tokens "$MAX_TARGET_TOKENS")
fi

echo "[balance] input=$INPUT_JSONL"
echo "[balance] output=$OUTPUT_JSONL"
if [[ -n "$TARGET_BELOW" ]]; then
  TARGET_LIMIT="<${TARGET_BELOW}"
else
  TARGET_LIMIT="<=${MAX_TARGET_TOKENS}"
fi
echo "[balance] detokenize=true source<=${MAX_SOURCE_TOKENS} target=${TARGET_LIMIT} source_bins=$SOURCE_BINS max_per_bin=$MAX_PER_SOURCE_BIN"
exec "$PYTHON_BIN" "${ARGS[@]}"
