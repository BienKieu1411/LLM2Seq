#!/usr/bin/env bash
set -Eeuo pipefail

# Mix batch files, prepared data, and WikiLingua train once; train once with
# WikiLingua validation/test.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN=python3
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BATCH_DIR="${BATCH_DIR:-}"
PREPARED_TRAIN="${PREPARED_TRAIN:-${SUMMARY_TRAIN:-${ROOT}/datasets/90k/train.jsonl}}"
WIKI_DIR="${WIKILINGUA_DIR:-${ROOT}/datasets/wikilingua}"
MIXED_DIR="${MIXED_DATA_DIR:-${ROOT}/datasets/summary_wikilingua_mixed}"
OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/summary_wikilingua_mixed}"
PREPARED_SOURCE_FIELD="${PREPARED_SOURCE_FIELD:-text}"
PREPARED_TARGET_FIELD="${PREPARED_TARGET_FIELD:-summary}"
PREPARED_ID_FIELD="${PREPARED_ID_FIELD:-id}"
BATCH_PATTERN="${BATCH_PATTERN:-batch*.jsonl}"
SEED="${MIX_SEED:-42}"
EPOCHS="${FULL_FINETUNE_EPOCHS:-4}"
OVERWRITE="${OVERWRITE_OUTPUT_DIR:-false}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mixed_wikilingua.sh [options]

Create one shuffled train.jsonl from batch*.jsonl (input/output), an existing
prepared train.jsonl (text/summary), and WikiLingua train. Copy WikiLingua
validation/test unchanged, train once, then evaluate the validation-best model.

  --batch-dir PATH         Directory containing batch*.jsonl (required)
  --batch-pattern GLOB     Batch filename pattern (default: batch*.jsonl)
  --prepared-train PATH    Existing prepared train file (default: datasets/90k/train.jsonl)
  --wikilingua-dir PATH    WikiLingua train/validation/test directory
  --mixed-dir PATH         Directory for the new combined dataset
  --output-dir PATH        Training checkpoints, config, and test predictions
  --batch-source-field NAME   Batch source key (default: input)
  --batch-target-field NAME   Batch target key (default: output)
  --batch-id-field NAME       Batch ID key (default: id)
  --prepared-source-field NAME  Prepared train source key (default: text)
  --prepared-target-field NAME  Prepared train target key (default: summary)
  --prepared-id-field NAME      Prepared train ID key (default: id)
  --seed N                 Shuffle seed (default: 42)
  --epochs N               One-stage full-finetuning epochs (default: 4)
  --train-batch-size N     Batch size per GPU
  --gradient-accumulation N
  --validation-batch-size N
  --eval-batch-size N
  --encoder-model PATH     Local encoder model
  --decoder-model PATH     Local decoder model
  --cuda-visible-devices IDS  One or two GPU IDs, e.g. 0,1
  --overwrite-output-dir  Intentionally rerun in an existing output directory
  -h, --help               Show this message

Other settings, including validation/eval batch sizes and maximum lengths,
use the environment variables accepted by scripts/run_wikilingua.sh.
EOF
}

value() {
  if (($# < 2)) || [[ "$2" == --* ]]; then
    echo "Missing value for $1" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --batch-dir) value "$@"; BATCH_DIR="$2"; shift 2 ;;
    --batch-pattern) value "$@"; BATCH_PATTERN="$2"; shift 2 ;;
    --prepared-train|--other-train|--summary-train) value "$@"; PREPARED_TRAIN="$2"; shift 2 ;;
    --batch-source-field) value "$@"; export BATCH_SOURCE_FIELD="$2"; shift 2 ;;
    --batch-target-field) value "$@"; export BATCH_TARGET_FIELD="$2"; shift 2 ;;
    --batch-id-field) value "$@"; export BATCH_ID_FIELD="$2"; shift 2 ;;
    --prepared-source-field) value "$@"; PREPARED_SOURCE_FIELD="$2"; shift 2 ;;
    --prepared-target-field) value "$@"; PREPARED_TARGET_FIELD="$2"; shift 2 ;;
    --prepared-id-field) value "$@"; PREPARED_ID_FIELD="$2"; shift 2 ;;
    --wikilingua-dir) value "$@"; WIKI_DIR="$2"; shift 2 ;;
    --mixed-dir) value "$@"; MIXED_DIR="$2"; shift 2 ;;
    --output-dir) value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --source-field) value "$@"; PREPARED_SOURCE_FIELD="$2"; shift 2 ;;
    --target-field) value "$@"; PREPARED_TARGET_FIELD="$2"; shift 2 ;;
    --id-field) value "$@"; PREPARED_ID_FIELD="$2"; shift 2 ;;
    --seed) value "$@"; SEED="$2"; shift 2 ;;
    --epochs) value "$@"; EPOCHS="$2"; shift 2 ;;
    --train-batch-size) value "$@"; export TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation) value "$@"; export GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --validation-batch-size) value "$@"; export VALIDATION_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) value "$@"; export EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --encoder-model) value "$@"; export ENCODER_MODEL="$2"; shift 2 ;;
    --decoder-model) value "$@"; export DECODER_MODEL="$2"; shift 2 ;;
    --cuda-visible-devices) value "$@"; export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --overwrite-output-dir) OVERWRITE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo "--epochs must be a positive integer" >&2; exit 2; }
[[ "${SEED}" =~ ^-?[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }
if [[ "${OVERWRITE}" != 1 && "${OVERWRITE}" != true && "${OVERWRITE}" != yes ]] &&
  compgen -G "${OUTPUT_DIR}/*.pt" > /dev/null; then
  echo "Checkpoints already exist in ${OUTPUT_DIR}; choose a new output directory or pass --overwrite-output-dir" >&2
  exit 1
fi

echo "=== Preparing one mixed train split ==="
[[ -n "${BATCH_DIR}" ]] || { echo "Pass --batch-dir PATH for the input/output batch JSONLs" >&2; exit 2; }
"${PYTHON_BIN}" "${ROOT}/scripts/mix_summary_wikilingua.py" \
  --batch-dir "${BATCH_DIR}" \
  --batch-pattern "${BATCH_PATTERN}" \
  --prepared-train "${PREPARED_TRAIN}" \
  --wikilingua-dir "${WIKI_DIR}" \
  --output-dir "${MIXED_DIR}" \
  --batch-source-field "${BATCH_SOURCE_FIELD:-input}" \
  --batch-target-field "${BATCH_TARGET_FIELD:-output}" \
  --batch-id-field "${BATCH_ID_FIELD:-id}" \
  --prepared-source-field "${PREPARED_SOURCE_FIELD}" \
  --prepared-target-field "${PREPARED_TARGET_FIELD}" \
  --prepared-id-field "${PREPARED_ID_FIELD}" \
  --seed "${SEED}"

echo "=== Training once; WikiLingua validation selects best.pt ==="
export WIKILINGUA_DATA_DIR="${MIXED_DIR}"
export AFMR_OUTPUT_DIR="${OUTPUT_DIR}"
export AFMR_CONFIG="${ROOT}/configs/afmr_summary_wikilingua.yaml"
export INTERFACE_WARMUP_EPOCHS=0
export FULL_FINETUNE_EPOCHS="${EPOCHS}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE}"
PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/run_wikilingua.sh"
