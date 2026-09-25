#!/usr/bin/env bash
set -Eeuo pipefail

# Mix the two training corpora once, then train once with WikiLingua validation/test.
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

OTHER_TRAIN="${SUMMARY_TRAIN:-${ROOT}/datasets/90k/train.jsonl}"
WIKI_DIR="${WIKILINGUA_DIR:-${ROOT}/datasets/wikilingua}"
MIXED_DIR="${MIXED_DATA_DIR:-${ROOT}/datasets/summary_wikilingua_mixed}"
OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/summary_wikilingua_mixed}"
SOURCE_FIELD="${SUMMARY_SOURCE_FIELD:-text}"
TARGET_FIELD="${SUMMARY_TARGET_FIELD:-summary}"
ID_FIELD="${SUMMARY_ID_FIELD:-id}"
SEED="${MIX_SEED:-42}"
EPOCHS="${FULL_FINETUNE_EPOCHS:-4}"
OVERWRITE="${OVERWRITE_OUTPUT_DIR:-false}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mixed_wikilingua.sh [options]

Create a shuffled train.jsonl from the 90k summary train split and WikiLingua
train split. Copy WikiLingua validation/test unchanged. Train from the local
pretrained backbones once and evaluate the validation-best checkpoint on Wiki test.

  --other-train PATH       Other training JSONL (default: datasets/90k/train.jsonl)
  --wikilingua-dir PATH    WikiLingua train/validation/test directory
  --mixed-dir PATH         Directory for the new combined dataset
  --output-dir PATH        Training checkpoints, config, and test predictions
  --source-field NAME      Source field in other train (default: text)
  --target-field NAME      Target field in other train (default: summary)
  --id-field NAME          ID field in other train (default: id)
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
    --other-train|--summary-train) value "$@"; OTHER_TRAIN="$2"; shift 2 ;;
    --wikilingua-dir) value "$@"; WIKI_DIR="$2"; shift 2 ;;
    --mixed-dir) value "$@"; MIXED_DIR="$2"; shift 2 ;;
    --output-dir) value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --source-field) value "$@"; SOURCE_FIELD="$2"; shift 2 ;;
    --target-field) value "$@"; TARGET_FIELD="$2"; shift 2 ;;
    --id-field) value "$@"; ID_FIELD="$2"; shift 2 ;;
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
"${PYTHON_BIN}" "${ROOT}/scripts/mix_summary_wikilingua.py" \
  --other-train "${OTHER_TRAIN}" \
  --wikilingua-dir "${WIKI_DIR}" \
  --output-dir "${MIXED_DIR}" \
  --source-field "${SOURCE_FIELD}" \
  --target-field "${TARGET_FIELD}" \
  --id-field "${ID_FIELD}" \
  --seed "${SEED}"

echo "=== Training once; WikiLingua validation selects best.pt ==="
export WIKILINGUA_DATA_DIR="${MIXED_DIR}"
export AFMR_OUTPUT_DIR="${OUTPUT_DIR}"
export AFMR_CONFIG="${ROOT}/configs/afmr_summary_wikilingua.yaml"
export INTERFACE_WARMUP_EPOCHS=0
export FULL_FINETUNE_EPOCHS="${EPOCHS}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE}"
PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/run_wikilingua.sh"
