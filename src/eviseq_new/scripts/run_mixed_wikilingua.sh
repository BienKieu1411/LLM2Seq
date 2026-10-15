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

SUMMARY_DATA_DIR="${SUMMARY_DATA_DIR:-${BATCH_DIR:-}}"
PREPARED_TRAIN="${PREPARED_TRAIN:-${SUMMARY_TRAIN:-${ROOT}/datasets/90k/train.jsonl}}"
WIKI_DIR="${WIKILINGUA_DIR:-${ROOT}/datasets/wikilingua}"
MIXED_DIR="${MIXED_DATA_DIR:-${ROOT}/datasets/summary_wikilingua_mixed}"
OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/summary_wikilingua_mixed_nemotron_copy}"
ENCODER_MODEL="${ENCODER_MODEL:-${NEMOTRON_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Nemotron-3-Embed-1B-BF16}}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
PREPARED_SOURCE_FIELD="${PREPARED_SOURCE_FIELD:-text}"
PREPARED_TARGET_FIELD="${PREPARED_TARGET_FIELD:-summary}"
PREPARED_ID_FIELD="${PREPARED_ID_FIELD:-id}"
SEED="${MIX_SEED:-42}"
EPOCHS="${FULL_FINETUNE_EPOCHS:-4}"
OVERWRITE="${OVERWRITE_OUTPUT_DIR:-false}"
WIKI_VALIDATION_FRACTION="${WIKILINGUA_VALIDATION_FRACTION:-}"
VALIDATION_FILE="${MIXED_VALIDATION_FILE:-}"
VALIDATION_SOURCE_FIELD="${MIXED_VALIDATION_SOURCE_FIELD:-text}"
VALIDATION_TARGET_FIELD="${MIXED_VALIDATION_TARGET_FIELD:-summary}"
VALIDATION_ID_FIELD="${MIXED_VALIDATION_ID_FIELD:-id}"
REBUILD_MIXED_DATA=false

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mixed_wikilingua.sh [options]

Create one shuffled train.jsonl from sds.jsonl and mds.jsonl (input/abstract_sum), an
existing prepared train.jsonl (text/summary), and WikiLingua train. Icons are
removed from the two summary-data files. By default, copy
WikiLingua validation/test unchanged; optionally derive validation from the
WikiLingua train split. Train once, then evaluate the validation-best model.

  --summary-data-dir PATH  Directory containing sds.jsonl and mds.jsonl (required)
  --batch-dir PATH         Alias for --summary-data-dir
  --prepared-train PATH    Existing prepared train file (default: datasets/90k/train.jsonl)
  --wikilingua-dir PATH    WikiLingua train/test directory (validation also required by default)
  --wikilingua-validation-fraction F  Derive validation from WikiLingua train (e.g. 0.05)
  --validation-file PATH   Use an external validation JSON/JSONL instead
  --validation-source-field NAME  External validation source key (default: text)
  --validation-target-field NAME  External validation target key (default: summary)
  --validation-id-field NAME      External validation ID key (default: id)
  --mixed-dir PATH         Directory for the new combined dataset
  --output-dir PATH        Training checkpoints, config, and test predictions
  --summary-source-field NAME SDS/MDS source key (default: input)
  --summary-target-field NAME SDS/MDS target key (default: abstract_sum)
  --summary-id-field NAME     SDS/MDS ID key (default: id)
  --prepared-source-field NAME  Prepared train source key (default: text)
  --prepared-target-field NAME  Prepared train target key (default: summary)
  --prepared-id-field NAME      Prepared train ID key (default: id)
  --seed N                 Shuffle seed (default: 42)
  --epochs N               One-stage full-finetuning epochs (default: 4)
  --train-batch-size N     Batch size per GPU
  --gradient-accumulation N
  --validation-batch-size N
  --eval-batch-size N
  --encoder-model PATH     Local encoder checkpoint (default: server Nemotron-3-Embed-1B-BF16 path)
  --decoder-model PATH     Local decoder checkpoint (default: server Qwen3-0.6B path)
  --cuda-visible-devices IDS  One or two GPU IDs, e.g. 0,1
  --rebuild-mixed-data     Recreate mixed train/validation/test even if present
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
    --summary-data-dir|--batch-dir) value "$@"; SUMMARY_DATA_DIR="$2"; shift 2 ;;
    --prepared-train|--other-train|--summary-train) value "$@"; PREPARED_TRAIN="$2"; shift 2 ;;
    --summary-source-field|--batch-source-field) value "$@"; export SUMMARY_SOURCE_FIELD="$2"; shift 2 ;;
    --summary-target-field|--batch-target-field) value "$@"; export SUMMARY_TARGET_FIELD="$2"; shift 2 ;;
    --summary-id-field|--batch-id-field) value "$@"; export SUMMARY_ID_FIELD="$2"; shift 2 ;;
    --prepared-source-field) value "$@"; PREPARED_SOURCE_FIELD="$2"; shift 2 ;;
    --prepared-target-field) value "$@"; PREPARED_TARGET_FIELD="$2"; shift 2 ;;
    --prepared-id-field) value "$@"; PREPARED_ID_FIELD="$2"; shift 2 ;;
    --wikilingua-dir) value "$@"; WIKI_DIR="$2"; shift 2 ;;
    --wikilingua-validation-fraction) value "$@"; WIKI_VALIDATION_FRACTION="$2"; shift 2 ;;
    --validation-file) value "$@"; VALIDATION_FILE="$2"; shift 2 ;;
    --validation-source-field) value "$@"; VALIDATION_SOURCE_FIELD="$2"; shift 2 ;;
    --validation-target-field) value "$@"; VALIDATION_TARGET_FIELD="$2"; shift 2 ;;
    --validation-id-field) value "$@"; VALIDATION_ID_FIELD="$2"; shift 2 ;;
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
    --rebuild-mixed-data) REBUILD_MIXED_DATA=true; shift ;;
    --overwrite-output-dir) OVERWRITE=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo "--epochs must be a positive integer" >&2; exit 2; }
[[ "${SEED}" =~ ^-?[0-9]+$ ]] || { echo "--seed must be an integer" >&2; exit 2; }
[[ -d "${ENCODER_MODEL}" ]] || { echo "Encoder not found: ${ENCODER_MODEL}; pass --encoder-model PATH or set ENCODER_MODEL" >&2; exit 1; }
[[ -d "${DECODER_MODEL}" ]] || { echo "Decoder not found: ${DECODER_MODEL}; pass --decoder-model PATH or set DECODER_MODEL" >&2; exit 1; }
if [[ "${OVERWRITE}" != 1 && "${OVERWRITE}" != true && "${OVERWRITE}" != yes ]] &&
  compgen -G "${OUTPUT_DIR}/*.pt" > /dev/null; then
  echo "Checkpoints already exist in ${OUTPUT_DIR}; choose a new output directory or pass --overwrite-output-dir" >&2
  exit 1
fi

mixed_data_ready=true
for split in train validation test; do
  if [[ ! -s "${MIXED_DIR}/${split}.jsonl" ]]; then
    mixed_data_ready=false
    break
  fi
done

if [[ "${mixed_data_ready}" == true && "${REBUILD_MIXED_DATA}" != true ]]; then
  echo "=== Mixed dataset already exists; skipping data preparation: ${MIXED_DIR} ==="
else
  echo "=== Preparing one mixed train split ==="
  [[ -n "${SUMMARY_DATA_DIR}" ]] || { echo "Pass --summary-data-dir PATH (or --batch-dir PATH) to build mixed data" >&2; exit 2; }
  mix_args=( \
    --summary-data-dir "${SUMMARY_DATA_DIR}" \
    --prepared-train "${PREPARED_TRAIN}" \
    --wikilingua-dir "${WIKI_DIR}" \
    --output-dir "${MIXED_DIR}" \
    --summary-source-field "${SUMMARY_SOURCE_FIELD:-${BATCH_SOURCE_FIELD:-input}}" \
    --summary-target-field "${SUMMARY_TARGET_FIELD:-${BATCH_TARGET_FIELD:-abstract_sum}}" \
    --summary-id-field "${SUMMARY_ID_FIELD:-${BATCH_ID_FIELD:-id}}" \
    --prepared-source-field "${PREPARED_SOURCE_FIELD}" \
    --prepared-target-field "${PREPARED_TARGET_FIELD}" \
    --prepared-id-field "${PREPARED_ID_FIELD}" \
    --seed "${SEED}" \
  )
  if [[ -n "${WIKI_VALIDATION_FRACTION}" ]]; then
    mix_args+=(--wikilingua-validation-fraction "${WIKI_VALIDATION_FRACTION}")
  fi
  if [[ -n "${VALIDATION_FILE}" ]]; then
    mix_args+=(
      --validation-file "${VALIDATION_FILE}"
      --validation-source-field "${VALIDATION_SOURCE_FIELD}"
      --validation-target-field "${VALIDATION_TARGET_FIELD}"
      --validation-id-field "${VALIDATION_ID_FIELD}"
    )
  fi
  "${PYTHON_BIN}" "${ROOT}/scripts/mix_summary_wikilingua.py" "${mix_args[@]}"
fi

echo "=== Training once; WikiLingua validation selects best.pt ==="
echo "Encoder: ${ENCODER_MODEL}"
echo "Decoder: ${DECODER_MODEL}"
echo "Objective: token-level CE only (no contrastive loss)"
echo "Grounded copy: enabled"
echo "LR scheduler: cosine"
export WIKILINGUA_DATA_DIR="${MIXED_DIR}"
export AFMR_OUTPUT_DIR="${OUTPUT_DIR}"
export AFMR_CONFIG="${AFMR_CONFIG:-${ROOT}/configs/afmr_summary_wikilingua.yaml}"
export ENCODER_MODEL DECODER_MODEL
export INTERFACE_WARMUP_EPOCHS=0
export FULL_FINETUNE_EPOCHS="${EPOCHS}"
export OVERWRITE_OUTPUT_DIR="${OVERWRITE}"
PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/run_wikilingua.sh"
