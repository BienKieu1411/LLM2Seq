#!/usr/bin/env bash
set -Eeuo pipefail

# Train a plain-summary corpus, continue the same model on WikiLingua, and
# evaluate the WikiLingua validation-best checkpoint.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

SUMMARY_TRAIN="${SUMMARY_TRAIN:-${ROOT}/datasets/90k/train.jsonl}"
WIKILINGUA_DIR="${WIKILINGUA_DIR:-${ROOT}/datasets/wikilingua}"
OUTPUT_ROOT="${SUMMARY_WIKILINGUA_OUTPUT:-${ROOT}/runs/afmr/summary_then_wikilingua}"
TEMPLATE="${SUMMARY_WIKILINGUA_CONFIG:-${ROOT}/configs/afmr_summary_wikilingua.yaml}"
ENCODER_MODEL="${ENCODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
SUMMARY_SOURCE_FIELD="${SUMMARY_SOURCE_FIELD:-text}"
SUMMARY_TARGET_FIELD="${SUMMARY_TARGET_FIELD:-summary}"
SUMMARY_ID_FIELD="${SUMMARY_ID_FIELD:-id}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VALIDATION_NUM_WORKERS="${VALIDATION_NUM_WORKERS:-1}"
SUMMARY_EPOCHS="${SUMMARY_EPOCHS:-4}"
WIKILINGUA_EPOCHS="${WIKILINGUA_EPOCHS:-4}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-3072}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
OVERWRITE_OUTPUT_DIR=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_summary_then_wikilingua.sh [options]

Stages:
  1. Train on the plain-summary JSONL for 4 epochs (train-only).
  2. Resume its last.pt and train 4 more epochs on WikiLingua.
  3. Evaluate WikiLingua best.pt selected by validation loss.

Options:
  --summary-train PATH       Plain-summary training JSONL
  --wikilingua-dir PATH      Directory containing train/validation/test.jsonl
  --output-dir PATH          Root for stage outputs and generated configs
  --encoder-model PATH       Local encoder checkpoint
  --decoder-model PATH       Local decoder checkpoint
  --source-field FIELD       Source field in summary JSONL (default: text)
  --target-field FIELD       Target field in summary JSONL (default: summary)
  --id-field FIELD           ID field in summary JSONL (default: id)
  --train-batch-size N       Batch size per GPU (default: 4)
  --gradient-accumulation N  Accumulation steps (default: 8)
  --validation-batch-size N  WikiLingua validation batch size (default: 4)
  --eval-batch-size N        WikiLingua test batch size (default: 4)
  --summary-epochs N         Summary-corpus epochs (default: 4)
  --wikilingua-epochs N      WikiLingua continuation epochs (default: 4)
  --cuda-visible-devices IDS Comma-separated visible GPU IDs (default: 0)
  --overwrite-output-dir     Clear known artifacts before starting
  -h, --help                 Show this help

The shared recipe uses max source length 3072 and max target/generation 512.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer; got $2"
}

while (($#)); do
  case "$1" in
    --summary-train) SUMMARY_TRAIN="$2"; shift 2 ;;
    --wikilingua-dir) WIKILINGUA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --encoder-model) ENCODER_MODEL="$2"; shift 2 ;;
    --decoder-model) DECODER_MODEL="$2"; shift 2 ;;
    --source-field) SUMMARY_SOURCE_FIELD="$2"; shift 2 ;;
    --target-field) SUMMARY_TARGET_FIELD="$2"; shift 2 ;;
    --id-field) SUMMARY_ID_FIELD="$2"; shift 2 ;;
    --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation) GRADIENT_ACCUMULATION_STEPS="$2"; shift 2 ;;
    --validation-batch-size) VALIDATION_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
    --summary-epochs) SUMMARY_EPOCHS="$2"; shift 2 ;;
    --wikilingua-epochs) WIKILINGUA_EPOCHS="$2"; shift 2 ;;
    --cuda-visible-devices) CUDA_DEVICES="$2"; shift 2 ;;
    --overwrite-output-dir) OVERWRITE_OUTPUT_DIR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "${TEMPLATE}" ]] || die "Recipe config not found: ${TEMPLATE}"
[[ -s "${SUMMARY_TRAIN}" ]] || die "Summary training JSONL not found: ${SUMMARY_TRAIN}"
[[ -d "${WIKILINGUA_DIR}" ]] || die "WikiLingua directory not found: ${WIKILINGUA_DIR}"
for split in train validation test; do
  [[ -s "${WIKILINGUA_DIR}/${split}.jsonl" ]] || die "Missing WikiLingua split: ${WIKILINGUA_DIR}/${split}.jsonl"
done
[[ -d "${ENCODER_MODEL}" ]] || die "Encoder checkpoint not found: ${ENCODER_MODEL}"
[[ -d "${DECODER_MODEL}" ]] || die "Decoder checkpoint not found: ${DECODER_MODEL}"

positive_int TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}"
positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
positive_int VALIDATION_BATCH_SIZE "${VALIDATION_BATCH_SIZE}"
positive_int EVAL_BATCH_SIZE "${EVAL_BATCH_SIZE}"
positive_int NUM_WORKERS "${NUM_WORKERS}"
positive_int VALIDATION_NUM_WORKERS "${VALIDATION_NUM_WORKERS}"
positive_int SUMMARY_EPOCHS "${SUMMARY_EPOCHS}"
positive_int WIKILINGUA_EPOCHS "${WIKILINGUA_EPOCHS}"
positive_int MAX_SOURCE_LENGTH "${MAX_SOURCE_LENGTH}"
positive_int MAX_TARGET_LENGTH "${MAX_TARGET_LENGTH}"
positive_int MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
positive_int MIN_NEW_TOKENS "${MIN_NEW_TOKENS}"

OUTPUT_ROOT="$(mkdir -p "${OUTPUT_ROOT}" && cd "${OUTPUT_ROOT}" && pwd)"
STAGE_SUMMARY="${OUTPUT_ROOT}/summary_pretrain"
STAGE_WIKILINGUA="${OUTPUT_ROOT}/wikilingua_continuation"
CONFIG_DIR="${OUTPUT_ROOT}/configs"
STAGE1_CONFIG="${CONFIG_DIR}/summary_pretrain.yaml"
STAGE2_CONFIG="${CONFIG_DIR}/wikilingua_continuation.yaml"
mkdir -p "${CONFIG_DIR}" "${STAGE_SUMMARY}" "${STAGE_WIKILINGUA}"

GPU_COUNT=1
if [[ "${CUDA_DEVICES}" == *,* ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_DEVICES}"
  GPU_COUNT="${#VISIBLE_GPUS[@]}"
fi
[[ "${GPU_COUNT}" == 1 || "${GPU_COUNT}" == 2 ]] || die "Use one or two visible GPUs"
PRIMARY_GPU="${CUDA_DEVICES%%,*}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

if (( OVERWRITE_OUTPUT_DIR )); then
  find "${STAGE_SUMMARY}" "${STAGE_WIKILINGUA}" -maxdepth 1 -type f \
    \( -name '*.pt' -o -name '*.jsonl' -o -name '*.json' -o -name '*.yaml' \) -delete
fi

"${PYTHON_BIN}" "${ROOT}/scripts/materialize_summary_wikilingua.py" \
  --template "${TEMPLATE}" \
  --summary-train "${SUMMARY_TRAIN}" \
  --wikilingua-dir "${WIKILINGUA_DIR}" \
  --stage1-output "${STAGE_SUMMARY}" \
  --stage2-output "${STAGE_WIKILINGUA}" \
  --stage1-config "${STAGE1_CONFIG}" \
  --stage2-config "${STAGE2_CONFIG}" \
  --encoder-model "${ENCODER_MODEL}" \
  --decoder-model "${DECODER_MODEL}" \
  --source-field "${SUMMARY_SOURCE_FIELD}" \
  --target-field "${SUMMARY_TARGET_FIELD}" \
  --id-field "${SUMMARY_ID_FIELD}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION_STEPS}" \
  --validation-batch-size "${VALIDATION_BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --validation-num-workers "${VALIDATION_NUM_WORKERS}" \
  --summary-epochs "${SUMMARY_EPOCHS}" \
  --wikilingua-epochs "${WIKILINGUA_EPOCHS}" \
  --max-source-length "${MAX_SOURCE_LENGTH}" \
  --max-target-length "${MAX_TARGET_LENGTH}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --min-new-tokens "${MIN_NEW_TOKENS}"

run_train() {
  local config="$1"
  shift
  if (( GPU_COUNT > 1 )); then
    "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone --nproc_per_node="${GPU_COUNT}" \
      "${ROOT}/run_afmr.py" train "${config}" "$@"
  else
    "${PYTHON_BIN}" "${ROOT}/run_afmr.py" train "${config}" "$@"
  fi
}

echo "=== Stage 1/2: plain-summary corpus (${SUMMARY_EPOCHS} epochs) ==="
echo "Train file: ${SUMMARY_TRAIN}"
echo "Output: ${STAGE_SUMMARY}"
stage1_args=(--train-only)
if (( OVERWRITE_OUTPUT_DIR )); then
  stage1_args+=(--overwrite-output-dir)
fi
run_train "${STAGE1_CONFIG}" "${stage1_args[@]}"

SUMMARY_CHECKPOINT="${STAGE_SUMMARY}/last.pt"
[[ -s "${SUMMARY_CHECKPOINT}" ]] || die "Stage 1 did not produce ${SUMMARY_CHECKPOINT}"

echo "=== Stage 2/2: WikiLingua continuation (${WIKILINGUA_EPOCHS} epochs) ==="
echo "Resume checkpoint: ${SUMMARY_CHECKPOINT}"
echo "Output: ${STAGE_WIKILINGUA}"
run_train "${STAGE2_CONFIG}" --resume-checkpoint "${SUMMARY_CHECKPOINT}"

BEST_CHECKPOINT="${STAGE_WIKILINGUA}/best.pt"
RESOLVED_CONFIG="${STAGE_WIKILINGUA}/resolved_config.yaml"
PREDICTIONS="${STAGE_WIKILINGUA}/best_test_predictions.jsonl"
[[ -s "${BEST_CHECKPOINT}" ]] || die "Stage 2 did not produce ${BEST_CHECKPOINT}"
[[ -s "${RESOLVED_CONFIG}" ]] || die "Stage 2 did not produce ${RESOLVED_CONFIG}"
rm -f "${PREDICTIONS}" "${PREDICTIONS}.metrics.json"

echo "=== Evaluating WikiLingua validation-best checkpoint ==="
CUDA_VISIBLE_DEVICES="${PRIMARY_GPU}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
  "${RESOLVED_CONFIG}" "${BEST_CHECKPOINT}" "${PREDICTIONS}" \
  --split test --batch-size "${EVAL_BATCH_SIZE}" --device cuda:0

echo "=== Two-stage run complete ==="
echo "Stage 1 last checkpoint: ${SUMMARY_CHECKPOINT}"
echo "WikiLingua best checkpoint: ${BEST_CHECKPOINT}"
echo "WikiLingua predictions: ${PREDICTIONS}"
echo "WikiLingua metrics: ${PREDICTIONS}.metrics.json"
