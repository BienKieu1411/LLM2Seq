#!/usr/bin/env bash
set -Eeuo pipefail

# Full-fine-tune T5Gemma on the canonical EviSeq ArXiv data.  The default
# recipe is two epochs; all length, optimizer, precision and decoding values
# follow the EviSeq ArXiv contract.  train.sh selects one process or DDP based
# on CUDA_VISIBLE_DEVICES.

CALLER_CWD="$(pwd)"
REQUESTED_SOURCE_DIR="${ARXIV_SOURCE_DIR:-}"
REQUESTED_CONFIG="${T5GEMMA_ARXIV_CONFIG:-}"
REQUESTED_MODEL="${T5GEMMA_MODEL_PATH:-}"
REQUESTED_OVERWRITE="${OVERWRITE_OUTPUT_DIR:-}"
REQUESTED_GPU="${CUDA_VISIBLE_DEVICES-}"
CALLER_SET_CUDA_VISIBLE_DEVICES=false
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
  CALLER_SET_CUDA_VISIBLE_DEVICES=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${PROJECT_ROOT}"

if [[ "${CALLER_SET_CUDA_VISIBLE_DEVICES}" == "true" ]]; then
  export CUDA_VISIBLE_DEVICES="${REQUESTED_GPU}"
fi
if [[ -n "${REQUESTED_SOURCE_DIR}" ]]; then
  ARXIV_SOURCE_DIR="${REQUESTED_SOURCE_DIR}"
fi
if [[ -n "${REQUESTED_CONFIG}" ]]; then
  T5GEMMA_ARXIV_CONFIG="${REQUESTED_CONFIG}"
fi
if [[ -n "${REQUESTED_MODEL}" ]]; then
  T5GEMMA_MODEL_PATH="${REQUESTED_MODEL}"
fi
if [[ -n "${REQUESTED_OVERWRITE}" ]]; then
  OVERWRITE_OUTPUT_DIR="${REQUESTED_OVERWRITE}"
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

CONFIG_TEMPLATE="${T5GEMMA_ARXIV_CONFIG:-T5Gemma/configs/arxiv_full_1b_1b_8192.yaml}"
ARXIV_DATA_DIR="${ARXIV_DATA_DIR:-eviseq_new/datasets/arxiv}"
ARXIV_RAW_DIR="${ARXIV_RAW_DIR:-eviseq_new/datasets/raw/arxiv}"
ARXIV_SOURCE_DIR="${ARXIV_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/arxiv}"
RUN_DIR="${T5GEMMA_ARXIV_RUN_DIR:-runs/t5gemma2_1b_1b_full_arxiv_8192}"
EVAL_DIR="${T5GEMMA_ARXIV_EVAL_DIR:-T5Gemma/eval_outputs/arxiv/1b_1b}"
LOG_DIR="${T5GEMMA_ARXIV_LOG_DIR:-T5Gemma/logs/arxiv}"
TRAIN_BATCH_SIZE="${T5GEMMA_ARXIV_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${T5GEMMA_ARXIV_EVAL_BATCH_SIZE:-8}"
NUM_TRAIN_EPOCHS="${T5GEMMA_ARXIV_EPOCHS:-2}"

if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
else
  GPU_COUNT=1
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
    IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT="${#visible_gpus[@]}"
  fi
  # EviSeq uses a global batch of 96 for ArXiv: 8 examples/card and
  # accumulation 12 on one card or 6 on two cards.
  GRADIENT_ACCUMULATION_STEPS="$((12 / GPU_COUNT))"
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/arxiv_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() {
  echo "ERROR: $*" >&2
  exit 1
}

is_true() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer; got $2"
}

positive_int TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}"
positive_int EVAL_BATCH_SIZE "${EVAL_BATCH_SIZE}"
positive_int NUM_TRAIN_EPOCHS "${NUM_TRAIN_EPOCHS}"
positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
[[ -f "${CONFIG_TEMPLATE}" ]] || die "T5Gemma ArXiv config not found: ${CONFIG_TEMPLATE}"

if [[ ! -s "${ARXIV_DATA_DIR}/train.jsonl" || ! -s "${ARXIV_DATA_DIR}/validation.jsonl" || ! -s "${ARXIV_DATA_DIR}/test.jsonl" ]]; then
  [[ -d "${ARXIV_SOURCE_DIR}" ]] || die "ArXiv raw directory not found: ${ARXIV_SOURCE_DIR}; set ARXIV_SOURCE_DIR"
  prepare_args=(
    --dataset arxiv
    --input-dir "${ARXIV_SOURCE_DIR}"
    --output-dir "${ARXIV_DATA_DIR}"
    --raw-copy-dir "${ARXIV_RAW_DIR}"
  )
  if is_true "${ALLOW_CROSS_SPLIT_CONTENT:-false}"; then
    prepare_args+=(--allow-cross-split-content)
  fi
  if is_true "${ALLOW_DUPLICATE_IDS:-false}"; then
    prepare_args+=(--allow-duplicate-ids)
  fi
  echo "=== Prepare canonical EviSeq ArXiv data ==="
  PYTHON="${PYTHON_BIN}" bash "${PROJECT_ROOT}/eviseq_new/scripts/prepare_afmr.sh" "${prepare_args[@]}"
else
  echo "ArXiv processed data already exists; skipping preparation."
fi

# Materialize the recipe in /tmp so model/data/run overrides do not mutate the
# checked-in config or create an artifact inside an existing run directory.
MATERIALIZED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/t5gemma_arxiv_config.XXXXXX.yaml")"
cleanup() {
  rm -f "${MATERIALIZED_CONFIG}"
}
trap cleanup EXIT

"${PYTHON_BIN}" - \
  "${CONFIG_TEMPLATE}" "${MATERIALIZED_CONFIG}" "${RUN_DIR}" "${ARXIV_DATA_DIR}" \
  "${T5GEMMA_MODEL_PATH:-}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${EVAL_BATCH_SIZE}" \
  "${NUM_TRAIN_EPOCHS}" <<'PY'
import sys
from pathlib import Path

import yaml

(
    template,
    destination,
    run_dir,
    data_dir,
    model_path,
    train_batch,
    accumulation,
    eval_batch,
    epochs,
) = sys.argv[1:]
config = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
config["project"]["output_dir"] = str(Path(run_dir).expanduser().resolve())
config["data"].update(
    {
        "train_file": str((Path(data_dir) / "train.jsonl").expanduser().resolve()),
        "eval_file": str((Path(data_dir) / "validation.jsonl").expanduser().resolve()),
        "validation_file": str((Path(data_dir) / "validation.jsonl").expanduser().resolve()),
        "test_file": str((Path(data_dir) / "test.jsonl").expanduser().resolve()),
    }
)
if model_path:
    config["model"]["model_name_or_path"] = str(Path(model_path).expanduser().resolve())
config["training"].update(
    {
        "num_train_epochs": int(epochs),
        "per_device_train_batch_size": int(train_batch),
        "per_device_eval_batch_size": int(eval_batch),
        "gradient_accumulation_steps": int(accumulation),
    }
)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Materialized T5Gemma ArXiv config: {destination}")
PY

echo "=== T5Gemma ArXiv full fine-tune ==="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all visible devices}"
echo "Epochs: ${NUM_TRAIN_EPOCHS}; batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Config: ${MATERIALIZED_CONFIG}"
echo "Output: ${RUN_DIR}"

train_args=(--config "${MATERIALIZED_CONFIG}")
if is_true "${OVERWRITE_OUTPUT_DIR:-false}"; then
  train_args+=(--overwrite-output-dir)
fi
bash "${T5GEMMA_ROOT}/scripts/train.sh" "${train_args[@]}"

CHECKPOINT="${RUN_DIR}/final_model"
[[ -d "${CHECKPOINT}" ]] || die "Training did not produce ${CHECKPOINT}"

mkdir -p "${EVAL_DIR}"
echo "=== Evaluate T5Gemma ArXiv test ==="
EVAL_GPU="${T5GEMMA_EVAL_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
CUDA_VISIBLE_DEVICES="${EVAL_GPU}" "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
  --config "${MATERIALIZED_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --test_file "${ARXIV_DATA_DIR}/test.jsonl" \
  --output_dir "${EVAL_DIR}" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  2>&1 | tee "${LOG_DIR}/arxiv_eval_$(date +%Y%m%d_%H%M%S).log"

if [[ -n "${PYROUGE_HOME_DIR:-}" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/rouge155/evaluate_rouge.py" \
    "${EVAL_DIR}/predictions.jsonl" \
    --output "${EVAL_DIR}/predictions.rouge155.json"
else
  echo "Perl ROUGE skipped; export PYROUGE_HOME_DIR to calculate ROUGE-1.5.5."
fi

echo "=== T5Gemma ArXiv run completed ==="
echo "Checkpoint: ${CHECKPOINT}"
echo "Predictions: ${EVAL_DIR}/predictions.jsonl"
