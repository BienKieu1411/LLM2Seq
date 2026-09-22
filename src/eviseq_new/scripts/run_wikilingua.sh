#!/usr/bin/env bash
set -Eeuo pipefail

# Train the WikiLingua recipe from pretrained backbones and evaluate the
# validation-best checkpoint on the untouched test split.

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

CONFIG_TEMPLATE="${AFMR_CONFIG:-${ROOT}/configs/afmr_wikilingua.yaml}"
DATA_DIR="${WIKILINGUA_DATA_DIR:-${ROOT}/datasets/wikilingua}"
OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/wikilingua_direct}"
ENCODER_MODEL="${ENCODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VALIDATION_NUM_WORKERS="${VALIDATION_NUM_WORKERS:-1}"
INTERFACE_WARMUP_EPOCHS="${INTERFACE_WARMUP_EPOCHS:-0}"
FULL_FINETUNE_EPOCHS="${FULL_FINETUNE_EPOCHS:-6}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-3072}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-16}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
DEVICE="${DEVICE:-}"

LOG_DIR="${ROOT}/logs/afmr"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/wikilingua_$(date +%Y%m%d_%H%M%S).log"
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

nonnegative_int() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer; got $2"
}

[[ -f "${CONFIG_TEMPLATE}" ]] || die "AFMR config not found: ${CONFIG_TEMPLATE}"
[[ -d "${ENCODER_MODEL}" ]] || die "Encoder not found: ${ENCODER_MODEL}"
[[ -d "${DECODER_MODEL}" ]] || die "Decoder not found: ${DECODER_MODEL}"
for split in train validation test; do
  [[ -s "${DATA_DIR}/${split}.jsonl" ]] || die "Missing WikiLingua split: ${DATA_DIR}/${split}.jsonl"
done

positive_int TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}"
positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
positive_int VALIDATION_BATCH_SIZE "${VALIDATION_BATCH_SIZE}"
positive_int EVAL_BATCH_SIZE "${EVAL_BATCH_SIZE}"
positive_int NUM_WORKERS "${NUM_WORKERS}"
positive_int VALIDATION_NUM_WORKERS "${VALIDATION_NUM_WORKERS}"
nonnegative_int INTERFACE_WARMUP_EPOCHS "${INTERFACE_WARMUP_EPOCHS}"
positive_int FULL_FINETUNE_EPOCHS "${FULL_FINETUNE_EPOCHS}"
positive_int MAX_SOURCE_LENGTH "${MAX_SOURCE_LENGTH}"
positive_int MAX_TARGET_LENGTH "${MAX_TARGET_LENGTH}"
positive_int MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
positive_int MIN_NEW_TOKENS "${MIN_NEW_TOKENS}"

GPU_COUNT=1
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#VISIBLE_GPUS[@]}"
fi
PRIMARY_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
[[ "${GPU_COUNT}" == 1 || "${GPU_COUNT}" == 2 ]] || die "Use one or two visible GPUs"

GENERATED_CONFIG_DIR="${OUTPUT_DIR}/configs"
GENERATED_CONFIG="${GENERATED_CONFIG_DIR}/afmr_wikilingua.yaml"
mkdir -p "${GENERATED_CONFIG_DIR}"

"${PYTHON_BIN}" - \
  "${CONFIG_TEMPLATE}" "${GENERATED_CONFIG}" "${ENCODER_MODEL}" "${DECODER_MODEL}" \
  "${OUTPUT_DIR}" "${DATA_DIR}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${VALIDATION_BATCH_SIZE}" "${NUM_WORKERS}" "${VALIDATION_NUM_WORKERS}" \
  "${INTERFACE_WARMUP_EPOCHS}" "${FULL_FINETUNE_EPOCHS}" "${MAX_SOURCE_LENGTH}" \
  "${MAX_TARGET_LENGTH}" "${EVAL_BATCH_SIZE}" "${MAX_NEW_TOKENS}" "${MIN_NEW_TOKENS}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

(
    template,
    destination,
    encoder,
    decoder,
    output_dir,
    data_dir,
    train_batch,
    accumulation,
    validation_batch,
    workers,
    validation_workers,
    warmup_epochs,
    full_epochs,
    max_source,
    max_target,
    eval_batch,
    max_new,
    min_new,
) = sys.argv[1:]

config = load_config(template)
config.pop("_meta", None)
config["model"]["encoder_name"] = str(Path(encoder).expanduser().resolve())
config["model"]["decoder_name"] = str(Path(decoder).expanduser().resolve())
config["experiment"]["output_dir"] = str(Path(output_dir).expanduser().resolve())
config["data"].update(
    {
        "train_file": str((Path(data_dir) / "train.jsonl").expanduser().resolve()),
        "validation_file": str((Path(data_dir) / "validation.jsonl").expanduser().resolve()),
        "test_file": str((Path(data_dir) / "test.jsonl").expanduser().resolve()),
        "max_source_length": int(max_source),
        "max_target_length": int(max_target),
    }
)
config["training"].update(
    {
        "batch_size": int(train_batch),
        "gradient_accumulation_steps": int(accumulation),
        "validation_batch_size": int(validation_batch),
        "num_workers": int(workers),
        "validation_num_workers": int(validation_workers),
        "interface_warmup_epochs": int(warmup_epochs),
        "full_finetune_epochs": int(full_epochs),
        "save_best": True,
        "save_each_epoch": True,
        "resume_checkpoint": "",
    }
)
config["generation"].update(
    {
        "batch_size": int(eval_batch),
        "max_new_tokens": int(max_new),
        "min_new_tokens": int(min_new),
        "num_beams": 1,
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
    }
)
validate_config(config)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Materialized WikiLingua config: {destination}")
PY

echo "=== WikiLingua direct fine-tuning ==="
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${GPU_COUNT} (DDP when two GPUs are visible)"
echo "Encoder: ${ENCODER_MODEL}"
echo "Decoder: ${DECODER_MODEL}"
echo "Train batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Epochs: warmup=${INTERFACE_WARMUP_EPOCHS}, full=${FULL_FINETUNE_EPOCHS}"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

train_args=(train "${GENERATED_CONFIG}")
if is_true "${OVERWRITE_OUTPUT_DIR}"; then
  train_args+=(--overwrite-output-dir)
fi
if [[ -n "${DEVICE}" ]]; then
  train_args+=(--device "${DEVICE}")
fi

echo "=== Training from pretrained backbones; no resume checkpoint ==="
if (( GPU_COUNT > 1 )); then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${GPU_COUNT}" \
    "${ROOT}/run_afmr.py" "${train_args[@]}"
else
  PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/run_afmr.sh" "${train_args[@]}"
fi

RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"
BEST_CHECKPOINT="${OUTPUT_DIR}/best.pt"
PREDICTIONS="${OUTPUT_DIR}/best_test_predictions.jsonl"
[[ -s "${RESOLVED_CONFIG}" ]] || die "Training did not produce ${RESOLVED_CONFIG}"
[[ -s "${BEST_CHECKPOINT}" ]] || die "Training did not produce ${BEST_CHECKPOINT}; check validation/save_best settings"

"${PYTHON_BIN}" - "${BEST_CHECKPOINT}" <<'PY'
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"Best checkpoint: epoch={state.get('epoch')}, stage={state.get('stage')}, validation_metric={state.get('best_metric')}")
PY

echo "=== Evaluating best.pt on WikiLingua test ==="
CUDA_VISIBLE_DEVICES="${PRIMARY_GPU}" PYTHON="${PYTHON_BIN}" \
  bash "${ROOT}/scripts/run_afmr.sh" evaluate \
  "${RESOLVED_CONFIG}" "${BEST_CHECKPOINT}" "${PREDICTIONS}" \
  --split test --batch-size "${EVAL_BATCH_SIZE}" --device cuda:0

echo "=== WikiLingua run complete ==="
echo "Best checkpoint: ${BEST_CHECKPOINT}"
echo "Predictions: ${PREDICTIONS}"
echo "Metrics: ${PREDICTIONS}.metrics.json"
