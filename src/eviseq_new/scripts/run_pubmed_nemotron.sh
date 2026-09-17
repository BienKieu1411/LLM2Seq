#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare, train and evaluate the AFMR PubMed recipe with a local Nemotron
# embedding checkpoint.  Set CUDA_VISIBLE_DEVICES to one or two devices; the
# wrapper launches one process for one GPU and torchrun/DDP for two GPUs.  It
# never asks Hugging Face to resolve a model name: both model paths must be
# local directories.

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
# Both checkpoints are required to be complete local folders.  This makes a
# missing tokenizer/config fail immediately instead of silently contacting the
# Hugging Face Hub during a long run.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPU_COUNT=1
if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
  GPU_COUNT="${#VISIBLE_GPUS[@]}"
fi
PRIMARY_GPU="${CUDA_VISIBLE_DEVICES%%,*}"

CONFIG_TEMPLATE="${AFMR_CONFIG:-${ROOT}/configs/afmr_pubmed.yaml}"
PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
DATA_DIR="${AFMR_DATA_DIR:-${ROOT}/datasets/pubmed}"
RAW_DATA_DIR="${AFMR_RAW_DATA_DIR:-${ROOT}/datasets/raw/pubmed}"
OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/pubmed_nemotron_contextual_value}"

# Nemotron-3-Embed-1B-BF16 is the default server-side folder used by the
# project.  Override ENCODER_MODEL for llama-nemotron-embed-1b-v2 or another
# local Nemotron checkpoint.  NEMOTRON_ENCODER is accepted as a short alias.
ENCODER_MODEL="${ENCODER_MODEL:-${NEMOTRON_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Nemotron-3-Embed-1B-BF16}}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"

# The 1B encoder is larger than the former PPLX encoder.  These defaults keep
# the same global effective batch (16 x GPU_COUNT x accumulation = 96) while
# reducing activation memory per micro-batch.  Set
# GRADIENT_ACCUMULATION_STEPS explicitly to choose a different effective
# batch.  All resource values are environment overrides.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
if [[ -n "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}"
else
  GRADIENT_ACCUMULATION_STEPS="$((6 / GPU_COUNT))"
fi
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
VALIDATION_NUM_WORKERS="${VALIDATION_NUM_WORKERS:-2}"
INTERFACE_WARMUP_EPOCHS="${INTERFACE_WARMUP_EPOCHS:-1}"
FULL_FINETUNE_EPOCHS="${FULL_FINETUNE_EPOCHS:-3}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-4096}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-32}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
DEVICE="${DEVICE:-}"
ALLOW_CROSS_SPLIT_CONTENT="${ALLOW_CROSS_SPLIT_CONTENT:-false}"
ALLOW_DUPLICATE_IDS="${ALLOW_DUPLICATE_IDS:-false}"
ROUGE155_SCRIPT="${ROUGE155_SCRIPT:-}"

LOG_DIR="${ROOT}/logs/afmr"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_nemotron_$(date +%Y%m%d_%H%M%S).log"
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

[[ "${GPU_COUNT}" == 1 || "${GPU_COUNT}" == 2 ]] || die "Use exactly one or two visible GPUs: CUDA_VISIBLE_DEVICES=0 or 0,1"
[[ -x "${PYTHON_BIN}" || "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
[[ -f "${CONFIG_TEMPLATE}" ]] || die "AFMR config not found: ${CONFIG_TEMPLATE}"
[[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
[[ -d "${ENCODER_MODEL}" ]] || die "Nemotron encoder not found: ${ENCODER_MODEL}; set ENCODER_MODEL to its local folder"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"

positive_int TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}"
positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
positive_int VALIDATION_BATCH_SIZE "${VALIDATION_BATCH_SIZE}"
positive_int EVAL_BATCH_SIZE "${EVAL_BATCH_SIZE}"
positive_int NUM_WORKERS "${NUM_WORKERS}"
positive_int VALIDATION_NUM_WORKERS "${VALIDATION_NUM_WORKERS}"
positive_int INTERFACE_WARMUP_EPOCHS "${INTERFACE_WARMUP_EPOCHS}"
positive_int FULL_FINETUNE_EPOCHS "${FULL_FINETUNE_EPOCHS}"
positive_int MAX_SOURCE_LENGTH "${MAX_SOURCE_LENGTH}"
positive_int MAX_TARGET_LENGTH "${MAX_TARGET_LENGTH}"
positive_int MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
positive_int MIN_NEW_TOKENS "${MIN_NEW_TOKENS}"

if (( MIN_NEW_TOKENS >= MAX_NEW_TOKENS )); then
  die "MIN_NEW_TOKENS must be smaller than MAX_NEW_TOKENS"
fi

if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/validation.jsonl" || ! -s "${DATA_DIR}/test.jsonl" ]]; then
  for split_file in train.label.jsonl val.label.jsonl test.label.jsonl; do
    [[ -s "${PUBMED_SOURCE_DIR}/${split_file}" ]] || die "Missing ${PUBMED_SOURCE_DIR}/${split_file}"
  done
  prepare_args=(
    --dataset pubmed
    --input-dir "${PUBMED_SOURCE_DIR}"
    --output-dir "${DATA_DIR}"
    --raw-copy-dir "${RAW_DATA_DIR}"
  )
  if is_true "${ALLOW_CROSS_SPLIT_CONTENT}"; then
    prepare_args+=(--allow-cross-split-content)
  fi
  if is_true "${ALLOW_DUPLICATE_IDS}"; then
    prepare_args+=(--allow-duplicate-ids)
  fi
  echo "=== Preparing PubMed ==="
  PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/prepare_afmr.sh" "${prepare_args[@]}"
else
  echo "=== Prepared PubMed data found; skipping preparation ==="
fi

GENERATED_CONFIG_DIR="${OUTPUT_DIR}/configs"
GENERATED_CONFIG="${GENERATED_CONFIG_DIR}/afmr_pubmed_nemotron.yaml"
mkdir -p "${GENERATED_CONFIG_DIR}"

PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" - \
  "${CONFIG_TEMPLATE}" "${GENERATED_CONFIG}" "${ENCODER_MODEL}" "${DECODER_MODEL}" \
  "${OUTPUT_DIR}" "${DATA_DIR}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" \
  "${VALIDATION_BATCH_SIZE}" "${NUM_WORKERS}" "${VALIDATION_NUM_WORKERS}" \
  "${INTERFACE_WARMUP_EPOCHS}" "${FULL_FINETUNE_EPOCHS}" "${MAX_SOURCE_LENGTH}" \
  "${MAX_TARGET_LENGTH}" "${EVAL_BATCH_SIZE}" "${MAX_NEW_TOKENS}" "${MIN_NEW_TOKENS}" <<'PY'
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

# Keep all model references local.  AutoConfig/AutoModel will therefore read
# the checkpoint files in these directories instead of resolving a Hub ID.
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
print(f"Materialized local Nemotron PubMed config: {destination}")
PY

echo "=== AFMR PubMed with Nemotron Embed ==="
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Processes: ${GPU_COUNT} (DDP when 2 GPUs are visible)"
echo "Encoder (local): ${ENCODER_MODEL}"
echo "Decoder (local): ${DECODER_MODEL}"
echo "Train batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}; global effective batch: $((TRAIN_BATCH_SIZE * GPU_COUNT * GRADIENT_ACCUMULATION_STEPS)); eval batch: ${EVAL_BATCH_SIZE}"
echo "Source length: ${MAX_SOURCE_LENGTH}; output: ${OUTPUT_DIR}"
echo "Log: ${LOG_FILE}"

if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
  die "Resume checkpoint not found: ${RESUME_CHECKPOINT}"
fi

train_args=(train "${GENERATED_CONFIG}")
if is_true "${OVERWRITE_OUTPUT_DIR}"; then
  train_args+=(--overwrite-output-dir)
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  train_args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [[ -n "${DEVICE}" ]]; then
  train_args+=(--device "${DEVICE}")
fi

echo "=== Training PubMed with Nemotron Embed ==="
if (( GPU_COUNT > 1 )); then
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${GPU_COUNT}" \
    "${ROOT}/run_afmr.py" "${train_args[@]}"
else
  PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/run_afmr.sh" "${train_args[@]}"
fi

CHECKPOINT="${OUTPUT_DIR}/last.pt"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"
PREDICTIONS="${OUTPUT_DIR}/last_test_predictions.jsonl"
[[ -f "${CHECKPOINT}" ]] || die "Training did not produce ${CHECKPOINT}"
[[ -f "${RESOLVED_CONFIG}" ]] || die "Training did not produce ${RESOLVED_CONFIG}"

eval_args=(evaluate "${RESOLVED_CONFIG}" "${CHECKPOINT}" "${PREDICTIONS}" --split test --batch-size "${EVAL_BATCH_SIZE}")
if [[ -n "${DEVICE}" && "${GPU_COUNT}" == 1 ]]; then
  eval_args+=(--device "${DEVICE}")
fi

echo "=== Evaluating last.pt on PubMed test ==="
if (( GPU_COUNT > 1 )); then
  # Inference has no gradient synchronization requirement. Run independent
  # shards concurrently, then merge them by the original dataset index.
  SHARD_ZERO="${PREDICTIONS}.shard0.jsonl"
  SHARD_ONE="${PREDICTIONS}.shard1.jsonl"
  run_eval_shard() {
    local shard_rank="$1"
    local gpu="$2"
    local output="$3"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHON="${PYTHON_BIN}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
      bash "${ROOT}/scripts/run_afmr.sh" evaluate \
      "${RESOLVED_CONFIG}" "${CHECKPOINT}" "${output}" \
      --split test --batch-size "${EVAL_BATCH_SIZE}" \
      --shard-rank "${shard_rank}" --num-shards "${GPU_COUNT}"
  }
  run_eval_shard 0 "${VISIBLE_GPUS[0]}" "${SHARD_ZERO}" &
  PID_ZERO=$!
  run_eval_shard 1 "${VISIBLE_GPUS[1]}" "${SHARD_ONE}" &
  PID_ONE=$!
  STATUS_ZERO=0
  STATUS_ONE=0
  wait "${PID_ZERO}" || STATUS_ZERO=$?
  wait "${PID_ONE}" || STATUS_ONE=$?
  (( STATUS_ZERO == 0 && STATUS_ONE == 0 )) || die "One or more PubMed evaluation shards failed"
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
    "${ROOT}/scripts/merge_eval_shards.py" \
    --output "${PREDICTIONS}" "${SHARD_ZERO}" "${SHARD_ONE}"
else
  PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/run_afmr.sh" "${eval_args[@]}"
fi

if [[ -n "${ROUGE155_SCRIPT}" ]]; then
  [[ -f "${ROUGE155_SCRIPT}" ]] || die "ROUGE155_SCRIPT not found: ${ROUGE155_SCRIPT}"
  echo "=== Perl ROUGE-1.5.5 ==="
  "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${PREDICTIONS}" \
    --output "${PREDICTIONS%.jsonl}.rouge155.json"
fi

echo "=== PubMed Nemotron run completed ==="
echo "Config: ${GENERATED_CONFIG}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Predictions: ${PREDICTIONS}"
