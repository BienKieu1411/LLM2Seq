#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN=python3
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export AFMR_SERIAL_MODEL_LOAD="${AFMR_SERIAL_MODEL_LOAD:-0}"
export AFMR_TRACE_FIRST_STEP="${AFMR_TRACE_FIRST_STEP:-1}"

RUN_MODE="${RUN_MODE:-train}"
DATASET="${AFMR_DATASET:-}"
case "${DATASET}" in
  govreport)
    CONFIG_TEMPLATE="${AFMR_GOVREPORT_CONFIG:-${ROOT}/configs/afmr_govreport.yaml}"
    DATA_DIR="${GOVREPORT_DATA_DIR:-${ROOT}/datasets/govreport}"
    SOURCE_DIR="${GOVREPORT_SOURCE_DIR:-}"
    SOURCE_DIR_ENV_NAME=GOVREPORT_SOURCE_DIR
    ;;
  booksum)
    CONFIG_TEMPLATE="${AFMR_BOOKSUM_CONFIG:-${ROOT}/configs/afmr_booksum.yaml}"
    DATA_DIR="${BOOKSUM_DATA_DIR:-${ROOT}/datasets/booksum}"
    SOURCE_DIR="${BOOKSUM_SOURCE_DIR:-}"
    SOURCE_DIR_ENV_NAME=BOOKSUM_SOURCE_DIR
    ;;
  *)
    echo "ERROR: AFMR_DATASET must be govreport or booksum" >&2
    exit 1
    ;;
esac
ENCODER_MODEL="${ENCODER_MODEL:-${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-96}"
VALIDATION_BATCH_SIZE="${VALIDATION_BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VALIDATION_NUM_WORKERS="${VALIDATION_NUM_WORKERS:-2}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-1}"
INTERFACE_WARMUP_EPOCHS="${INTERFACE_WARMUP_EPOCHS:-1}"
FULL_FINETUNE_EPOCHS="${FULL_FINETUNE_EPOCHS:-2}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-12288}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-32}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ROUGE155_SCRIPT="${ROUGE155_SCRIPT:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer; got $2"
}

nonnegative_int() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a non-negative integer; got $2"
}

is_true() {
  [[ "$1" == 1 || "$1" == true || "$1" == yes ]]
}

[[ "${RUN_MODE}" == all || "${RUN_MODE}" == train || "${RUN_MODE}" == eval || "${RUN_MODE}" == config ]] || die "RUN_MODE must be all, train, eval, or config"
IFS=',' read -r -a VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#VISIBLE_GPUS[@]}"
(( GPU_COUNT > 0 )) || die "CUDA_VISIBLE_DEVICES must contain at least one GPU ID"
SEEN_GPUS=","
for gpu in "${VISIBLE_GPUS[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "Invalid GPU ID: ${gpu}"
  [[ "${SEEN_GPUS}" != *",${gpu},"* ]] || die "Duplicate GPU ID: ${gpu}"
  SEEN_GPUS="${SEEN_GPUS}${gpu},"
done

if [[ "${RUN_MODE}" == eval ]]; then
  [[ -n "${AFMR_OUTPUT_DIR:-}" ]] || die "Set AFMR_OUTPUT_DIR to the trained run directory for RUN_MODE=eval"
  [[ -d "${AFMR_OUTPUT_DIR}" ]] || die "Trained run directory not found: ${AFMR_OUTPUT_DIR}"
else
  AFMR_OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/${DATASET}_${GPU_COUNT}gpu_$(date +%Y%m%d_%H%M%S)_$$}"
fi
OUTPUT_DIR="${AFMR_OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
LOG_FILE="${OUTPUT_DIR}/run_${DATASET}_${GPU_COUNT}gpu.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python not found: ${PYTHON_BIN}"
positive_int TRAIN_BATCH_SIZE "${TRAIN_BATCH_SIZE}"
positive_int TARGET_GLOBAL_BATCH "${TARGET_GLOBAL_BATCH}"
if [[ -z "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
  global_micro_batch=$((TRAIN_BATCH_SIZE * GPU_COUNT))
  GRADIENT_ACCUMULATION_STEPS=$(((TARGET_GLOBAL_BATCH + global_micro_batch / 2) / global_micro_batch))
  (( GRADIENT_ACCUMULATION_STEPS > 0 )) || GRADIENT_ACCUMULATION_STEPS=1
fi
positive_int GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
positive_int VALIDATION_BATCH_SIZE "${VALIDATION_BATCH_SIZE}"
positive_int EVAL_BATCH_SIZE "${EVAL_BATCH_SIZE}"
positive_int MAX_SOURCE_LENGTH "${MAX_SOURCE_LENGTH}"
positive_int MAX_TARGET_LENGTH "${MAX_TARGET_LENGTH}"
positive_int MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
nonnegative_int MIN_NEW_TOKENS "${MIN_NEW_TOKENS}"
nonnegative_int NUM_WORKERS "${NUM_WORKERS}"
nonnegative_int VALIDATION_NUM_WORKERS "${VALIDATION_NUM_WORKERS}"
positive_int LOG_EVERY_STEPS "${LOG_EVERY_STEPS}"
nonnegative_int INTERFACE_WARMUP_EPOCHS "${INTERFACE_WARMUP_EPOCHS}"
nonnegative_int FULL_FINETUNE_EPOCHS "${FULL_FINETUNE_EPOCHS}"
(( MIN_NEW_TOKENS < MAX_NEW_TOKENS )) || die "MIN_NEW_TOKENS must be less than MAX_NEW_TOKENS"
(( INTERFACE_WARMUP_EPOCHS + FULL_FINETUNE_EPOCHS > 0 )) || die "At least one training epoch is required"

GENERATED_CONFIG="${OUTPUT_DIR}/input_config.yaml"
RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"
CHECKPOINT="${OUTPUT_DIR}/last.pt"
PREDICTIONS="${OUTPUT_DIR}/last_test_predictions.jsonl"

if [[ "${RUN_MODE}" != eval ]]; then
  [[ -f "${CONFIG_TEMPLATE}" ]] || die "Config not found: ${CONFIG_TEMPLATE}"
  if [[ -e "${GENERATED_CONFIG}" && -z "${RESUME_CHECKPOINT}" ]]; then
    die "Input config already exists: ${GENERATED_CONFIG}; use a fresh AFMR_OUTPUT_DIR"
  fi
  if [[ "${RUN_MODE}" != config ]]; then
    [[ -d "${ENCODER_MODEL}" ]] || die "Encoder not found: ${ENCODER_MODEL}"
    [[ -d "${DECODER_MODEL}" ]] || die "Decoder not found: ${DECODER_MODEL}"
  fi
  required_splits=(train)
  if [[ "${RUN_MODE}" != train ]]; then
    required_splits+=(validation test)
  fi
  prepared_data_missing=false
  for split in "${required_splits[@]}"; do
    if [[ ! -s "${DATA_DIR}/${split}.jsonl" ]]; then
      prepared_data_missing=true
      break
    fi
  done
  if [[ "${prepared_data_missing}" == true ]]; then
    if [[ "${RUN_MODE}" == config ]]; then
      echo "Prepared ${DATASET} data absent; writing config without running training"
    else
      [[ -n "${SOURCE_DIR}" && -d "${SOURCE_DIR}" ]] || die "Prepared ${DATASET} data absent in ${DATA_DIR}; set ${SOURCE_DIR_ENV_NAME} to prepare it"
      prepare_args=(--dataset "${DATASET}" --input-dir "${SOURCE_DIR}" --output-dir "${DATA_DIR}")
      if is_true "${ALLOW_CROSS_SPLIT_CONTENT:-false}"; then
        prepare_args+=(--allow-cross-split-content)
      fi
      if is_true "${ALLOW_DUPLICATE_IDS:-false}"; then
        prepare_args+=(--allow-duplicate-ids)
      fi
      PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/prepare_afmr.sh" "${prepare_args[@]}"
    fi
  fi

  "${PYTHON_BIN}" - "${CONFIG_TEMPLATE}" "${GENERATED_CONFIG}" "${ENCODER_MODEL}" \
    "${DECODER_MODEL}" "${OUTPUT_DIR}" "${DATA_DIR}" "${TRAIN_BATCH_SIZE}" \
    "${GRADIENT_ACCUMULATION_STEPS}" "${VALIDATION_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" \
    "${NUM_WORKERS}" "${VALIDATION_NUM_WORKERS}" "${LOG_EVERY_STEPS}" "${INTERFACE_WARMUP_EPOCHS}" \
    "${FULL_FINETUNE_EPOCHS}" "${MAX_SOURCE_LENGTH}" "${MAX_TARGET_LENGTH}" \
    "${MAX_NEW_TOKENS}" "${MIN_NEW_TOKENS}" "${DATASET}" "${RUN_MODE}" <<'PY'
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
    eval_batch,
    workers,
    validation_workers,
    log_every_steps,
    warmup_epochs,
    full_epochs,
    max_source,
    max_target,
    max_new,
    min_new,
    dataset,
    run_mode,
) = sys.argv[1:]

template_data = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
if Path(template).name == f"afmr_{dataset}.yaml" and template_data.get("_base_") != "afmr_base.yaml":
    raise ValueError(
        f"{template} must inherit afmr_base.yaml, not {template_data.get('_base_')!r}; "
        "copy the updated eviseq_new config to the server"
    )
config = load_config(template)
config.pop("_meta", None)
config["experiment"]["output_dir"] = str(Path(output_dir).expanduser().resolve())
config["model"]["encoder_name"] = str(Path(encoder).expanduser().resolve())
config["model"]["decoder_name"] = str(Path(decoder).expanduser().resolve())
config["data"].update(
    {
        f"{split}_file": str((Path(data_dir) / f"{split}.jsonl").expanduser().resolve())
        for split in ("train", "validation", "test")
    }
)
config["data"]["max_source_length"] = int(max_source)
config["data"]["max_target_length"] = int(max_target)
config["training"].update(
    {
        "batch_size": int(train_batch),
        "gradient_accumulation_steps": int(accumulation),
        "validation_batch_size": int(validation_batch),
        "num_workers": int(workers),
        "validation_num_workers": int(validation_workers),
        "log_every_steps": int(log_every_steps),
        "interface_warmup_epochs": int(warmup_epochs),
        "full_finetune_epochs": int(full_epochs),
    }
)
config["generation"].update(
    {
        "batch_size": int(eval_batch),
        "max_new_tokens": int(max_new),
        "min_new_tokens": int(min_new),
    }
)
validate_config(config, train_only=run_mode == "train")
serialized = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
if Path(destination).exists() and Path(destination).read_text(encoding="utf-8") != serialized:
    raise ValueError(f"Existing input config differs from the requested run: {destination}")
Path(destination).write_text(serialized, encoding="utf-8")
print(f"Materialized {dataset} config: {destination}")
PY

  echo "GPU: ${CUDA_VISIBLE_DEVICES}; train batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}; global effective batch: $((GPU_COUNT * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
  echo "Source/target budget: ${MAX_SOURCE_LENGTH}/${MAX_TARGET_LENGTH}; warmup/full epochs: ${INTERFACE_WARMUP_EPOCHS}/${FULL_FINETUNE_EPOCHS}"
  echo "Serial model load: ${AFMR_SERIAL_MODEL_LOAD}; first-step trace: ${AFMR_TRACE_FIRST_STEP}; log interval: ${LOG_EVERY_STEPS} optimizer steps"
  echo "Output: ${OUTPUT_DIR}"
  [[ "${RUN_MODE}" == config ]] && exit 0

  train_args=(train "${GENERATED_CONFIG}")
  if [[ "${RUN_MODE}" == train ]]; then
    train_args+=(--train-only)
  fi
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    [[ -f "${RESUME_CHECKPOINT}" ]] || die "Resume checkpoint not found: ${RESUME_CHECKPOINT}"
    train_args+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
  fi
  if (( GPU_COUNT > 1 )); then
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPU_COUNT}" \
      "${ROOT}/run_afmr.py" "${train_args[@]}"
  else
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[0]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" "${train_args[@]}"
  fi
  [[ -s "${RESOLVED_CONFIG}" && -s "${CHECKPOINT}" ]] || die "Training did not produce resolved_config.yaml and last.pt"
  [[ "${RUN_MODE}" == train ]] && exit 0
fi

[[ -s "${RESOLVED_CONFIG}" && -s "${CHECKPOINT}" ]] || die "Evaluation needs ${RESOLVED_CONFIG} and ${CHECKPOINT}"
if ! is_true "${EVAL_RESUME:-false}"; then
  [[ ! -e "${PREDICTIONS}" ]] || die "Existing predictions found; set EVAL_RESUME=true to resume the same checkpoint"
  for shard_rank in "${!VISIBLE_GPUS[@]}"; do
    [[ ! -e "${OUTPUT_DIR}/last_test_predictions.shard${shard_rank}.jsonl" ]] || \
      die "Existing evaluation shard found; set EVAL_RESUME=true to resume the same checkpoint"
  done
fi
echo "Evaluating ${DATASET} test with ${GPU_COUNT} independent GPU shards"
if (( GPU_COUNT == 1 )); then
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[0]}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
      "${RESOLVED_CONFIG}" "${CHECKPOINT}" "${PREDICTIONS}" \
      --split test --batch-size "${EVAL_BATCH_SIZE}"
else
  shards=()
  pids=()
  for shard_rank in "${!VISIBLE_GPUS[@]}"; do
    shard="${OUTPUT_DIR}/last_test_predictions.shard${shard_rank}.jsonl"
    shards+=("${shard}")
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[${shard_rank}]}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
      "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
        "${RESOLVED_CONFIG}" "${CHECKPOINT}" "${shard}" \
        --split test --batch-size "${EVAL_BATCH_SIZE}" \
        --shard-rank "${shard_rank}" --num-shards "${GPU_COUNT}" \
        > "${OUTPUT_DIR}/eval_shard${shard_rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for shard_rank in "${!pids[@]}"; do
    if ! wait "${pids[${shard_rank}]}"; then
      echo "Evaluation shard ${shard_rank} failed; log: ${OUTPUT_DIR}/eval_shard${shard_rank}.log" >&2
      tail -n 30 "${OUTPUT_DIR}/eval_shard${shard_rank}.log" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || die "At least one evaluation shard failed; predictions were not merged"

  "${PYTHON_BIN}" "${ROOT}/scripts/merge_eval_shards.py" --output "${PREDICTIONS}" "${shards[@]}"
fi

if [[ -z "${ROUGE155_SCRIPT}" ]]; then
  for candidate in "${ROOT}/../evaluation/evaluate_rouge.py" "${ROOT}/../rouge155/evaluate_rouge.py"; do
    if [[ -f "${candidate}" ]]; then
      ROUGE155_SCRIPT="${candidate}"
      break
    fi
  done
fi
if [[ -n "${ROUGE155_SCRIPT}" ]]; then
  [[ -f "${ROUGE155_SCRIPT}" ]] || die "ROUGE155_SCRIPT not found: ${ROUGE155_SCRIPT}"
  "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${PREDICTIONS}" \
    --output "${PREDICTIONS%.jsonl}.rouge155.json"
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Predictions: ${PREDICTIONS}"
echo "Log: ${LOG_FILE}"
