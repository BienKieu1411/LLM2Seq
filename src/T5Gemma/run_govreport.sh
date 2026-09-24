#!/usr/bin/env bash
set -Eeuo pipefail

T5GEMMA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${T5GEMMA_ROOT}/.." && pwd)"
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x /Users/kieugiangbien/bienkieu_env/bin/python ]]; then
  PYTHON_BIN=/Users/kieugiangbien/bienkieu_env/bin/python
else
  PYTHON_BIN=python3
fi
export PYTHON_BIN
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

RUN_MODE="${RUN_MODE:-all}"
CONFIG_TEMPLATE="${T5GEMMA_GOVREPORT_CONFIG:-${T5GEMMA_ROOT}/configs/govreport_full_1b_1b_16384.yaml}"
DATA_DIR="${GOVREPORT_DATA_DIR:-${PROJECT_ROOT}/eviseq_new/datasets/govreport}"
SOURCE_DIR="${GOVREPORT_SOURCE_DIR:-}"
MODEL_PATH="${T5GEMMA_MODEL_PATH:-/workspace/storage-shared/nlp/dungdx4/BERT/t5gemma-2-1b-1b}"
TRAIN_BATCH_SIZE="${T5GEMMA_GOVREPORT_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-96}"
EVAL_BATCH_SIZE="${T5GEMMA_GOVREPORT_EVAL_BATCH_SIZE:-2}"
NUM_TRAIN_EPOCHS="${T5GEMMA_GOVREPORT_EPOCHS:-5}"
MAX_SOURCE_LENGTH="${MAX_SOURCE_LENGTH:-16384}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MIN_NEW_TOKENS="${MIN_NEW_TOKENS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ROUGE155_SCRIPT="${ROUGE155_SCRIPT:-}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer; got $2"
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
  [[ -n "${T5GEMMA_GOVREPORT_RUN_DIR:-}" ]] || die "Set T5GEMMA_GOVREPORT_RUN_DIR for RUN_MODE=eval"
  [[ -d "${T5GEMMA_GOVREPORT_RUN_DIR}" ]] || die "Run directory not found: ${T5GEMMA_GOVREPORT_RUN_DIR}"
else
  T5GEMMA_GOVREPORT_RUN_DIR="${T5GEMMA_GOVREPORT_RUN_DIR:-${PROJECT_ROOT}/runs/t5gemma2_govreport_${GPU_COUNT}gpu_$(date +%Y%m%d_%H%M%S)_$$}"
fi
RUN_DIR="${T5GEMMA_GOVREPORT_RUN_DIR}"
EVAL_DIR="${T5GEMMA_GOVREPORT_EVAL_DIR:-${RUN_DIR}/test_eval}"
LOG_DIR="${T5GEMMA_GOVREPORT_LOG_DIR:-${T5GEMMA_ROOT}/logs/govreport}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/govreport_${GPU_COUNT}gpu_$(date +%Y%m%d_%H%M%S)_$$.log"
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
positive_int EVAL_BATCH_SIZE "${EVAL_BATCH_SIZE}"
positive_int NUM_TRAIN_EPOCHS "${NUM_TRAIN_EPOCHS}"
positive_int MAX_SOURCE_LENGTH "${MAX_SOURCE_LENGTH}"
positive_int MAX_TARGET_LENGTH "${MAX_TARGET_LENGTH}"
positive_int MAX_NEW_TOKENS "${MAX_NEW_TOKENS}"
[[ "${MIN_NEW_TOKENS}" =~ ^[0-9]+$ ]] || die "MIN_NEW_TOKENS must be non-negative"
[[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || die "NUM_WORKERS must be non-negative"
(( MIN_NEW_TOKENS < MAX_NEW_TOKENS )) || die "MIN_NEW_TOKENS must be less than MAX_NEW_TOKENS"

if [[ "${RUN_MODE}" != eval ]]; then
  [[ -f "${CONFIG_TEMPLATE}" ]] || die "Config not found: ${CONFIG_TEMPLATE}"
  if [[ "${RUN_MODE}" != config ]]; then
    [[ -d "${MODEL_PATH}" ]] || die "Local T5Gemma model not found: ${MODEL_PATH}"
    [[ ! -e "${RUN_DIR}" ]] || die "Run directory already exists; choose a fresh T5GEMMA_GOVREPORT_RUN_DIR: ${RUN_DIR}"
  fi
  if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/validation.jsonl" || ! -s "${DATA_DIR}/test.jsonl" ]]; then
    if [[ "${RUN_MODE}" == config ]]; then
      echo "Prepared GovReport data absent; writing config without running training"
    else
      [[ -n "${SOURCE_DIR}" && -d "${SOURCE_DIR}" ]] || die "Prepared GovReport data absent in ${DATA_DIR}; set GOVREPORT_SOURCE_DIR to prepare it"
      prepare_args=(--dataset govreport --input-dir "${SOURCE_DIR}" --output-dir "${DATA_DIR}")
      if is_true "${ALLOW_CROSS_SPLIT_CONTENT:-false}"; then
        prepare_args+=(--allow-cross-split-content)
      fi
      if is_true "${ALLOW_DUPLICATE_IDS:-false}"; then
        prepare_args+=(--allow-duplicate-ids)
      fi
      PYTHON="${PYTHON_BIN}" bash "${PROJECT_ROOT}/eviseq_new/scripts/prepare_afmr.sh" "${prepare_args[@]}"
    fi
  fi

  MATERIALIZED_CONFIG="${T5GEMMA_CONFIG_OUTPUT:-${LOG_DIR}/govreport_config_$(date +%Y%m%d_%H%M%S)_$$.yaml}"
  [[ ! -e "${MATERIALIZED_CONFIG}" ]] || die "Materialized config already exists: ${MATERIALIZED_CONFIG}"
  "${PYTHON_BIN}" - "${CONFIG_TEMPLATE}" "${MATERIALIZED_CONFIG}" "${RUN_DIR}" \
    "${DATA_DIR}" "${MODEL_PATH}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" \
    "${EVAL_BATCH_SIZE}" "${NUM_TRAIN_EPOCHS}" "${MAX_SOURCE_LENGTH}" \
    "${MAX_TARGET_LENGTH}" "${MAX_NEW_TOKENS}" "${MIN_NEW_TOKENS}" "${NUM_WORKERS}" <<'PY'
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
    max_source,
    max_target,
    max_new,
    min_new,
    workers,
) = sys.argv[1:]
config = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
config["project"]["output_dir"] = str(Path(run_dir).expanduser().resolve())
config["model"]["model_name_or_path"] = str(Path(model_path).expanduser().resolve())
config["data"].update(
    {
        "train_file": str((Path(data_dir) / "train.jsonl").expanduser().resolve()),
        "eval_file": str((Path(data_dir) / "validation.jsonl").expanduser().resolve()),
        "validation_file": str((Path(data_dir) / "validation.jsonl").expanduser().resolve()),
        "test_file": str((Path(data_dir) / "test.jsonl").expanduser().resolve()),
        "max_source_length": int(max_source),
        "max_target_length": int(max_target),
    }
)
config["training"].update(
    {
        "num_train_epochs": int(epochs),
        "per_device_train_batch_size": int(train_batch),
        "per_device_eval_batch_size": int(eval_batch),
        "gradient_accumulation_steps": int(accumulation),
        "dataloader_num_workers": int(workers),
    }
)
config["generation"].update(
    {
        "eval_batch_size": int(eval_batch),
        "max_new_tokens": int(max_new),
        "min_new_tokens": int(min_new),
    }
)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Materialized T5Gemma GovReport config: {destination}")
PY

  echo "GPU: ${CUDA_VISIBLE_DEVICES}; train batch/GPU: ${TRAIN_BATCH_SIZE}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}; global effective batch: $((GPU_COUNT * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
  echo "Source/target budget: ${MAX_SOURCE_LENGTH}/${MAX_TARGET_LENGTH}; epochs: ${NUM_TRAIN_EPOCHS}"
  echo "Run: ${RUN_DIR}"
  [[ "${RUN_MODE}" == config ]] && exit 0

  if (( GPU_COUNT > 1 )); then
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${GPU_COUNT}" \
      "${T5GEMMA_ROOT}/scripts/train_full.py" --config "${MATERIALIZED_CONFIG}"
  else
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[0]}" "${PYTHON_BIN}" \
      "${T5GEMMA_ROOT}/scripts/train_full.py" --config "${MATERIALIZED_CONFIG}"
  fi
  [[ -d "${RUN_DIR}/final_model" && -s "${RUN_DIR}/training_config.yaml" ]] || \
    die "Training did not produce final_model and training_config.yaml"
  [[ "${RUN_MODE}" == train ]] && exit 0
fi

CHECKPOINT="${RUN_DIR}/final_model"
EVAL_CONFIG="${RUN_DIR}/training_config.yaml"
[[ -d "${CHECKPOINT}" && -s "${EVAL_CONFIG}" ]] || die "Evaluation needs ${CHECKPOINT} and ${EVAL_CONFIG}"
[[ ! -e "${EVAL_DIR}/predictions.jsonl" ]] || die "Merged predictions already exist; choose a fresh T5GEMMA_GOVREPORT_EVAL_DIR"
mkdir -p "${EVAL_DIR}"

if (( GPU_COUNT == 1 )); then
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[0]}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
      --config "${EVAL_CONFIG}" --checkpoint "${CHECKPOINT}" \
      --output_dir "${EVAL_DIR}" --batch_size "${EVAL_BATCH_SIZE}"
else
  shards=()
  pids=()
  for rank in "${!VISIBLE_GPUS[@]}"; do
    shard_dir="${EVAL_DIR}/shard${rank}"
    mkdir -p "${shard_dir}"
    shards+=("${shard_dir}/predictions.jsonl")
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[${rank}]}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
      "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
        --config "${EVAL_CONFIG}" --checkpoint "${CHECKPOINT}" \
        --output_dir "${shard_dir}" \
        --batch_size "${EVAL_BATCH_SIZE}" --shard-rank "${rank}" --num-shards "${GPU_COUNT}" \
        > "${EVAL_DIR}/shard${rank}.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for rank in "${!pids[@]}"; do
    if ! wait "${pids[${rank}]}"; then
      echo "Evaluation shard ${rank} failed; log: ${EVAL_DIR}/shard${rank}.log" >&2
      tail -n 30 "${EVAL_DIR}/shard${rank}.log" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || die "At least one evaluation shard failed; predictions were not merged"

  "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/merge_eval_shards.py" \
    --output "${EVAL_DIR}/predictions.jsonl" "${shards[@]}"
fi

if [[ -z "${ROUGE155_SCRIPT}" ]]; then
  for candidate in "${PROJECT_ROOT}/evaluation/evaluate_rouge.py" "${PROJECT_ROOT}/rouge155/evaluate_rouge.py"; do
    if [[ -f "${candidate}" ]]; then
      ROUGE155_SCRIPT="${candidate}"
      break
    fi
  done
fi
if [[ -n "${PYROUGE_HOME_DIR:-}" && -n "${ROUGE155_SCRIPT}" ]]; then
  "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${EVAL_DIR}/predictions.jsonl" \
    --output "${EVAL_DIR}/predictions.rouge155.json"
else
  echo "Perl ROUGE skipped; set PYROUGE_HOME_DIR and ROUGE155_SCRIPT if needed"
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Predictions: ${EVAL_DIR}/predictions.jsonl"
echo "Log: ${LOG_FILE}"
