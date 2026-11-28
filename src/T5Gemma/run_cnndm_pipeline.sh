#!/usr/bin/env bash
set -Eeuo pipefail

CALLER_GPU="${CUDA_VISIBLE_DEVICES-}"
CALLER_SET_GPU=false
if [[ "${CUDA_VISIBLE_DEVICES+x}" == "x" ]]; then
  CALLER_SET_GPU=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
if [[ "${CALLER_SET_GPU}" == true ]]; then
  export CUDA_VISIBLE_DEVICES="${CALLER_GPU}"
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
[[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-9]+$ ]] || {
  echo "Use exactly one GPU, e.g. CUDA_VISIBLE_DEVICES=0" >&2
  exit 2
}
export T5GEMMA_NPROC_PER_NODE=1

cd "${PROJECT_ROOT}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/users/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

CONFIG_TEMPLATE="${T5GEMMA_CNNDM_CONFIG:-${T5GEMMA_ROOT}/configs/cnndm_full_1b_1b_4096.yaml}"
CNNDM_DATA_DIR="${CNNDM_DATA_DIR:-${PROJECT_ROOT}/eviseq_new/datasets/cnndm}"
CNNDM_SOURCE_DIR="${CNNDM_SOURCE_DIR:-/workspace/storage-shared/nlp/users/dungdx4/datasets/cnndm}"
MODEL_PATH="${T5GEMMA_MODEL_PATH:-}"
RUN_DIR="${T5GEMMA_CNNDM_RUN_DIR:-runs/t5gemma2_1b_1b_full_cnndm_4096}"
EVAL_DIR="${T5GEMMA_CNNDM_EVAL_DIR:-T5Gemma/eval_outputs/cnndm/1b_1b}"
LOG_DIR="${T5GEMMA_CNNDM_LOG_DIR:-T5Gemma/logs/cnndm}"
TRAIN_BATCH_SIZE="${T5GEMMA_CNNDM_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${T5GEMMA_CNNDM_GRAD_ACCUM:-12}"
EVAL_BATCH_SIZE="${T5GEMMA_CNNDM_EVAL_BATCH_SIZE:-8}"
NUM_TRAIN_EPOCHS="${T5GEMMA_CNNDM_EPOCHS:-3}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -f "${CONFIG_TEMPLATE}" ]] || die "T5Gemma config not found: ${CONFIG_TEMPLATE}"
[[ -n "${MODEL_PATH}" && -d "${MODEL_PATH}" ]] || die "Set T5GEMMA_MODEL_PATH to the local T5Gemma-2-1B-1B checkpoint directory"
CNNDM_DATA_DIR="${CNNDM_DATA_DIR}" CNNDM_SOURCE_DIR="${CNNDM_SOURCE_DIR}" PYTHON="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/eviseq_new/scripts/ensure_cnndm_prepared.sh"
for value in "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${EVAL_BATCH_SIZE}" "${NUM_TRAIN_EPOCHS}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "Epoch and batch settings must be positive integers"
done

mkdir -p "${LOG_DIR}" "${EVAL_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
MATERIALIZED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/t5gemma_cnndm.XXXXXX.yaml")"
cleanup() {
  rm -f "${MATERIALIZED_CONFIG}"
}
trap cleanup EXIT

"${PYTHON_BIN}" - "${CONFIG_TEMPLATE}" "${MATERIALIZED_CONFIG}" "${RUN_DIR}" \
  "${CNNDM_DATA_DIR}" "${MODEL_PATH}" "${TRAIN_BATCH_SIZE}" \
  "${GRADIENT_ACCUMULATION_STEPS}" "${EVAL_BATCH_SIZE}" "${NUM_TRAIN_EPOCHS}" <<'PY'
import sys
from pathlib import Path

import yaml

template, destination, run_dir, data_dir, model_path, batch, accumulation, eval_batch, epochs = sys.argv[1:]
config = yaml.safe_load(Path(template).read_text(encoding="utf-8"))
data = Path(data_dir).expanduser().resolve()
config["project"]["output_dir"] = str(Path(run_dir).expanduser().resolve())
config["model"]["model_name_or_path"] = str(Path(model_path).expanduser().resolve())
config["data"].update(
    {
        "train_file": str(data / "train.jsonl"),
        "eval_file": str(data / "validation.jsonl"),
        "validation_file": str(data / "validation.jsonl"),
        "test_file": str(data / "test.jsonl"),
        "max_source_length": 4096,
        "max_target_length": 256,
    }
)
config["training"].update(
    {
        "num_train_epochs": int(epochs),
        "per_device_train_batch_size": int(batch),
        "per_device_eval_batch_size": int(eval_batch),
        "gradient_accumulation_steps": int(accumulation),
    }
)
config["generation"]["max_new_tokens"] = 256
config["generation"]["min_new_tokens"] = 8
config["generation"]["eval_batch_size"] = int(eval_batch)
Path(destination).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(
    f"source=4096 target=256 epochs={epochs} batch_per_gpu={batch} "
    f"accumulation={accumulation} effective_batch={int(batch) * int(accumulation)}"
)
PY

echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo "Model: ${MODEL_PATH}"
echo "Config: ${MATERIALIZED_CONFIG}"
echo "Run: ${RUN_DIR}"
echo "Log: ${LOG_FILE}"
bash "${T5GEMMA_ROOT}/scripts/train.sh" --config "${MATERIALIZED_CONFIG}" 2>&1 | tee "${LOG_FILE}"

CHECKPOINT="${RUN_DIR}/final_model"
[[ -d "${CHECKPOINT}" ]] || die "Training did not produce ${CHECKPOINT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${T5GEMMA_ROOT}/scripts/evaluate_full_test.py" \
  --config "${MATERIALIZED_CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --test_file "${CNNDM_DATA_DIR}/test.jsonl" \
  --output_dir "${EVAL_DIR}" \
  --batch_size "${EVAL_BATCH_SIZE}" 2>&1 | tee "${LOG_DIR}/eval_$(date +%Y%m%d_%H%M%S).log"

if [[ -f "${PROJECT_ROOT}/rouge155/evaluate_rouge.py" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/rouge155/evaluate_rouge.py" \
    "${EVAL_DIR}/predictions.jsonl" \
    --output "${EVAL_DIR}/predictions.rouge155.json"
fi

echo "Completed T5Gemma CNN/DM run. Checkpoint: ${CHECKPOINT}"
echo "Predictions: ${EVAL_DIR}/predictions.jsonl"
