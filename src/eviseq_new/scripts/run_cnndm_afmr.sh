#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
cd "${ROOT}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="python3"
fi

VARIANT="${1:-}"
[[ "$#" -eq 1 ]] || die "Usage: bash scripts/run_cnndm_afmr.sh {seam|vanilla}"
case "${VARIANT}" in
  seam)
    BRIDGE_MODE=afmr
    GROUNDED_COPY=true
    ;;
  vanilla)
    BRIDGE_MODE=direct_projection
    GROUNDED_COPY=false
    ;;
  *)
    die "Usage: bash scripts/run_cnndm_afmr.sh {seam|vanilla}"
    ;;
esac

GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || die "Use exactly one GPU, e.g. GPU_ID=0"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/users/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

DATA_DIR="${CNNDM_DATA_DIR:-${ROOT}/datasets/cnndm}"
ENCODER_MODEL="${ENCODER_MODEL:-${PPLX_ENCODER:-/workspace/storage-shared/nlp/users/dungdx4/BERT/pplx-embed-v1-0.6b}}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-0.6B}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-12}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/afmr/cnndm_${VARIANT}_seed42_$(date +%Y%m%d_%H%M%S)}"

[[ -d "${ENCODER_MODEL}" ]] || die "Encoder not found: ${ENCODER_MODEL}"
[[ -d "${DECODER_MODEL}" ]] || die "Decoder not found: ${DECODER_MODEL}"
for split in train validation test; do
  [[ -s "${DATA_DIR}/${split}.jsonl" ]] || die "Missing ${DATA_DIR}/${split}.jsonl"
done
for value in "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${EVAL_BATCH_SIZE}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "Batch sizes and accumulation must be positive integers"
done

RUN_DIR="$(mkdir -p "${RUN_DIR}" && cd "${RUN_DIR}" && pwd)"
CONFIG_PATH="${RUN_DIR}/input_config.yaml"
LOG_PATH="${RUN_DIR}/run.log"
ROUGE_SCRIPT="${ROUGE155_SCRIPT:-${PROJECT_ROOT}/src/rouge155/evaluate_rouge.py}"
[[ -f "${ROUGE_SCRIPT}" ]] || ROUGE_SCRIPT="${PROJECT_ROOT}/src/evaluation/evaluate_rouge.py"
[[ -f "${ROUGE_SCRIPT}" ]] || die "ROUGE evaluator not found; set ROUGE155_SCRIPT"

"${PYTHON_BIN}" - "${ROOT}" "${CONFIG_PATH}" "${DATA_DIR}" "${RUN_DIR}" \
  "${ENCODER_MODEL}" "${DECODER_MODEL}" "${VARIANT}" "${BRIDGE_MODE}" \
  "${GROUNDED_COPY}" "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

root, config_path, data_dir, run_dir, encoder, decoder, variant, bridge_mode, copy_enabled, batch, accumulation = sys.argv[1:]
config = load_config(Path(root) / "configs/afmr_cnndm.yaml")
config.pop("_meta", None)
config["experiment"]["name"] = f"cnndm_{variant}_seed42"
config["experiment"]["output_dir"] = str(Path(run_dir).resolve())
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["architecture"]["bridge_mode"] = bridge_mode
config["decoder"]["grounded_copy"]["enabled"] = copy_enabled == "true"
config["data"].update(
    {
        "train_file": str(Path(data_dir, "train.jsonl").resolve()),
        "validation_file": str(Path(data_dir, "validation.jsonl").resolve()),
        "test_file": str(Path(data_dir, "test.jsonl").resolve()),
        "max_source_length": 4096,
        "max_target_length": 256,
    }
)
config["training"].update(
    {
        "seed": 42,
        "interface_warmup_epochs": 0,
        "full_finetune_epochs": 3,
        "batch_size": int(batch),
        "gradient_accumulation_steps": int(accumulation),
        "validation_batch_size": 8,
        "save_best": True,
    }
)
config["generation"].update({"batch_size": 8, "max_new_tokens": 256, "min_new_tokens": 8})
validate_config(config)
Path(config_path).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
print(
    f"variant={variant} seed=42 source=4096 target=256 epochs=3 "
    f"batch_per_gpu={batch} accumulation={accumulation} effective_batch={int(batch) * int(accumulation)}"
)
PY

echo "GPU: ${GPU_ID}"
echo "Encoder: ${ENCODER_MODEL}"
echo "Decoder: ${DECODER_MODEL}"
echo "Run: ${RUN_DIR}"
echo "Log: ${LOG_PATH}"

TRAIN_ARGS=(train "${CONFIG_PATH}")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -s "${RESUME_CHECKPOINT}" ]] || die "Resume checkpoint not found: ${RESUME_CHECKPOINT}"
  TRAIN_ARGS+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" "${TRAIN_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"

CHECKPOINT="${RUN_DIR}/best.pt"
[[ -s "${CHECKPOINT}" ]] || die "Training did not produce validation-selected checkpoint ${CHECKPOINT}"
RESOLVED_CONFIG="${RUN_DIR}/resolved_config.yaml"
PREDICTIONS="${RUN_DIR}/best_test_predictions.jsonl"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
  "${RESOLVED_CONFIG}" "${CHECKPOINT}" "${PREDICTIONS}" \
  --split test --batch-size "${EVAL_BATCH_SIZE}" 2>&1 | tee "${RUN_DIR}/evaluate_test.log"
"${PYTHON_BIN}" "${ROUGE_SCRIPT}" "${PREDICTIONS}"

echo "Completed ${VARIANT}; checkpoint: ${CHECKPOINT}"
echo "Predictions: ${PREDICTIONS}"
