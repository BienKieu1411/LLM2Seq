#!/usr/bin/env bash
set -Eeuo pipefail

# Initialize from a selected checkpoint, then fine-tune only on WikiLingua.
# The WikiLingua validation split selects a new best.pt; test remains untouched
# until the final evaluation.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
RUN_DIR=""
CHECKPOINT="best.pt"
WIKILINGUA_DIR="${WIKILINGUA_DIR:-${ROOT}/datasets/wikilingua}"
OUTPUT_DIR=""
EPOCHS=4
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_BATCH_SIZE=""
GRADIENT_ACCUMULATION=""
VALIDATION_BATCH_SIZE=""
EVAL_BATCH_SIZE=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/finetune_best_wikilingua.sh \
    --run-dir MIXED_RUN_DIR [options]

Initialize from MIXED_RUN_DIR/best.pt, fine-tune on WikiLingua train, select
the new best checkpoint by WikiLingua validation loss, then evaluate it on the
WikiLingua test split. Model weights are reused; optimizer, scheduler, epoch
counter, and best-validation score start fresh for this in-domain stage.

Options:
  --run-dir PATH             Mixed-training run with resolved_config.yaml
  --checkpoint PATH          Checkpoint path or filename in RUN_DIR (default: best.pt)
  --wikilingua-dir PATH      WikiLingua train/validation/test JSONLs
  --output-dir PATH          New output directory (default: runs/afmr/<run>_wikilingua_indomain)
  --epochs N                 WikiLingua fine-tuning epochs (default: 4)
  --gpus IDS                 One or two GPU IDs, e.g. 0 or 0,1
  --train-batch-size N       Override per-GPU training batch size
  --gradient-accumulation N  Override gradient accumulation steps
  --validation-batch-size N  Override validation batch size
  --eval-batch-size N        Override test generation batch size (default: config value)
  -h, --help                 Show this message
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
need_value() { (($# >= 2)) && [[ "$2" != --* ]] || die "Missing value for $1"; }

while (($#)); do
  case "$1" in
    --run-dir) need_value "$@"; RUN_DIR="$2"; shift 2 ;;
    --checkpoint) need_value "$@"; CHECKPOINT="$2"; shift 2 ;;
    --wikilingua-dir) need_value "$@"; WIKILINGUA_DIR="$2"; shift 2 ;;
    --output-dir) need_value "$@"; OUTPUT_DIR="$2"; shift 2 ;;
    --epochs) need_value "$@"; EPOCHS="$2"; shift 2 ;;
    --gpus) need_value "$@"; GPU_IDS="$2"; shift 2 ;;
    --train-batch-size) need_value "$@"; TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --gradient-accumulation) need_value "$@"; GRADIENT_ACCUMULATION="$2"; shift 2 ;;
    --validation-batch-size) need_value "$@"; VALIDATION_BATCH_SIZE="$2"; shift 2 ;;
    --eval-batch-size) need_value "$@"; EVAL_BATCH_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "${RUN_DIR}" ]] || { usage >&2; die "--run-dir is required"; }
[[ -d "${RUN_DIR}" ]] || die "Run directory not found: ${RUN_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
CONFIG="${RUN_DIR}/resolved_config.yaml"
[[ -s "${CONFIG}" ]] || die "Missing resolved config: ${CONFIG}"

if [[ "${CHECKPOINT}" != /* && -s "${RUN_DIR}/${CHECKPOINT}" ]]; then
  CHECKPOINT="${RUN_DIR}/${CHECKPOINT}"
fi
[[ -s "${CHECKPOINT}" ]] || die "Checkpoint not found: ${CHECKPOINT}"
CHECKPOINT="$(cd "$(dirname "${CHECKPOINT}")" && pwd)/$(basename "${CHECKPOINT}")"

[[ -d "${WIKILINGUA_DIR}" ]] || die "WikiLingua directory not found: ${WIKILINGUA_DIR}"
WIKILINGUA_DIR="$(cd "${WIKILINGUA_DIR}" && pwd)"
for split in train validation test; do
  [[ -s "${WIKILINGUA_DIR}/${split}.jsonl" ]] || die "Missing ${WIKILINGUA_DIR}/${split}.jsonl"
done

[[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer"
for item in "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION}" "${VALIDATION_BATCH_SIZE}" "${EVAL_BATCH_SIZE}"; do
  [[ -z "${item}" || "${item}" =~ ^[1-9][0-9]*$ ]] || die "Batch sizes and accumulation must be positive integers"
done
[[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)?$ ]] || die "--gpus accepts one or two comma-separated GPU IDs"
if [[ "${GPU_IDS}" == *,* ]]; then
  GPU_COUNT=2
  GPU0="${GPU_IDS%%,*}"
  GPU1="${GPU_IDS##*,}"
  [[ "${GPU0}" != "${GPU1}" ]] || die "--gpus must list two distinct GPU IDs"
else
  GPU_COUNT=1
  GPU0="${GPU_IDS}"
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="${ROOT}/runs/afmr/$(basename "${RUN_DIR}")_wikilingua_indomain"
elif [[ "${OUTPUT_DIR}" != /* ]]; then
  OUTPUT_DIR="${ROOT}/${OUTPUT_DIR}"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"
[[ "${OUTPUT_DIR}" != "${RUN_DIR}" ]] || die "Output directory must differ from the mixed-training run"
if compgen -G "${OUTPUT_DIR}/*.pt" >/dev/null; then
  die "Checkpoints already exist in ${OUTPUT_DIR}; choose a new output directory"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
GENERATED_CONFIG="${OUTPUT_DIR}/wikilingua_indomain.yaml"

"${PYTHON_BIN}" - \
  "${CONFIG}" "${WIKILINGUA_DIR}" "${OUTPUT_DIR}" "${GENERATED_CONFIG}" "${EPOCHS}" \
  "${TRAIN_BATCH_SIZE}" "${GRADIENT_ACCUMULATION}" "${VALIDATION_BATCH_SIZE}" "${EVAL_BATCH_SIZE}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

(
    source_config,
    wikilingua_dir,
    output_dir,
    destination,
    epochs,
    train_batch,
    accumulation,
    validation_batch,
    eval_batch,
) = sys.argv[1:]

config = load_config(source_config)
config.pop("_meta", None)
wiki = Path(wikilingua_dir).resolve()
config["data"].update(
    {
        "train_file": str(wiki / "train.jsonl"),
        "validation_file": str(wiki / "validation.jsonl"),
        "test_file": str(wiki / "test.jsonl"),
    }
)
config["experiment"].update(
    {"name": "afmr_wikilingua_in_domain", "output_dir": str(Path(output_dir).resolve())}
)
training = config["training"]
training.update(
    {
        "interface_warmup_epochs": 0,
        "full_finetune_epochs": int(epochs),
        "save_best": True,
        "save_each_epoch": True,
        "resume_checkpoint": "",
        "resume_scheduler": False,
        "lr_scheduler": "cosine",
    }
)
if train_batch:
    training["batch_size"] = int(train_batch)
if accumulation:
    training["gradient_accumulation_steps"] = int(accumulation)
if validation_batch:
    training["validation_batch_size"] = int(validation_batch)
if eval_batch:
    config["generation"]["batch_size"] = int(eval_batch)
validate_config(config)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"WikiLingua train: {config['data']['train_file']}")
print(f"WikiLingua validation: {config['data']['validation_file']}")
print(f"epochs={epochs}; scheduler=cosine; save_best=True; best metric=validation loss")
print(f"generated_config={destination}")
PY

echo "=== In-domain WikiLingua fine-tuning ==="
echo "Starting weights: ${CHECKPOINT}"
echo "GPUs: ${GPU_IDS}; epochs: ${EPOCHS}"
echo "Output: ${OUTPUT_DIR}"

if (( GPU_COUNT == 2 )); then
  "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node=2 \
    "${ROOT}/run_afmr.py" train "${GENERATED_CONFIG}" --init-checkpoint "${CHECKPOINT}"
else
  "${PYTHON_BIN}" "${ROOT}/run_afmr.py" train "${GENERATED_CONFIG}" \
    --init-checkpoint "${CHECKPOINT}" --device cuda:0
fi

RESOLVED_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"
BEST_CHECKPOINT="${OUTPUT_DIR}/best.pt"
PREDICTIONS="${OUTPUT_DIR}/best_test_predictions.jsonl"
[[ -s "${RESOLVED_CONFIG}" ]] || die "Training did not produce ${RESOLVED_CONFIG}"
[[ -s "${BEST_CHECKPOINT}" ]] || die "Training did not produce ${BEST_CHECKPOINT}; check WikiLingua validation"

echo "=== Evaluating WikiLingua validation-best checkpoint on test ==="
CUDA_VISIBLE_DEVICES="${GPU0}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
  "${RESOLVED_CONFIG}" "${BEST_CHECKPOINT}" "${PREDICTIONS}" \
  --split test --device cuda:0
