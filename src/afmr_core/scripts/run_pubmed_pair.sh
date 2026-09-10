#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN="python3"
fi

export PYTHON="${PYTHON_BIN}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# The runner is local-only: a missing checkpoint fails fast instead of fetching it.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-84}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# Keep this switch exported so an optional preparation step can preserve
# repeated examples across splits when the experiment explicitly requests it.
export ALLOW_CROSS_SPLIT_CONTENT="${ALLOW_CROSS_SPLIT_CONTENT:-true}"

CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT}/configs/afmr_pubmed.yaml}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/afmr_core/pubmed_pair}"
RESOLVED_CONFIG="${RESOLVED_CONFIG:-${RUN_ROOT}/resolved_config.yaml}"
CHECKPOINT="${CHECKPOINT:-${RUN_ROOT}/last.pt}"
PREDICTIONS="${PREDICTIONS:-${RUN_ROOT}/test_predictions.jsonl}"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
DATA_DIR="${DATA_DIR:-${ROOT}/datasets/pubmed}"
DRY_RUN="${DRY_RUN:-false}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be positive"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be positive"
[[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "GRADIENT_ACCUMULATION_STEPS must be positive"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be positive"
[[ -f "${CONFIG_TEMPLATE}" ]] || die "Config not found: ${CONFIG_TEMPLATE}"

mkdir -p "${RUN_ROOT}"
"${PYTHON_BIN}" - "${CONFIG_TEMPLATE}" "${RESOLVED_CONFIG}" "${RUN_ROOT}" "${PPLX_ENCODER}" "${DECODER_MODEL}" "${DATA_DIR}" <<'PY'
import sys
from pathlib import Path
import yaml

from afmr_core.config import load_config, validate_config

template, destination, run_root, encoder, decoder, data_dir = sys.argv[1:]
config = load_config(template)
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["experiment"]["output_dir"] = run_root
config["training"]["batch_size"] = int(__import__("os").environ["BATCH_SIZE"])
config["training"]["gradient_accumulation_steps"] = int(__import__("os").environ["GRADIENT_ACCUMULATION_STEPS"])
config["data"]["train_file"] = str(Path(data_dir) / "train.jsonl")
config["data"]["validation_file"] = str(Path(data_dir) / "validation.jsonl")
config["data"]["test_file"] = str(Path(data_dir) / "test.jsonl")
validate_config(config)
Path(destination).parent.mkdir(parents=True, exist_ok=True)
payload = dict(config)
payload.pop("_meta", None)
Path(destination).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"resolved_config={destination}")
print(f"global_batch={config['training']['batch_size'] * int(__import__('os').environ['NPROC_PER_NODE'])}")
print(f"optimizer_updates_per_window={config['training']['gradient_accumulation_steps']}")
print(f"dtype={config['model']['dtype']} compute_dtype={config['model']['compute_dtype']} seed={config['training']['seed']}")
PY

echo "AFMR PubMed run"
echo "config=${RESOLVED_CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} workers=${NPROC_PER_NODE} batch_per_gpu=${BATCH_SIZE} accumulation=${GRADIENT_ACCUMULATION_STEPS}"
echo "effective_examples_per_update=$((BATCH_SIZE * NPROC_PER_NODE * GRADIENT_ACCUMULATION_STEPS))"
echo "allow_cross_split_content=${ALLOW_CROSS_SPLIT_CONTENT}"
echo "checkpoint=${CHECKPOINT}"

case "${DRY_RUN}" in
  true|TRUE|True|1|yes|YES|Yes)
  exit 0
  ;;
esac

for split in train validation test; do
  [[ -s "${DATA_DIR}/${split}.jsonl" ]] || die "Missing prepared PubMed ${split}.jsonl under ${DATA_DIR}"
done
[[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
[[ -d "${DECODER_MODEL}" ]] || die "Decoder model not found: ${DECODER_MODEL}"

if (( NPROC_PER_NODE > 1 )); then
  "${PYTHON_BIN}" - <<'PY'
import os
import torch
n = int(os.environ["NPROC_PER_NODE"])
if not torch.cuda.is_available() or torch.cuda.device_count() < n:
    raise SystemExit(f"requested {n} CUDA workers but PyTorch sees {torch.cuda.device_count()}")
if not torch.distributed.is_nccl_available():
    raise SystemExit("multi-GPU AFMR training requires NCCL")
PY
fi

bash "${ROOT}/run_afmr.sh" train "${RESOLVED_CONFIG}" --overwrite-output-dir

EVAL_DEVICE="${CUDA_VISIBLE_DEVICES%%,*}"
CUDA_VISIBLE_DEVICES="${EVAL_DEVICE}" NPROC_PER_NODE=1 bash "${ROOT}/run_afmr.sh" evaluate \
  "${RESOLVED_CONFIG}" "${CHECKPOINT}" "${PREDICTIONS}" \
  --split test --batch-size "${EVAL_BATCH_SIZE}"
