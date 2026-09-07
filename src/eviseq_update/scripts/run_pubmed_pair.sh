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
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# Each encoder run uses both GPUs through run_afmr.sh -> torchrun/DDP.
# Effective batch: 4 examples/GPU * 2 GPUs * 4 accumulation steps = 32.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT}/datasets/pubmed}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/datasets/raw/pubmed}"
AFMR_ARCHITECTURE="${AFMR_ARCHITECTURE:-afmr_value_anchor}"
AFMR_GROUNDED_COPY="${AFMR_GROUNDED_COPY:-true}"
AFMR_SEMANTIC_READ="${AFMR_SEMANTIC_READ:-${AFMR_GROUNDED_COPY}}"
[[ "${AFMR_GROUNDED_COPY}" == true || "${AFMR_GROUNDED_COPY}" == false ]] || { echo "AFMR_GROUNDED_COPY must be true or false" >&2; exit 1; }
[[ "${AFMR_SEMANTIC_READ}" == true || "${AFMR_SEMANTIC_READ}" == false ]] || { echo "AFMR_SEMANTIC_READ must be true or false" >&2; exit 1; }
[[ "${AFMR_SEMANTIC_READ}" == false || "${AFMR_GROUNDED_COPY}" == true ]] || { echo "Semantic read requires grounded copy" >&2; exit 1; }
COPY_VARIANT=lm
[[ "${AFMR_GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
[[ "${AFMR_SEMANTIC_READ}" == false ]] || COPY_VARIANT=copy_read
RUN_ROOT="${ROOT}/runs/eviseq_update/pubmed_pair_${AFMR_ARCHITECTURE}_${COPY_VARIANT}"
GENERATED_CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${ROOT}/logs/afmr"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
QWEN_ENCODER="${QWEN_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-Embedding-0.6B}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${GENERATED_CONFIG_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_pair_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" || "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
[[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
[[ -d "${QWEN_ENCODER}" ]] || die "Qwen embedding encoder not found: ${QWEN_ENCODER}"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
[[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "GRADIENT_ACCUMULATION_STEPS must be a positive integer"
[[ "${AFMR_ARCHITECTURE}" == afmr_value_anchor || "${AFMR_ARCHITECTURE}" == afmr_v1 ]] || die "Unsupported AFMR_ARCHITECTURE"

"${PYTHON_BIN}" - <<'PY'
import os

import torch

processes = int(os.environ["NPROC_PER_NODE"])
visible = torch.cuda.device_count()
if not torch.cuda.is_available() or visible < processes:
    raise SystemExit(
        f"Requested {processes} GPU workers, but PyTorch sees {visible} CUDA GPUs. "
        "Check CUDA_VISIBLE_DEVICES and the CUDA-enabled PyTorch installation."
    )
if processes > 1 and not torch.distributed.is_nccl_available():
    raise SystemExit("Multi-GPU training requires PyTorch with NCCL support.")
PY

echo "=== AFMR PubMed sequential benchmark ==="
echo "=== GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Training: ${NPROC_PER_NODE} GPU worker(s); accumulation=${GRADIENT_ACCUMULATION_STEPS} ==="
echo "=== Evaluation: one GPU after each training run ==="
echo "=== Architecture: ${AFMR_ARCHITECTURE}; FP32 updates, BF16 compute ==="
echo "=== Grounded copy: ${AFMR_GROUNDED_COPY} ==="
echo "=== Shared semantic read: ${AFMR_SEMANTIC_READ} ==="
echo "=== Python: ${PYTHON_BIN} ==="
echo "=== Log: ${LOG_FILE} ==="
echo "=== Main run: PPLX encoder -> Qwen3 decoder ==="
echo "=== Control run: Qwen3-Embedding encoder -> Qwen3 decoder ==="

if [[ ! -s "${PROCESSED_DATA_DIR}/train.jsonl" || ! -s "${PROCESSED_DATA_DIR}/validation.jsonl" || ! -s "${PROCESSED_DATA_DIR}/test.jsonl" ]]; then
  [[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
  for split_file in train.label.jsonl val.label.jsonl test.label.jsonl; do
    [[ -s "${PUBMED_SOURCE_DIR}/${split_file}" ]] || die "Missing ${PUBMED_SOURCE_DIR}/${split_file}"
  done
  prepare_args=(
    --dataset pubmed
    --input-dir "${PUBMED_SOURCE_DIR}"
    --output-dir "${PROCESSED_DATA_DIR}"
    --raw-copy-dir "${RAW_DATA_DIR}"
  )
  if [[ "${ALLOW_CROSS_SPLIT_CONTENT:-false}" =~ ^(1|true|yes)$ ]]; then
    prepare_args+=(--allow-cross-split-content)
  fi
  echo "=== Preparing PubMed ==="
  bash "${ROOT}/scripts/prepare_afmr.sh" "${prepare_args[@]}"
else
  echo "=== Prepared PubMed data found; skipping preparation ==="
fi

make_config() {
  local base_config="$1"
  local output_config="$2"
  local encoder_name="$3"
  local output_dir="$4"
"${PYTHON_BIN}" - "${base_config}" "${output_config}" "${encoder_name}" "${DECODER_MODEL}" "${output_dir}" "${PROCESSED_DATA_DIR}" "${AFMR_ARCHITECTURE}" "${AFMR_GROUNDED_COPY}" "${AFMR_SEMANTIC_READ}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

from eviseq_update.config import load_config, validate_config

base, destination, encoder, decoder, output_dir, data_dir, architecture, grounded_copy, semantic_read = sys.argv[1:]
config = load_config(base)
config["architecture"]["name"] = architecture
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = semantic_read == "true"
config.pop("_meta", None)
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["experiment"]["output_dir"] = output_dir
config["training"]["gradient_accumulation_steps"] = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
config["data"]["train_file"] = str(Path(data_dir) / "train.jsonl")
config["data"]["validation_file"] = str(Path(data_dir) / "validation.jsonl")
config["data"]["test_file"] = str(Path(data_dir) / "test.jsonl")
validate_config(config)
effective_batch = config["training"]["batch_size"] * int(os.environ["NPROC_PER_NODE"]) * config["training"]["gradient_accumulation_steps"]
print(f"=== Effective batch: {effective_batch} examples/update ===")
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
}

run_one() {
  local name="$1"
  local encoder="$2"
  local config_path="${GENERATED_CONFIG_DIR}/${name}.yaml"
  local output_dir="${RUN_ROOT}/${name}"
  local predictions="${output_dir}/last_test_predictions.jsonl"

  make_config "${ROOT}/configs/afmr_pubmed.yaml" "${config_path}" "${encoder}" "${output_dir}"
  echo "=== Training ${name} ==="
  train_args=(train "${config_path}")
  if [[ "${OVERWRITE_OUTPUT_DIR}" =~ ^(1|true|yes)$ ]]; then
    train_args+=(--overwrite-output-dir)
  fi
  bash "${ROOT}/scripts/run_afmr.sh" "${train_args[@]}"

  echo "=== Evaluating ${name}: last.pt on PubMed test ==="
  bash "${ROOT}/scripts/run_afmr.sh" evaluate \
    "${config_path}" \
    "${output_dir}/last.pt" \
    "${predictions}" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}"

  if [[ -n "${ROUGE155_SCRIPT:-}" && -f "${ROUGE155_SCRIPT}" ]]; then
    echo "=== Perl ROUGE-1.5.5 for ${name} ==="
    "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${predictions}" --output "${predictions%.jsonl}.rouge155.json"
  else
    echo "=== ROUGE-1.5.5 skipped for ${name}; set ROUGE155_SCRIPT to evaluate it ==="
  fi
}

run_one "pplx" "${PPLX_ENCODER}"
run_one "qwen_embedding" "${QWEN_ENCODER}"

echo "=== PubMed pair completed ==="
echo "PPLX output: ${RUN_ROOT}/pplx"
echo "Qwen embedding output: ${RUN_ROOT}/qwen_embedding"
