#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
else
  PYTHON_BIN=python3
fi
export PYTHON="${PYTHON_BIN}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT}/datasets/pubmed}"
RAW_DATA_DIR="${ROOT}/datasets/raw/pubmed"
BRIDGE_MODE="${AFMR_BRIDGE_MODE:-side_memory}"
GROUNDED_COPY="${AFMR_GROUNDED_COPY:-true}"
SIDE_TOKENS="${AFMR_SIDE_TOKENS:-24}"
SIDE_EVERY="${AFMR_SIDE_ATTENTION_EVERY:-4}"
SIDE_CAP="${AFMR_SIDE_RMS_CAP:-0.10}"
AFMR_ENCODERS="${AFMR_ENCODERS:-pplx}"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
QWEN_ENCODER="${QWEN_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-Embedding-0.6B}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"

COPY_VARIANT=lm
[[ "${GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
RUN_ROOT="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr_side_memory/pubmed_${BRIDGE_MODE}_${COPY_VARIANT}}"
GENERATED_CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${ROOT}/logs/afmr_side_memory"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${GENERATED_CONFIG_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_pair_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "${BRIDGE_MODE}" == side_memory || "${BRIDGE_MODE}" == direct_projection ]] || die "AFMR_BRIDGE_MODE must be side_memory or direct_projection"
[[ "${GROUNDED_COPY}" == true || "${GROUNDED_COPY}" == false ]] || die "AFMR_GROUNDED_COPY must be true or false"
[[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == pplx || "${AFMR_ENCODERS}" == qwen_embedding ]] || die "AFMR_ENCODERS must be both, pplx or qwen_embedding"
[[ "${SIDE_TOKENS}" =~ ^(1[6-9]|2[0-9]|3[0-2])$ ]] || die "AFMR_SIDE_TOKENS must be in [16,32]"
[[ "${SIDE_EVERY}" =~ ^[1-9][0-9]*$ ]] || die "AFMR_SIDE_ATTENTION_EVERY must be positive"
[[ -d "${DECODER_MODEL}" ]] || die "decoder not found: ${DECODER_MODEL}"

echo "=== Independent gated side-memory PubMed benchmark ==="
echo "=== GPUs: ${CUDA_VISIBLE_DEVICES}; bridge=${BRIDGE_MODE}; side_tokens=${SIDE_TOKENS}; every=${SIDE_EVERY}; cap=${SIDE_CAP} ==="
echo "=== Base exact-token path and grounded-copy path remain unchanged ==="
echo "=== Objective: CE only; parameters FP32; BF16 autocast ==="

if [[ ! -s "${PROCESSED_DATA_DIR}/train.jsonl" || ! -s "${PROCESSED_DATA_DIR}/validation.jsonl" || ! -s "${PROCESSED_DATA_DIR}/test.jsonl" ]]; then
  [[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
  prepare_args=(--dataset pubmed --input-dir "${PUBMED_SOURCE_DIR}" --output-dir "${PROCESSED_DATA_DIR}" --raw-copy-dir "${RAW_DATA_DIR}")
  if [[ "${ALLOW_CROSS_SPLIT_CONTENT:-false}" =~ ^(1|true|yes)$ ]]; then
    prepare_args+=(--allow-cross-split-content)
  fi
  bash "${ROOT}/scripts/prepare_afmr.sh" "${prepare_args[@]}"
fi

make_config() {
  local destination="$1" encoder="$2" output_dir="$3"
  "${PYTHON_BIN}" - "${ROOT}/configs/afmr_pubmed.yaml" "${destination}" "${encoder}" "${output_dir}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

from afmr_side_memory.config import load_config, validate_config

base, destination, encoder, output_dir = sys.argv[1:]
config = load_config(base)
config.pop("_meta", None)
config["architecture"]["bridge_mode"] = os.environ["BRIDGE_MODE"]
config["architecture"]["side_tokens"] = int(os.environ["SIDE_TOKENS"])
config["decoder"]["side_attention_every"] = int(os.environ["SIDE_EVERY"])
config["decoder"]["side_relative_rms_cap"] = float(os.environ["SIDE_CAP"])
config["decoder"]["grounded_copy"]["enabled"] = os.environ["GROUNDED_COPY"] == "true"
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = os.environ["DECODER_MODEL"]
config["experiment"]["output_dir"] = output_dir
data_dir = Path(os.environ["PROCESSED_DATA_DIR"])
for split in ("train", "validation", "test"):
    config["data"][f"{split}_file"] = str(data_dir / f"{split}.jsonl")
validate_config(config)
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
}
export BRIDGE_MODE GROUNDED_COPY SIDE_TOKENS SIDE_EVERY SIDE_CAP DECODER_MODEL PROCESSED_DATA_DIR

run_one() {
  local name="$1" encoder="$2" config_path="${GENERATED_CONFIG_DIR}/$1.yaml" output_dir="${RUN_ROOT}/$1"
  [[ -d "${encoder}" ]] || die "encoder not found: ${encoder}"
  make_config "${config_path}" "${encoder}" "${output_dir}"
  train_args=(train "${config_path}")
  [[ "${OVERWRITE_OUTPUT_DIR}" =~ ^(1|true|yes)$ ]] && train_args+=(--overwrite-output-dir)
  bash "${ROOT}/scripts/run_afmr.sh" "${train_args[@]}"
  local eval_config="${output_dir}/resolved_config.yaml" predictions="${output_dir}/last_test_predictions.jsonl"
  bash "${ROOT}/scripts/run_afmr.sh" evaluate "${eval_config}" "${output_dir}/last.pt" "${predictions}" --split test --batch-size "${EVAL_BATCH_SIZE}"
  if [[ -n "${ROUGE155_SCRIPT:-}" && -f "${ROUGE155_SCRIPT}" ]]; then
    "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${predictions}" --output "${predictions%.jsonl}.rouge155.json"
  fi
}

[[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == pplx ]] && run_one pplx "${PPLX_ENCODER}"
[[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == qwen_embedding ]] && run_one qwen_embedding "${QWEN_ENCODER}"
echo "=== Completed: ${RUN_ROOT} ==="
