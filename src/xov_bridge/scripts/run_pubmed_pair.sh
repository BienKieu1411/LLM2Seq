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

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT}/../eviseq_new/datasets/pubmed}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/../eviseq_new/datasets/raw/pubmed}"
XOV_ARCHITECTURE="${XOV_ARCHITECTURE:-cross_tokenizer_ordered_value}"
XOV_BRIDGE_MODE="${XOV_BRIDGE_MODE:-cross_tokenizer_ordered_value}"
XOV_GROUNDED_COPY="${XOV_GROUNDED_COPY:-true}"
[[ "${XOV_GROUNDED_COPY}" == true || "${XOV_GROUNDED_COPY}" == false ]] || { echo "XOV_GROUNDED_COPY must be true or false" >&2; exit 1; }
XOV_ENCODERS="${XOV_ENCODERS:-pplx}"
XOV_EVAL_TEST="${XOV_EVAL_TEST:-false}"
COPY_VARIANT=lm
[[ "${XOV_GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
BRIDGE_VARIANT=cross_tokenizer_ordered_value
[[ "${XOV_BRIDGE_MODE}" == direct_projection ]] && BRIDGE_VARIANT=direct_projection
RUN_ROOT="${ROOT}/runs/xov/pubmed_pair_${XOV_ARCHITECTURE}_${BRIDGE_VARIANT}_${COPY_VARIANT}"
GENERATED_CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${ROOT}/logs/xov"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
QWEN_ENCODER="${QWEN_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-Embedding-0.6B}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${GENERATED_CONFIG_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_pair_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" || "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
if [[ "${XOV_ENCODERS}" == pplx || "${XOV_ENCODERS}" == both ]]; then
  [[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
fi
if [[ "${XOV_ENCODERS}" == qwen_embedding || "${XOV_ENCODERS}" == both ]]; then
  [[ -d "${QWEN_ENCODER}" ]] || die "Qwen embedding encoder not found: ${QWEN_ENCODER}"
fi
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"
[[ "${XOV_ARCHITECTURE}" == cross_tokenizer_ordered_value ]] || die "XOV_ARCHITECTURE must be cross_tokenizer_ordered_value"
[[ "${XOV_BRIDGE_MODE}" == cross_tokenizer_ordered_value || "${XOV_BRIDGE_MODE}" == direct_projection ]] || die "XOV_BRIDGE_MODE must be cross_tokenizer_ordered_value or direct_projection"
[[ "${XOV_ENCODERS}" == pplx || "${XOV_ENCODERS}" == qwen_embedding || "${XOV_ENCODERS}" == both ]] || die "XOV_ENCODERS must be pplx, qwen_embedding, or both"
[[ "${XOV_EVAL_TEST}" == true || "${XOV_EVAL_TEST}" == false ]] || die "XOV_EVAL_TEST must be true or false"

echo "=== XOV PubMed sequential benchmark ==="
echo "=== GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Architecture: ${XOV_ARCHITECTURE}; FP32 updates, BF16 compute ==="
echo "=== Bridge mode: ${XOV_BRIDGE_MODE} ==="
echo "=== Grounded copy: ${XOV_GROUNDED_COPY} ==="
echo "=== Python: ${PYTHON_BIN} ==="
echo "=== Log: ${LOG_FILE} ==="
echo "=== Encoders: ${XOV_ENCODERS} ==="
echo "=== Test evaluation: ${XOV_EVAL_TEST} (set XOV_EVAL_TEST=true only after locking the run) ==="

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
  bash "${ROOT}/scripts/prepare_xov.sh" "${prepare_args[@]}"
else
  echo "=== Prepared PubMed data found; skipping preparation ==="
fi

make_config() {
  local base_config="$1"
  local output_config="$2"
  local encoder_name="$3"
  local output_dir="$4"
  "${PYTHON_BIN}" - "${base_config}" "${output_config}" "${encoder_name}" "${DECODER_MODEL}" "${output_dir}" "${PROCESSED_DATA_DIR}" "${XOV_ARCHITECTURE}" "${XOV_BRIDGE_MODE}" "${XOV_GROUNDED_COPY}" <<'PY'
import sys
from pathlib import Path

import yaml

from xov.config import load_config

base, destination, encoder, decoder, output_dir, data_dir, architecture, bridge_mode, grounded_copy = sys.argv[1:]
config = load_config(base)
config["architecture"]["name"] = architecture
if bridge_mode == "direct_projection":
    config["architecture"]["bridge_mode"] = bridge_mode
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config.pop("_meta", None)
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["experiment"]["output_dir"] = output_dir
config["data"]["train_file"] = str(Path(data_dir) / "train.jsonl")
config["data"]["validation_file"] = str(Path(data_dir) / "validation.jsonl")
config["data"]["test_file"] = str(Path(data_dir) / "test.jsonl")
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
}

run_one() {
  local name="$1"
  local encoder="$2"
  local config_path="${GENERATED_CONFIG_DIR}/${name}.yaml"
  local output_dir="${RUN_ROOT}/${name}"
  local validation_predictions="${output_dir}/last_validation_predictions.jsonl"
  local predictions="${output_dir}/last_test_predictions.jsonl"

  make_config "${ROOT}/configs/xov_pubmed.yaml" "${config_path}" "${encoder}" "${output_dir}"
  echo "=== Training ${name} ==="
  train_args=(train "${config_path}")
  if [[ "${OVERWRITE_OUTPUT_DIR}" =~ ^(1|true|yes)$ ]]; then
    train_args+=(--overwrite-output-dir)
  fi
  bash "${ROOT}/scripts/run_xov.sh" "${train_args[@]}"

  local eval_config="${output_dir}/resolved_config.yaml"
  [[ -s "${eval_config}" ]] || die "Training did not write ${eval_config}; refuse to evaluate with a different config"
  echo "=== Evaluating ${name}: last.pt on PubMed validation ==="
  bash "${ROOT}/scripts/run_xov.sh" evaluate \
    "${eval_config}" \
    "${output_dir}/last.pt" \
    "${validation_predictions}" \
    --split validation \
    --batch-size "${EVAL_BATCH_SIZE}"
  if [[ -n "${ROUGE155_SCRIPT:-}" && -f "${ROUGE155_SCRIPT}" ]]; then
    echo "=== Perl ROUGE-1.5.5 validation for ${name} ==="
    "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${validation_predictions}" --output "${validation_predictions%.jsonl}.rouge155.json"
  fi

  if [[ "${XOV_EVAL_TEST}" != true ]]; then
    echo "=== Test evaluation skipped by default; set XOV_EVAL_TEST=true to opt in ==="
    return
  fi
  echo "=== Evaluating ${name}: last.pt on PubMed test ==="
  bash "${ROOT}/scripts/run_xov.sh" evaluate \
    "${eval_config}" \
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

if [[ "${XOV_ENCODERS}" == pplx || "${XOV_ENCODERS}" == both ]]; then
  run_one "pplx" "${PPLX_ENCODER}"
fi
if [[ "${XOV_ENCODERS}" == qwen_embedding || "${XOV_ENCODERS}" == both ]]; then
  run_one "qwen_embedding" "${QWEN_ENCODER}"
fi

echo "=== PubMed pair completed ==="
echo "PPLX output: ${RUN_ROOT}/pplx"
echo "Qwen embedding output: ${RUN_ROOT}/qwen_embedding"
