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
PROCESSED_DATA_DIR="${ROOT}/datasets/pubmed"
RAW_DATA_DIR="${ROOT}/datasets/raw/pubmed"
AFMR_ARCHITECTURE="${AFMR_ARCHITECTURE:-afmr_value_anchor}"
AFMR_BRIDGE_MODE="${AFMR_BRIDGE_MODE:-afmr}"
AFMR_GROUNDED_COPY="${AFMR_GROUNDED_COPY:-true}"
AFMR_CONTEXTUAL_VALUE="${AFMR_CONTEXTUAL_VALUE:-false}"
AFMR_SALIENCE_WEIGHT="${AFMR_SALIENCE_WEIGHT:-0}"
AFMR_ENCODERS="${AFMR_ENCODERS:-both}"
[[ "${AFMR_GROUNDED_COPY}" == true || "${AFMR_GROUNDED_COPY}" == false ]] || { echo "AFMR_GROUNDED_COPY must be true or false" >&2; exit 1; }
[[ "${AFMR_CONTEXTUAL_VALUE}" == true || "${AFMR_CONTEXTUAL_VALUE}" == false ]] || { echo "AFMR_CONTEXTUAL_VALUE must be true or false" >&2; exit 1; }
if [[ "${AFMR_BRIDGE_MODE}" == direct_projection || "${AFMR_ARCHITECTURE}" != afmr_value_anchor ]]; then
  AFMR_CONTEXTUAL_VALUE=false
fi
if [[ "${AFMR_BRIDGE_MODE}" == direct_projection ]]; then
  AFMR_SALIENCE_WEIGHT=0
fi
COPY_VARIANT=lm
[[ "${AFMR_GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
BRIDGE_VARIANT=evidence_slots
[[ "${AFMR_BRIDGE_MODE}" == direct_projection ]] && BRIDGE_VARIANT=direct_projection
EVIDENCE_SLOTS=true
[[ "${AFMR_BRIDGE_MODE}" == direct_projection ]] && EVIDENCE_SLOTS=false
if [[ -n "${AFMR_OUTPUT_DIR:-}" ]]; then
  RUN_ROOT="${AFMR_OUTPUT_DIR}"
elif [[ "${AFMR_CONTEXTUAL_VALUE}" == true ]]; then
  RUN_ROOT="${ROOT}/runs/afmr/pubmed_pair_${AFMR_ARCHITECTURE}_local_topdown_value_${COPY_VARIANT}"
elif [[ "${BRIDGE_VARIANT}" == evidence_slots && "${AFMR_SALIENCE_WEIGHT}" != 0 && "${AFMR_SALIENCE_WEIGHT}" != 0.0 ]]; then
  RUN_ROOT="${ROOT}/runs/afmr/pubmed_pair_evidence_slots_auxiliary_prior_${COPY_VARIANT}"
elif [[ "${BRIDGE_VARIANT}" == evidence_slots ]]; then
  RUN_ROOT="${ROOT}/runs/afmr/pubmed_pair_evidence_slots_${COPY_VARIANT}"
else
  RUN_ROOT="${ROOT}/runs/afmr/pubmed_pair_${AFMR_ARCHITECTURE}_${BRIDGE_VARIANT}_${COPY_VARIANT}"
fi
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
[[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
[[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == pplx || "${AFMR_ENCODERS}" == qwen_embedding ]] || die "AFMR_ENCODERS must be both, pplx or qwen_embedding"
if [[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == pplx ]]; then
  [[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
fi
if [[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == qwen_embedding ]]; then
  [[ -d "${QWEN_ENCODER}" ]] || die "Qwen embedding encoder not found: ${QWEN_ENCODER}"
fi
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"
[[ "${AFMR_ARCHITECTURE}" == afmr_value_anchor || "${AFMR_ARCHITECTURE}" == afmr_v1 ]] || die "Unsupported AFMR_ARCHITECTURE"
[[ "${AFMR_BRIDGE_MODE}" == afmr || "${AFMR_BRIDGE_MODE}" == direct_projection ]] || die "AFMR_BRIDGE_MODE must be afmr or direct_projection"

echo "=== AFMR PubMed sequential benchmark ==="
echo "=== GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Architecture: ${AFMR_ARCHITECTURE}; FP32 updates, BF16 compute ==="
echo "=== Bridge mode: ${AFMR_BRIDGE_MODE} ==="
echo "=== Competitive evidence slots: ${EVIDENCE_SLOTS} ==="
echo "=== Grounded copy: ${AFMR_GROUNDED_COPY} ==="
echo "=== Local top-down value bridge: ${AFMR_CONTEXTUAL_VALUE} ==="
echo "=== Salience supervision weight: ${AFMR_SALIENCE_WEIGHT} ==="
echo "=== Python: ${PYTHON_BIN} ==="
echo "=== Log: ${LOG_FILE} ==="
echo "=== Encoders: ${AFMR_ENCODERS} ==="

if [[ ! -s "${PROCESSED_DATA_DIR}/train.jsonl" || ! -s "${PROCESSED_DATA_DIR}/validation.jsonl" || ! -s "${PROCESSED_DATA_DIR}/test.jsonl" ]]; then
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
  "${PYTHON_BIN}" - "${base_config}" "${output_config}" "${encoder_name}" "${DECODER_MODEL}" "${output_dir}" "${PROCESSED_DATA_DIR}" "${AFMR_ARCHITECTURE}" "${AFMR_BRIDGE_MODE}" "${AFMR_GROUNDED_COPY}" "${AFMR_CONTEXTUAL_VALUE}" "${AFMR_SALIENCE_WEIGHT}" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

base, destination, encoder, decoder, output_dir, data_dir, architecture, bridge_mode, grounded_copy, context, salience_weight = sys.argv[1:]
config = load_config(base)
config["architecture"]["name"] = architecture
if bridge_mode == "direct_projection":
    config["architecture"]["bridge_mode"] = bridge_mode
else:
    config["architecture"].pop("bridge_mode", None)
config["architecture"].setdefault("contextual_value", {})["enabled"] = context == "true"
config["training"]["salience_loss_weight"] = 0.0 if bridge_mode == "direct_projection" else float(salience_weight)
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config.pop("_meta", None)
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["experiment"]["output_dir"] = output_dir
config["data"]["train_file"] = str(Path(data_dir) / "train.jsonl")
config["data"]["validation_file"] = str(Path(data_dir) / "validation.jsonl")
config["data"]["test_file"] = str(Path(data_dir) / "test.jsonl")
validate_config(config)
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
  local eval_config="${output_dir}/resolved_config.yaml"
  [[ -s "${eval_config}" ]] || die "Training did not write ${eval_config}; refuse to evaluate with a different config"
  echo "=== Eval config: ${eval_config} ==="
  grep -E "^[[:space:]]*(encoder_name|decoder_name):" "${eval_config}"
  bash "${ROOT}/scripts/run_afmr.sh" evaluate \
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

if [[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == pplx ]]; then
  run_one "pplx" "${PPLX_ENCODER}"
fi
if [[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == qwen_embedding ]]; then
  run_one "qwen_embedding" "${QWEN_ENCODER}"
fi

echo "=== PubMed pair completed ==="
if [[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == pplx ]]; then
  echo "PPLX output: ${RUN_ROOT}/pplx"
fi
if [[ "${AFMR_ENCODERS}" == both || "${AFMR_ENCODERS}" == qwen_embedding ]]; then
  echo "Qwen embedding output: ${RUN_ROOT}/qwen_embedding"
fi
