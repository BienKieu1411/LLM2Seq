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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
BATCH_SIZE="${BATCH_SIZE:-84}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
ATTENTION_IMPLEMENTATION="${ATTENTION_IMPLEMENTATION:-sdpa}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${ROOT}/datasets/pubmed"
RAW_DATA_DIR="${ROOT}/datasets/raw/pubmed"
AFMR_ARCHITECTURE="${AFMR_ARCHITECTURE:-afmr_value_anchor}"
AFMR_GROUNDED_COPY="${AFMR_GROUNDED_COPY:-true}"
AFMR_COPY_KEY_DIM="${AFMR_COPY_KEY_DIM:-256}"
AFMR_REGION_ROUTER="${AFMR_REGION_ROUTER:-true}"
[[ "${AFMR_GROUNDED_COPY}" == true || "${AFMR_GROUNDED_COPY}" == false ]] || { echo "AFMR_GROUNDED_COPY must be true or false" >&2; exit 1; }
[[ "${AFMR_COPY_KEY_DIM}" =~ ^[1-9][0-9]*$ ]] || { echo "AFMR_COPY_KEY_DIM must be a positive integer" >&2; exit 1; }
[[ "${AFMR_REGION_ROUTER}" == true || "${AFMR_REGION_ROUTER}" == false ]] || { echo "AFMR_REGION_ROUTER must be true or false" >&2; exit 1; }
COPY_VARIANT=lm
[[ "${AFMR_GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
ROUTER_VARIANT=static
[[ "${AFMR_REGION_ROUTER}" == false ]] || ROUTER_VARIANT=region
RUN_ROOT="${ROOT}/runs/grounded_summary/pubmed_pair_${AFMR_ARCHITECTURE}_${COPY_VARIANT}_${ROUTER_VARIANT}_k${AFMR_COPY_KEY_DIM}"
GENERATED_CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${ROOT}/logs/grounded_summary"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
QWEN_ENCODER="${QWEN_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-Embedding-0.6B}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-true}"
EVAL_DO_SAMPLE="${EVAL_DO_SAMPLE:-false}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.0}"
EVAL_TOP_K="${EVAL_TOP_K:-0}"
EVAL_TOP_P="${EVAL_TOP_P:-1.0}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${PYTHON_BIN%/*}/torchrun}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${GENERATED_CONFIG_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_pair_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" || "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
[[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
[[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
[[ -d "${QWEN_ENCODER}" ]] || die "Qwen embedding encoder not found: ${QWEN_ENCODER}"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"
[[ "${EVAL_DO_SAMPLE}" == true || "${EVAL_DO_SAMPLE}" == false ]] || die "EVAL_DO_SAMPLE must be true or false"
[[ "${EVAL_TEMPERATURE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "EVAL_TEMPERATURE must be non-negative"
if [[ "${EVAL_DO_SAMPLE}" == true ]]; then
  (( $(printf '%s\n' "${EVAL_TEMPERATURE}" | awk '{print ($1 > 0)}') )) || die "EVAL_TEMPERATURE must be positive when EVAL_DO_SAMPLE=true"
fi
[[ "${EVAL_TOP_K}" =~ ^[0-9]+$ ]] || die "EVAL_TOP_K must be non-negative"
[[ "${EVAL_TOP_P}" =~ ^(0|1|0\.[0-9]+|1\.0+)$ ]] || die "EVAL_TOP_P must lie in (0,1]"
(( $(printf '%s\n' "${EVAL_TOP_P}" | awk '{print ($1 > 0 && $1 <= 1)}') )) || die "EVAL_TOP_P must lie in (0,1]"
[[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || die "NPROC_PER_NODE must be a positive integer"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer"
[[ "${GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]] || die "GRADIENT_ACCUMULATION_STEPS must be a positive integer"
[[ "${ATTENTION_IMPLEMENTATION}" == sdpa || "${ATTENTION_IMPLEMENTATION}" == flash_attention_2 || "${ATTENTION_IMPLEMENTATION}" == eager ]] || die "ATTENTION_IMPLEMENTATION must be sdpa, flash_attention_2, or eager"
[[ "${AFMR_ARCHITECTURE}" == afmr_value_anchor || "${AFMR_ARCHITECTURE}" == afmr_shared_memory ]] || die "Unsupported AFMR_ARCHITECTURE"
if (( NPROC_PER_NODE > 1 )) && [[ ! -x "${TORCHRUN_BIN}" ]]; then
  TORCHRUN_BIN="$(command -v torchrun 2>/dev/null || true)"
fi
if (( NPROC_PER_NODE > 1 )) && [[ -z "${TORCHRUN_BIN}" ]]; then
  die "torchrun not found; install PyTorch in the selected environment"
fi

echo "=== Grounded Summary PubMed DDP benchmark ==="
echo "=== GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; processes=${NPROC_PER_NODE} ==="
echo "=== Per-GPU train batch=${BATCH_SIZE}; accumulation=${GRADIENT_ACCUMULATION_STEPS}; effective batch=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NPROC_PER_NODE)) ==="
echo "=== Architecture: ${AFMR_ARCHITECTURE}; FP32 updates, BF16 compute ==="
echo "=== Attention implementation: ${ATTENTION_IMPLEMENTATION} ==="
echo "=== Eval sampling: do_sample=${EVAL_DO_SAMPLE}; temperature=${EVAL_TEMPERATURE}; top_k=${EVAL_TOP_K}; top_p=${EVAL_TOP_P} ==="
echo "=== Grounded copy: ${AFMR_GROUNDED_COPY} ==="
echo "=== Grounded-copy key dimension: ${AFMR_COPY_KEY_DIM} ==="
echo "=== Query-conditioned region router: ${AFMR_REGION_ROUTER} ==="
echo "=== Python: ${PYTHON_BIN} ==="
echo "=== Log: ${LOG_FILE} ==="
echo "=== Main run: PPLX encoder -> Qwen3 decoder ==="
echo "=== Control run: Qwen3-Embedding encoder -> Qwen3 decoder ==="

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
  PYTHON="${PYTHON_BIN}" bash "${ROOT}/scripts/prepare_afmr.sh" "${prepare_args[@]}"
else
  echo "=== Prepared PubMed data found; skipping preparation ==="
fi

run_afmr() {
  if (( NPROC_PER_NODE > 1 )); then
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "${ROOT}/run_afmr.py" "$@"
  else
    "${PYTHON_BIN}" "${ROOT}/run_afmr.py" "$@"
  fi
}

make_config() {
  local base_config="$1"
  local output_config="$2"
  local encoder_name="$3"
  local output_dir="$4"
"${PYTHON_BIN}" - "${base_config}" "${output_config}" "${encoder_name}" "${DECODER_MODEL}" "${output_dir}" "${PROCESSED_DATA_DIR}" "${AFMR_ARCHITECTURE}" "${AFMR_GROUNDED_COPY}" "${AFMR_COPY_KEY_DIM}" "${AFMR_REGION_ROUTER}" "${BATCH_SIZE}" "${GRADIENT_ACCUMULATION_STEPS}" "${ATTENTION_IMPLEMENTATION}" <<'PY'
import sys
from pathlib import Path

import yaml

from grounded_summary.config import load_config

(
    base,
    destination,
    encoder,
    decoder,
    output_dir,
    data_dir,
    architecture,
    grounded_copy,
    copy_key_dim,
    region_router,
    batch_size,
    accumulation,
    attention_implementation,
) = sys.argv[1:]
config = load_config(base)
config["architecture"]["name"] = architecture
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config["decoder"]["grounded_copy"]["key_dim"] = int(copy_key_dim)
config["architecture"]["region_router"]["enabled"] = region_router == "true"
config["training"]["batch_size"] = int(batch_size)
config["training"]["gradient_accumulation_steps"] = int(accumulation)
config["model"]["attention_implementation"] = attention_implementation
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
  local predictions="${output_dir}/last_test_predictions.jsonl"

  make_config "${ROOT}/configs/afmr_pubmed.yaml" "${config_path}" "${encoder}" "${output_dir}"
  echo "=== Training ${name} ==="
  train_args=(train "${config_path}")
  if [[ "${OVERWRITE_OUTPUT_DIR}" =~ ^(1|true|yes)$ ]]; then
    train_args+=(--overwrite-output-dir)
  fi
  run_afmr "${train_args[@]}"

  echo "=== Evaluating ${name}: last.pt on PubMed test (${NPROC_PER_NODE} processes) ==="
  eval_args=(evaluate \
    "${config_path}" \
    "${output_dir}/last.pt" \
    "${predictions}" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --temperature "${EVAL_TEMPERATURE}" \
    --top-k "${EVAL_TOP_K}" \
    --top-p "${EVAL_TOP_P}")
  if [[ "${EVAL_DO_SAMPLE}" == true ]]; then
    eval_args+=(--do-sample)
  fi
  run_afmr "${eval_args[@]}"

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
