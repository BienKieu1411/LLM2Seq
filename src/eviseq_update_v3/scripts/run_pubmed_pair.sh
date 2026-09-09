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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-84}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export ALLOW_CROSS_SPLIT_CONTENT=true

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT}/datasets/pubmed}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/datasets/raw/pubmed}"
AFMR_ARCHITECTURE="${AFMR_ARCHITECTURE:-afmr_value_anchor}"
AFMR_GROUNDED_COPY="${AFMR_GROUNDED_COPY:-true}"
AFMR_SEMANTIC_READ="${AFMR_SEMANTIC_READ:-${AFMR_GROUNDED_COPY}}"
AFMR_SEMANTIC_VARIANT="${AFMR_SEMANTIC_VARIANT:-hierarchical_coverage}"
CROSS_QUERY_GATE="${CROSS_QUERY_GATE:-false}"
SEMANTIC_HEADS="${SEMANTIC_HEADS:-4}"
SEMANTIC_RANK="${SEMANTIC_RANK:-512}"
SEMANTIC_FUSION="${SEMANTIC_FUSION:-auto}"
PARTITION_HEADS="${PARTITION_HEADS:-false}"
export SEMANTIC_HEAD_GATE_POSITION="${SEMANTIC_HEAD_GATE_POSITION:-post_norm}"
USE_COVERAGE="${USE_COVERAGE:-true}"
USE_CONTINUITY="${USE_CONTINUITY:-true}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
case "${AFMR_SEMANTIC_VARIANT}" in
  shared_v1|shared_bounded|independent_unbounded|independent_bounded|hierarchical_coverage) ;;
  *) echo "Unsupported AFMR_SEMANTIC_VARIANT: ${AFMR_SEMANTIC_VARIANT}" >&2; exit 1 ;;
esac
[[ "${AFMR_GROUNDED_COPY}" == true || "${AFMR_GROUNDED_COPY}" == false ]] || { echo "AFMR_GROUNDED_COPY must be true or false" >&2; exit 1; }
[[ "${AFMR_SEMANTIC_READ}" == true || "${AFMR_SEMANTIC_READ}" == false ]] || { echo "AFMR_SEMANTIC_READ must be true or false" >&2; exit 1; }
[[ "${AFMR_SEMANTIC_READ}" == false || "${AFMR_GROUNDED_COPY}" == true ]] || { echo "Semantic read requires grounded copy" >&2; exit 1; }
[[ "${CROSS_QUERY_GATE}" == true || "${CROSS_QUERY_GATE}" == false ]] || { echo "CROSS_QUERY_GATE must be true or false" >&2; exit 1; }
[[ "${SEMANTIC_HEADS}" == 1 || "${SEMANTIC_HEADS}" == 4 ]] || { echo "SEMANTIC_HEADS must be 1 or 4" >&2; exit 1; }
[[ "${SEMANTIC_RANK}" =~ ^[1-9][0-9]*$ ]] && (( SEMANTIC_RANK % SEMANTIC_HEADS == 0 )) || { echo "SEMANTIC_RANK must be positive and divisible by SEMANTIC_HEADS" >&2; exit 1; }
for setting in PARTITION_HEADS USE_COVERAGE USE_CONTINUITY; do
  [[ "${!setting}" == true || "${!setting}" == false ]] || { echo "${setting} must be true or false" >&2; exit 1; }
done
[[ "${EVAL_SPLIT}" == test || "${EVAL_SPLIT}" == validation ]] || { echo "EVAL_SPLIT must be test or validation" >&2; exit 1; }
if [[ "${SEMANTIC_FUSION}" == auto ]]; then
  SEMANTIC_FUSION=residual
  [[ "${AFMR_SEMANTIC_VARIANT}" != hierarchical_coverage ]] || SEMANTIC_FUSION=norm_preserving
fi
[[ "${SEMANTIC_FUSION}" == residual || "${SEMANTIC_FUSION}" == norm_preserving ]] || { echo "Invalid SEMANTIC_FUSION" >&2; exit 1; }
[[ "${SEMANTIC_HEAD_GATE_POSITION}" == pre_norm || "${SEMANTIC_HEAD_GATE_POSITION}" == post_norm ]] || { echo "SEMANTIC_HEAD_GATE_POSITION must be pre_norm or post_norm" >&2; exit 1; }
if [[ "${AFMR_SEMANTIC_VARIANT}" == shared_* && "${SEMANTIC_HEADS}" != 1 ]]; then
  echo "Shared-copy semantic variants require SEMANTIC_HEADS=1" >&2
  exit 1
fi
COPY_VARIANT=lm
[[ "${AFMR_GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
[[ "${AFMR_SEMANTIC_READ}" == false ]] || COPY_VARIANT="copy_read_${AFMR_SEMANTIC_VARIANT}"
RUN_TAG="${AFMR_ARCHITECTURE}_${COPY_VARIANT}_qgate_${CROSS_QUERY_GATE}_h${SEMANTIC_HEADS}_r${SEMANTIC_RANK}_${SEMANTIC_FUSION}_hgate_${SEMANTIC_HEAD_GATE_POSITION}_part${PARTITION_HEADS}_cov${USE_COVERAGE}_cont${USE_CONTINUITY}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/eviseq_update_v3/pubmed_pair_${RUN_TAG}}"
GENERATED_CONFIG_DIR="${RUN_ROOT}/configs"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/eviseq_update_v3}"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
QWEN_ENCODER="${QWEN_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-Embedding-0.6B}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
read -r -a ENCODER_NAMES <<< "${RUN_ENCODERS:-pplx qwen_embedding}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${GENERATED_CONFIG_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_pair_v3_${RUN_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" || "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
for name in "${ENCODER_NAMES[@]}"; do
  case "${name}" in
    pplx) [[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}" ;;
    qwen_embedding) [[ -d "${QWEN_ENCODER}" ]] || die "Qwen embedding encoder not found: ${QWEN_ENCODER}" ;;
    *) die "RUN_ENCODERS supports pplx and qwen_embedding" ;;
  esac
done
[[ ${#ENCODER_NAMES[@]} -gt 0 ]] || die "RUN_ENCODERS must select at least one encoder"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE must be a positive integer"
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

echo "=== EviSeq update v3 PubMed sequential benchmark ==="
echo "=== GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
echo "=== Training: ${BATCH_SIZE} examples/GPU; ${NPROC_PER_NODE} GPU worker(s); accumulation=${GRADIENT_ACCUMULATION_STEPS}; clip=${MAX_GRAD_NORM} ==="
echo "=== Evaluation: one GPU after each training run ==="
echo "=== Architecture: ${AFMR_ARCHITECTURE}; FP32 updates, BF16 compute ==="
echo "=== Grounded copy: ${AFMR_GROUNDED_COPY} ==="
echo "=== Semantic read: ${AFMR_SEMANTIC_READ}; variant=${AFMR_SEMANTIC_VARIANT} ==="
echo "=== V3: query cross gate=${CROSS_QUERY_GATE}; ${SEMANTIC_HEADS} semantic heads x $((SEMANTIC_RANK / SEMANTIC_HEADS)) dimensions; fusion=${SEMANTIC_FUSION} ==="
echo "=== Planner: partition=${PARTITION_HEADS}; coverage=${USE_COVERAGE}; continuity=${USE_CONTINUITY}; eval=${EVAL_SPLIT} ==="
echo "=== Semantic head gate: ${SEMANTIC_HEAD_GATE_POSITION} ==="
echo "=== Python: ${PYTHON_BIN} ==="
echo "=== Log: ${LOG_FILE} ==="
echo "=== Encoder queue: ${ENCODER_NAMES[*]} -> Qwen3 decoder ==="

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
"${PYTHON_BIN}" - "${base_config}" "${output_config}" "${encoder_name}" "${DECODER_MODEL}" "${output_dir}" "${PROCESSED_DATA_DIR}" "${AFMR_ARCHITECTURE}" "${AFMR_GROUNDED_COPY}" "${AFMR_SEMANTIC_READ}" "${AFMR_SEMANTIC_VARIANT}" "${CROSS_QUERY_GATE}" "${SEMANTIC_HEADS}" "${SEMANTIC_RANK}" "${SEMANTIC_FUSION}" "${PARTITION_HEADS}" "${USE_COVERAGE}" "${USE_CONTINUITY}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

from eviseq_update_v3.config import load_config, validate_config

(
    base, destination, encoder, decoder, output_dir, data_dir, architecture,
    grounded_copy, semantic_read, variant, query_gate, semantic_heads, semantic_rank,
    fusion, partition, coverage, continuity,
) = sys.argv[1:]
config = load_config(base)
config["architecture"]["name"] = architecture
config["decoder"]["query_cross_gate"] = query_gate == "true"
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = semantic_read == "true"
config["decoder"]["grounded_copy"]["semantic_read"]["num_heads"] = int(semantic_heads)
config["decoder"]["grounded_copy"]["semantic_read"]["rank"] = int(semantic_rank)
config["decoder"]["grounded_copy"]["semantic_read"]["fusion"] = fusion
config["decoder"]["grounded_copy"]["semantic_read"]["head_gate_position"] = os.environ["SEMANTIC_HEAD_GATE_POSITION"]
config["decoder"]["grounded_copy"]["semantic_read"]["planner"].update(
    partition_heads=partition == "true", use_coverage=coverage == "true", use_continuity=continuity == "true"
)
attention, cap = {
    "shared_v1": ("shared_copy", None),
    "shared_bounded": ("shared_copy", 0.1),
    "independent_unbounded": ("independent_source", None),
    "independent_bounded": ("independent_source", 0.1),
    "hierarchical_coverage": ("hierarchical_coverage", 0.1),
}[variant]
config["decoder"]["grounded_copy"]["semantic_read"].update(attention=attention, max_relative_rms=cap)
config.pop("_meta", None)
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["experiment"]["output_dir"] = output_dir
config["training"]["batch_size"] = int(os.environ["BATCH_SIZE"])
config["training"]["gradient_accumulation_steps"] = int(os.environ["GRADIENT_ACCUMULATION_STEPS"])
clip = os.environ["MAX_GRAD_NORM"].strip().lower()
config["training"]["max_grad_norm"] = None if clip in {"none", "null"} else float(clip)
config["data"]["train_file"] = str(Path(data_dir) / "train.jsonl")
config["data"]["validation_file"] = str(Path(data_dir) / "validation.jsonl")
config["data"]["test_file"] = str(Path(data_dir) / "test.jsonl")
validate_config(config)
effective_batch = config["training"]["batch_size"] * int(os.environ["NPROC_PER_NODE"]) * config["training"]["gradient_accumulation_steps"]
print(f"=== Effective batch: {effective_batch} examples/update ===")
print(f"=== Epochs: {config['training']['interface_warmup_epochs']} warmup + {config['training']['full_finetune_epochs']} full; clip={config['training']['max_grad_norm']} ===")
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
}

run_one() {
  local name="$1"
  local encoder="$2"
  local config_path="${GENERATED_CONFIG_DIR}/${name}.yaml"
  local output_dir="${RUN_ROOT}/${name}"
  local predictions="${output_dir}/last_${EVAL_SPLIT}_predictions.jsonl"

  make_config "${ROOT}/configs/afmr_pubmed.yaml" "${config_path}" "${encoder}" "${output_dir}"
  echo "=== Training ${name} ==="
  train_args=(train "${config_path}")
  if [[ "${OVERWRITE_OUTPUT_DIR}" =~ ^(1|true|yes)$ ]]; then
    train_args+=(--overwrite-output-dir)
  fi
  bash "${ROOT}/scripts/run_afmr.sh" "${train_args[@]}"

  echo "=== Evaluating ${name}: last.pt on PubMed ${EVAL_SPLIT} ==="
  bash "${ROOT}/scripts/run_afmr.sh" evaluate \
    "${config_path}" \
    "${output_dir}/last.pt" \
    "${predictions}" \
    --split "${EVAL_SPLIT}" \
    --batch-size "${EVAL_BATCH_SIZE}"

  if [[ -n "${ROUGE155_SCRIPT:-}" && -f "${ROUGE155_SCRIPT}" ]]; then
    echo "=== Perl ROUGE-1.5.5 for ${name} ==="
    "${PYTHON_BIN}" "${ROUGE155_SCRIPT}" "${predictions}" --output "${predictions%.jsonl}.rouge155.json"
  else
    echo "=== ROUGE-1.5.5 skipped for ${name}; set ROUGE155_SCRIPT to evaluate it ==="
  fi
}

for name in "${ENCODER_NAMES[@]}"; do
  case "${name}" in
    pplx) run_one "pplx" "${PPLX_ENCODER}" ;;
    qwen_embedding) run_one "qwen_embedding" "${QWEN_ENCODER}" ;;
  esac
done

echo "=== PubMed pair completed ==="
for name in "${ENCODER_NAMES[@]}"; do
  echo "${name} output: ${RUN_ROOT}/${name}"
done
