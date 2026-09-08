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
# Default CE + cosine: 48 sources/GPU * 2 GPUs * 1 accumulation step = 96.
export AFMR_TRAINING_RECIPE="${AFMR_TRAINING_RECIPE:-cosine}"
case "${AFMR_TRAINING_RECIPE}" in
  ce|cosine|dropout|evidence) ;;
  *) echo "AFMR_TRAINING_RECIPE must be ce, cosine, dropout or evidence" >&2; exit 1 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-48}"
[[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE and NPROC_PER_NODE must be positive integers" >&2; exit 1; }
if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
  (( 96 % (BATCH_SIZE * NPROC_PER_NODE) == 0 )) || { echo "Set GRADIENT_ACCUMULATION_STEPS explicitly when batch * GPUs does not divide 96" >&2; exit 1; }
  GRADIENT_ACCUMULATION_STEPS=$((96 / (BATCH_SIZE * NPROC_PER_NODE)))
fi
export GRADIENT_ACCUMULATION_STEPS
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${PROCESSED_DATA_DIR:-${ROOT}/datasets/pubmed}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/datasets/raw/pubmed}"
AFMR_ARCHITECTURE="${AFMR_ARCHITECTURE:-afmr_value_anchor}"
AFMR_GROUNDED_COPY="${AFMR_GROUNDED_COPY:-true}"
AFMR_SEMANTIC_READ="${AFMR_SEMANTIC_READ:-${AFMR_GROUNDED_COPY}}"
AFMR_SEMANTIC_VARIANT="${AFMR_SEMANTIC_VARIANT:-independent_bounded}"
case "${AFMR_SEMANTIC_VARIANT}" in
  shared_v1|shared_bounded|independent_unbounded|independent_bounded) ;;
  *) echo "Unsupported AFMR_SEMANTIC_VARIANT: ${AFMR_SEMANTIC_VARIANT}" >&2; exit 1 ;;
esac
[[ "${AFMR_GROUNDED_COPY}" == true || "${AFMR_GROUNDED_COPY}" == false ]] || { echo "AFMR_GROUNDED_COPY must be true or false" >&2; exit 1; }
[[ "${AFMR_SEMANTIC_READ}" == true || "${AFMR_SEMANTIC_READ}" == false ]] || { echo "AFMR_SEMANTIC_READ must be true or false" >&2; exit 1; }
[[ "${AFMR_SEMANTIC_READ}" == false || "${AFMR_GROUNDED_COPY}" == true ]] || { echo "Semantic read requires grounded copy" >&2; exit 1; }
AFMR_EVIDENCE_MODE="${AFMR_EVIDENCE_MODE:-both}"
AFMR_EVIDENCE_MAX_WEIGHT="${AFMR_EVIDENCE_MAX_WEIGHT:-0.05}"
AFMR_EVIDENCE_RAMP_RATIO="${AFMR_EVIDENCE_RAMP_RATIO:-0.10}"
export AFMR_EVIDENCE_MODE AFMR_EVIDENCE_MAX_WEIGHT AFMR_EVIDENCE_RAMP_RATIO
case "${AFMR_EVIDENCE_MODE}" in copy|semantic|both) ;; *) echo "AFMR_EVIDENCE_MODE must be copy, semantic or both" >&2; exit 1 ;; esac
COPY_VARIANT=lm
[[ "${AFMR_GROUNDED_COPY}" == false ]] || COPY_VARIANT=copy
[[ "${AFMR_SEMANTIC_READ}" == false ]] || COPY_VARIANT="copy_read_${AFMR_SEMANTIC_VARIANT}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/eviseq_update/pubmed_pair_${AFMR_ARCHITECTURE}_${COPY_VARIANT}_${AFMR_TRAINING_RECIPE}}"
GENERATED_CONFIG_DIR="${RUN_ROOT}/configs"
EVIDENCE_CACHE_ROOT="${EVIDENCE_CACHE_ROOT:-${RUN_ROOT}/evidence}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/afmr}"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
QWEN_ENCODER="${QWEN_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-Embedding-0.6B}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
OVERWRITE_OUTPUT_DIR="${OVERWRITE_OUTPUT_DIR:-false}"
read -r -a ENCODER_NAMES <<< "${RUN_ENCODERS:-pplx qwen_embedding}"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}" "${GENERATED_CONFIG_DIR}"
LOG_FILE="${LOG_DIR}/pubmed_pair_$(date +%Y%m%d_%H%M%S).log"
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
if [[ "${AFMR_TRAINING_RECIPE}" == evidence ]]; then
  [[ "${AFMR_ARCHITECTURE}" == afmr_value_anchor ]] || die "Evidence recipe requires AFMR_ARCHITECTURE=afmr_value_anchor"
  [[ "${AFMR_GROUNDED_COPY}" == true && "${AFMR_SEMANTIC_READ}" == true && "${AFMR_SEMANTIC_VARIANT}" == independent_bounded ]] || die "Evidence recipe requires grounded copy plus independent_bounded semantic read"
fi

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
echo "=== Training: ${BATCH_SIZE} examples/GPU; ${NPROC_PER_NODE} GPU worker(s); accumulation=${GRADIENT_ACCUMULATION_STEPS}; clip=${MAX_GRAD_NORM} ==="
echo "=== Evaluation: one GPU after each training run ==="
echo "=== Architecture: ${AFMR_ARCHITECTURE}; FP32 updates, BF16 compute ==="
echo "=== Grounded copy: ${AFMR_GROUNDED_COPY} ==="
echo "=== Semantic read: ${AFMR_SEMANTIC_READ}; variant=${AFMR_SEMANTIC_VARIANT} ==="
echo "=== Training recipe: ${AFMR_TRAINING_RECIPE}; training uses only source/reference, no text generation ==="
[[ "${AFMR_TRAINING_RECIPE}" != evidence ]] || echo "=== Evidence contrastive: mode=${AFMR_EVIDENCE_MODE}; max_lambda=${AFMR_EVIDENCE_MAX_WEIGHT}; ramp=${AFMR_EVIDENCE_RAMP_RATIO} ==="
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
  local evidence_cache="$5"
"${PYTHON_BIN}" - "${base_config}" "${output_config}" "${encoder_name}" "${DECODER_MODEL}" "${output_dir}" "${PROCESSED_DATA_DIR}" "${AFMR_ARCHITECTURE}" "${AFMR_GROUNDED_COPY}" "${AFMR_SEMANTIC_READ}" "${AFMR_SEMANTIC_VARIANT}" "${evidence_cache}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

from eviseq_update.config import load_config, validate_config

base, destination, encoder, decoder, output_dir, data_dir, architecture, grounded_copy, semantic_read, variant, evidence_cache = sys.argv[1:]
config = load_config(base)
config["architecture"]["name"] = architecture
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config["decoder"]["grounded_copy"]["semantic_read"]["enabled"] = semantic_read == "true"
attention, cap = {
    "shared_v1": ("shared_copy", None),
    "shared_bounded": ("shared_copy", 0.1),
    "independent_unbounded": ("independent_source", None),
    "independent_bounded": ("independent_source", 0.1),
}[variant]
config["decoder"]["grounded_copy"]["semantic_read"].update(attention=attention, max_relative_rms=cap)
recipe = os.environ["AFMR_TRAINING_RECIPE"]
config["decoder"]["attention_dropout"] = 0.1 if recipe == "dropout" else 0.0
config["training"].update(
    lr_scheduler="linear" if recipe == "ce" else "cosine",
    lr_warmup_ratio=0.0 if recipe == "ce" else 0.05,
    save_best=True,
)
if recipe == "evidence":
    config["training"]["evidence_contrastive"] = {
        "enabled": True,
        "cache_path": evidence_cache,
        "mode": os.environ["AFMR_EVIDENCE_MODE"],
        "max_weight": float(os.environ["AFMR_EVIDENCE_MAX_WEIGHT"]),
        "ramp_ratio": float(os.environ["AFMR_EVIDENCE_RAMP_RATIO"]),
        "mining": {},
    }
else:
    config["training"]["evidence_contrastive"] = {
        "enabled": False, "mode": "both", "max_weight": 0.05, "ramp_ratio": 0.10, "cache_path": "", "mining": {}
    }
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
objective = "gold CE + actual-read evidence contrastive" if recipe == "evidence" else "gold reference CE"
print(f"=== Objective: {objective}; scheduler={config['training']['lr_scheduler']} ===")
Path(destination).write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY
}

run_one() {
  local name="$1"
  local encoder="$2"
  local config_path="${GENERATED_CONFIG_DIR}/${name}.yaml"
  local output_dir="${RUN_ROOT}/${name}"
  local predictions="${output_dir}/last_test_predictions.jsonl"
  local evidence_dir="${EVIDENCE_CACHE_ROOT}/${name}"

  make_config "${ROOT}/configs/afmr_pubmed.yaml" "${config_path}" "${encoder}" "${output_dir}" "${evidence_dir}/evidence.jsonl"
  if [[ "${AFMR_TRAINING_RECIPE}" == evidence ]]; then
    if [[ -f "${evidence_dir}/evidence.jsonl" ]]; then
      echo "=== Verifying existing deterministic evidence cache for ${name} ==="
      "${PYTHON_BIN}" - "${config_path}" "${evidence_dir}/evidence.jsonl" <<'PY'
import sys
from eviseq_update.config import load_config
from eviseq_update.data.evidence_cache import validate_cache_manifest

validate_cache_manifest(sys.argv[2], load_config(sys.argv[1]))
PY
    else
      echo "=== Preparing deterministic evidence cache for ${name} ==="
      "${PYTHON_BIN}" "${ROOT}/scripts/prepare_evidence.py" --config "${config_path}" --split train --output-dir "${evidence_dir}"
    fi
  fi
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
