#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
cd "${PROJECT_ROOT}"

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

CONDITION="${1:-}"
case "${CONDITION}" in
  vanilla)
    BRIDGE_MODE=direct_projection
    GROUNDED_COPY=false
    ;;
  wo_copy)
    BRIDGE_MODE=afmr
    GROUNDED_COPY=false
    ;;
  wo_bridge)
    BRIDGE_MODE=direct_projection
    GROUNDED_COPY=true
    ;;
  *)
    die "Usage: bash scripts/run_pubmed_pair.sh {vanilla|wo_copy|wo_bridge}"
    ;;
esac
[[ "$#" -eq 1 ]] || die "Pass exactly one condition: vanilla, wo_copy, or wo_bridge"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

PUBMED_SOURCE_DIR="${PUBMED_SOURCE_DIR:-/workspace/storage-shared/nlp/dungdx4/datasets/pubmed}"
PROCESSED_DATA_DIR="${ROOT}/datasets/pubmed"
RAW_DATA_DIR="${ROOT}/datasets/raw/pubmed"
PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
SEEDS="${SEEDS:-43 44}"
GPU_IDS="${GPU_IDS:-0,1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/afmr/pubmed_${CONDITION}_seeds_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

[[ -x "${PYTHON_BIN}" || -n "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
[[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
[[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be positive"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
[[ "${#GPU_LIST[@]}" -eq 2 ]] || die "GPU_IDS must contain exactly two comma-separated GPU indices, e.g. 0,1"
[[ "${GPU_LIST[0]}" =~ ^[0-9]+$ && "${GPU_LIST[1]}" =~ ^[0-9]+$ ]] || die "GPU_IDS must contain numeric GPU indices"
[[ "${GPU_LIST[0]}" != "${GPU_LIST[1]}" ]] || die "GPU_IDS must name two different GPUs"

read -r -a SEED_LIST <<< "${SEEDS}"
(( ${#SEED_LIST[@]} > 0 )) || die "SEEDS must contain at least one seed"
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed '${seed}' in SEEDS"
done

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/run_pubmed_${CONDITION}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "=== PubMed ${CONDITION} runs; each seed uses two-GPU DDP ==="
echo "=== Seeds, run sequentially: ${SEED_LIST[*]} ==="
echo "=== GPUs: ${GPU_IDS} ==="
echo "=== Bridge mode: ${BRIDGE_MODE}; grounded copy: ${GROUNDED_COPY} ==="
echo "=== Training batch size and accumulation use configs/afmr_pubmed.yaml and afmr_base.yaml unchanged ==="
echo "=== Encoder: ${PPLX_ENCODER} ==="
echo "=== Decoder: ${DECODER_MODEL} ==="
echo "=== Output root: ${RUN_ROOT} ==="
echo "=== Main log: ${MAIN_LOG} ==="

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

ROUGE_SCRIPT="${ROUGE155_SCRIPT:-${PROJECT_ROOT}/src/rouge155/evaluate_rouge.py}"
if [[ ! -f "${ROUGE_SCRIPT}" ]]; then
  ROUGE_SCRIPT="${PROJECT_ROOT}/src/evaluation/evaluate_rouge.py"
fi
[[ -f "${ROUGE_SCRIPT}" ]] || die "ROUGE evaluator not found under ${PROJECT_ROOT}/src; set ROUGE155_SCRIPT"

make_config() {
  local seed="$1"
  local run_dir="$2"
  local config_path="${run_dir}/input_config.yaml"

  mkdir -p "${run_dir}"
  "${PYTHON_BIN}" - "${ROOT}" "${config_path}" "${PPLX_ENCODER}" "${DECODER_MODEL}" "${run_dir}" "${PROCESSED_DATA_DIR}" "${CONDITION}" "${seed}" "${BRIDGE_MODE}" "${GROUNDED_COPY}" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

root, config_path, encoder, decoder, run_dir, data_dir = map(Path, sys.argv[1:7])
condition, seed, bridge_mode, grounded_copy = sys.argv[7:]
config = load_config(root / "configs/afmr_pubmed.yaml")
config.pop("_meta", None)
config["experiment"]["name"] = f"pubmed_{condition}_seed{seed}"
config["experiment"]["output_dir"] = str(run_dir)
config["architecture"]["bridge_mode"] = bridge_mode
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config["model"]["encoder_name"] = str(encoder)
config["model"]["decoder_name"] = str(decoder)
config["training"]["seed"] = int(seed)
for split in ("train", "validation", "test"):
    config["data"][f"{split}_file"] = str(data_dir / f"{split}.jsonl")
validate_config(config)
config_path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
PY
}

run_one_seed() {
  local seed="$1"
  local run_dir="${RUN_ROOT}/seed${seed}"
  local config_path="${run_dir}/input_config.yaml"
  local predictions="${run_dir}/last_test_predictions.jsonl"

  make_config "${seed}" "${run_dir}"
  echo "=== Training ${CONDITION}, seed ${seed}, GPUs ${GPU_IDS} ==="
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    "${ROOT}/run_afmr.py" train "${config_path}" --overwrite-output-dir \
    2>&1 | tee "${run_dir}/train.log"

  [[ -s "${run_dir}/resolved_config.yaml" && -s "${run_dir}/last.pt" ]] || \
    die "Training did not produce resolved_config.yaml and last.pt in ${run_dir}"

  echo "=== Evaluating ${CONDITION}, seed ${seed} across the two GPUs ==="
  eval_failed=0
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
    "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${run_dir}/shard0.jsonl" \
    --split test --batch-size "${EVAL_BATCH_SIZE}" --shard-rank 0 --num-shards 2 \
    >"${run_dir}/eval_shard0.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
    "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${run_dir}/shard1.jsonl" \
    --split test --batch-size "${EVAL_BATCH_SIZE}" --shard-rank 1 --num-shards 2 \
    >"${run_dir}/eval_shard1.log" 2>&1 &
  pid1=$!

  if wait "${pid0}"; then
    echo "=== Evaluation shard 0 completed ==="
  else
    cat "${run_dir}/eval_shard0.log" >&2 || true
    eval_failed=1
  fi
  if wait "${pid1}"; then
    echo "=== Evaluation shard 1 completed ==="
  else
    cat "${run_dir}/eval_shard1.log" >&2 || true
    eval_failed=1
  fi
  (( eval_failed == 0 )) || die "Evaluation failed for ${CONDITION}, seed ${seed}"

  "${PYTHON_BIN}" "${ROOT}/scripts/merge_eval_shards.py" \
    --output "${predictions}" "${run_dir}/shard0.jsonl" "${run_dir}/shard1.jsonl"
  echo "=== ROUGE evaluation: ${CONDITION}, seed ${seed} ==="
  "${PYTHON_BIN}" "${ROUGE_SCRIPT}" "${predictions}"
  echo "=== Completed ${CONDITION}, seed ${seed}; output: ${run_dir} ==="
}

for seed in "${SEED_LIST[@]}"; do
  run_one_seed "${seed}"
done

echo "=== Completed all ${CONDITION} seeds: ${SEED_LIST[*]} ==="
echo "Runs and predictions: ${RUN_ROOT}"
