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
  PYTHON_BIN=python3
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
    die "Usage: bash scripts/run_govreport_ablation.sh {vanilla|wo_copy|wo_bridge}"
    ;;
esac
[[ "$#" -eq 1 ]] || die "Pass exactly one condition: vanilla, wo_copy, or wo_bridge"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

CONFIG_TEMPLATE="${AFMR_GOVREPORT_CONFIG:-${ROOT}/configs/afmr_govreport.yaml}"
DATA_DIR="${GOVREPORT_DATA_DIR:-${ROOT}/datasets/govreport}"
SOURCE_DIR="${GOVREPORT_SOURCE_DIR:-}"
ENCODER_MODEL="${ENCODER_MODEL:-${PPLX_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/pplx-embed-v1-0.6b}}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
SEEDS="${SEEDS:-42}"
GPU_IDS="${GPU_IDS:-0,1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/afmr/govreport_${CONDITION}_seeds_$(date +%Y%m%d_%H%M%S)}"

[[ -n "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
[[ -f "${CONFIG_TEMPLATE}" ]] || die "Config not found: ${CONFIG_TEMPLATE}"
[[ -d "${ENCODER_MODEL}" ]] || die "Encoder not found: ${ENCODER_MODEL}"
[[ -d "${DECODER_MODEL}" ]] || die "Decoder not found: ${DECODER_MODEL}"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
[[ "${#GPU_LIST[@]}" -eq 2 ]] || die "GPU_IDS must contain exactly two GPU IDs, e.g. 0,1"
[[ "${GPU_LIST[0]}" =~ ^[0-9]+$ && "${GPU_LIST[1]}" =~ ^[0-9]+$ ]] || die "GPU_IDS must be numeric"
[[ "${GPU_LIST[0]}" != "${GPU_LIST[1]}" ]] || die "GPU_IDS must name two different GPUs"

read -r -a SEED_LIST <<< "${SEEDS}"
(( ${#SEED_LIST[@]} > 0 )) || die "SEEDS must contain at least one seed"
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed '${seed}' in SEEDS"
done

if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/validation.jsonl" || ! -s "${DATA_DIR}/test.jsonl" ]]; then
  [[ -n "${SOURCE_DIR}" && -d "${SOURCE_DIR}" ]] || \
    die "Prepared GovReport splits are missing in ${DATA_DIR}; set GOVREPORT_SOURCE_DIR to prepare them"
  prepare_args=(--dataset govreport --input-dir "${SOURCE_DIR}" --output-dir "${DATA_DIR}")
  if [[ "${ALLOW_CROSS_SPLIT_CONTENT:-false}" =~ ^(1|true|yes)$ ]]; then
    prepare_args+=(--allow-cross-split-content)
  fi
  if [[ "${ALLOW_DUPLICATE_IDS:-false}" =~ ^(1|true|yes)$ ]]; then
    prepare_args+=(--allow-duplicate-ids)
  fi
  bash "${ROOT}/scripts/prepare_afmr.sh" "${prepare_args[@]}"
fi
for split in train validation test; do
  [[ -s "${DATA_DIR}/${split}.jsonl" ]] || die "Missing prepared split: ${DATA_DIR}/${split}.jsonl"
done

ROUGE_SCRIPT="${ROUGE155_SCRIPT:-${PROJECT_ROOT}/src/rouge155/evaluate_rouge.py}"
[[ -f "${ROUGE_SCRIPT}" ]] || die "ROUGE evaluator not found: ${ROUGE_SCRIPT}; set ROUGE155_SCRIPT"

mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
echo "=== GovReport ${CONDITION}; bridge=${BRIDGE_MODE}; grounded_copy=${GROUNDED_COPY} ==="
echo "=== Seeds, run sequentially: ${SEED_LIST[*]}; GPUs: ${GPU_IDS} (DDP) ==="
echo "=== GovReport config batch size and gradient accumulation are preserved ==="
echo "=== Encoder: ${ENCODER_MODEL}"
echo "=== Decoder: ${DECODER_MODEL}"
echo "=== Output: ${RUN_ROOT}"

make_config() {
  local seed="$1"
  local run_dir="$2"
  mkdir -p "${run_dir}"
  "${PYTHON_BIN}" - "${CONFIG_TEMPLATE}" "${run_dir}/input_config.yaml" \
    "${ENCODER_MODEL}" "${DECODER_MODEL}" "${run_dir}" "${DATA_DIR}" \
    "${CONDITION}" "${seed}" "${BRIDGE_MODE}" "${GROUNDED_COPY}" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

template, destination, encoder, decoder, run_dir, data_dir = map(Path, sys.argv[1:7])
condition, seed, bridge_mode, grounded_copy = sys.argv[7:]
config = load_config(template)
config.pop("_meta", None)
config["experiment"]["name"] = f"govreport_{condition}_seed{seed}"
config["experiment"]["output_dir"] = str(run_dir.resolve())
config["architecture"]["bridge_mode"] = bridge_mode
config["decoder"]["grounded_copy"]["enabled"] = grounded_copy == "true"
config["model"]["encoder_name"] = str(encoder.resolve())
config["model"]["decoder_name"] = str(decoder.resolve())
config["training"]["seed"] = int(seed)
for split in ("train", "validation", "test"):
    config["data"][f"{split}_file"] = str((data_dir / f"{split}.jsonl").resolve())

# Keep batch size, gradient accumulation, epochs, and optimizer settings from afmr_govreport.yaml.
validate_config(config)
destination.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
train = config["training"]
print(
    f"Config {destination}: seed={seed}, batch/GPU={train['batch_size']}, "
    f"grad_accum={train['gradient_accumulation_steps']}"
)
PY
}

run_one_seed() {
  local seed="$1"
  local run_dir="${RUN_ROOT}/seed${seed}"
  local config_path="${run_dir}/input_config.yaml"
  local predictions="${run_dir}/last_test_predictions.jsonl"

  make_config "${seed}" "${run_dir}"
  echo "=== Training GovReport ${CONDITION}, seed ${seed}, GPUs ${GPU_IDS} ==="
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone --nproc_per_node=2 \
    "${ROOT}/run_afmr.py" train "${config_path}" --overwrite-output-dir \
    2>&1 | tee "${run_dir}/train.log"
  [[ -s "${run_dir}/resolved_config.yaml" && -s "${run_dir}/last.pt" ]] || \
    die "Training did not produce resolved_config.yaml and last.pt in ${run_dir}"

  echo "=== Evaluating GovReport ${CONDITION}, seed ${seed} on two GPU shards ==="
  CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
    "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${run_dir}/shard0.jsonl" \
    --split test --batch-size "${EVAL_BATCH_SIZE}" --shard-rank 0 --num-shards 2 --device cuda:0 \
    >"${run_dir}/eval_shard0.log" 2>&1 &
  local pid0=$!
  CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
    "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
    "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${run_dir}/shard1.jsonl" \
    --split test --batch-size "${EVAL_BATCH_SIZE}" --shard-rank 1 --num-shards 2 --device cuda:0 \
    >"${run_dir}/eval_shard1.log" 2>&1 &
  local pid1=$!

  local eval_failed=0
  if ! wait "${pid0}"; then
    echo "GPU ${GPU_LIST[0]} evaluation failed; log: ${run_dir}/eval_shard0.log" >&2
    tail -n 40 "${run_dir}/eval_shard0.log" >&2 || true
    eval_failed=1
  fi
  if ! wait "${pid1}"; then
    echo "GPU ${GPU_LIST[1]} evaluation failed; log: ${run_dir}/eval_shard1.log" >&2
    tail -n 40 "${run_dir}/eval_shard1.log" >&2 || true
    eval_failed=1
  fi
  (( eval_failed == 0 )) || die "Evaluation failed for ${CONDITION}, seed ${seed}"

  "${PYTHON_BIN}" "${ROOT}/scripts/merge_eval_shards.py" \
    --output "${predictions}" "${run_dir}/shard0.jsonl" "${run_dir}/shard1.jsonl"
  PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR}" "${PYTHON_BIN}" "${ROUGE_SCRIPT}" "${predictions}"
  echo "=== Completed GovReport ${CONDITION}, seed ${seed}: ${run_dir} ==="
}

for seed in "${SEED_LIST[@]}"; do
  run_one_seed "${seed}"
done

echo "=== Completed GovReport ${CONDITION}; seeds: ${SEED_LIST[*]}; runs: ${RUN_ROOT} ==="
