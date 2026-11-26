#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/../.." && pwd)"
cd "${PROJECT_ROOT}"

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
GPU_FULL="${GPU_FULL:-0}"
GPU_VANILLA="${GPU_VANILLA:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/afmr/pubmed_full_vanilla_multiseed_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${RUN_ROOT}/logs}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -x "${PYTHON_BIN}" || -n "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]] || die "Python not found: ${PYTHON_BIN}"
[[ -d "${PUBMED_SOURCE_DIR}" ]] || die "PubMed source directory not found: ${PUBMED_SOURCE_DIR}"
[[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
[[ "${GPU_FULL}" =~ ^[0-9]+$ && "${GPU_VANILLA}" =~ ^[0-9]+$ ]] || die "GPU_FULL and GPU_VANILLA must each be one GPU index"
[[ "${GPU_FULL}" != "${GPU_VANILLA}" ]] || die "GPU_FULL and GPU_VANILLA must be different; each training uses one GPU"
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be a positive integer"

read -r -a SEED_LIST <<< "${SEEDS}"
(( ${#SEED_LIST[@]} > 0 )) || die "SEEDS must contain at least one seed"
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed '${seed}' in SEEDS"
done

mkdir -p "${LOG_DIR}" "${RUN_ROOT}"
MAIN_LOG="${LOG_DIR}/run_pubmed_pair_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "=== PubMed full SEAM vs vanilla cross-attention multi-seed run ==="
echo "=== Seeds: ${SEED_LIST[*]} ==="
echo "=== Full GPU: ${GPU_FULL}; vanilla GPU: ${GPU_VANILLA} (one GPU per training) ==="
echo "=== Encoder: ${PPLX_ENCODER} ==="
echo "=== Decoder: ${DECODER_MODEL} ==="
echo "=== Training batch size and learning rates come from configs/afmr_pubmed.yaml and afmr_base.yaml ==="
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
  local condition="$1"
  local seed="$2"
  local run_dir="$3"
  local config_path="${run_dir}/input_config.yaml"
  local bridge_mode="afmr"
  local use_copy=true
  [[ "${condition}" == full ]] || { bridge_mode=direct_projection; use_copy=false; }

  mkdir -p "${run_dir}"
  "${PYTHON_BIN}" - "${ROOT}" "${config_path}" "${PPLX_ENCODER}" "${DECODER_MODEL}" "${run_dir}" "${PROCESSED_DATA_DIR}" "${condition}" "${seed}" "${bridge_mode}" "${use_copy}" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

root, config_path, encoder, decoder, run_dir, data_dir = map(Path, sys.argv[1:7])
condition, seed, bridge_mode, use_copy = sys.argv[7:]
config = load_config(root / "configs/afmr_pubmed.yaml")
config.pop("_meta", None)
config["experiment"]["name"] = f"pubmed_{condition}_seed{seed}"
config["experiment"]["output_dir"] = str(run_dir)
config["architecture"]["bridge_mode"] = bridge_mode
config["decoder"]["grounded_copy"]["enabled"] = use_copy == "true"
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

run_condition() {
  local condition="$1"
  local seed="$2"
  local gpu="$3"
  local run_dir="${RUN_ROOT}/${condition}_seed${seed}"
  local predictions="${run_dir}/last_test_predictions.jsonl"
  local config_path="${run_dir}/input_config.yaml"

  make_config "${condition}" "${seed}" "${run_dir}"
  echo "=== Training ${condition}, seed ${seed}, GPU ${gpu} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" train \
    "${config_path}" --overwrite-output-dir

  [[ -s "${run_dir}/resolved_config.yaml" && -s "${run_dir}/last.pt" ]] || \
    die "Training did not produce resolved_config.yaml and last.pt in ${run_dir}"

  echo "=== Evaluating ${condition}, seed ${seed}, GPU ${gpu} ==="
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
    "${run_dir}/resolved_config.yaml" \
    "${run_dir}/last.pt" \
    "${predictions}" \
    --split test \
    --batch-size "${EVAL_BATCH_SIZE}"

  echo "=== ROUGE evaluation: ${condition}, seed ${seed} ==="
  "${PYTHON_BIN}" "${ROUGE_SCRIPT}" "${predictions}"
}

for seed in "${SEED_LIST[@]}"; do
  full_log="${LOG_DIR}/full_seed${seed}.log"
  vanilla_log="${LOG_DIR}/vanilla_seed${seed}.log"
  echo "=== Starting paired runs for seed ${seed} ==="
  run_condition full "${seed}" "${GPU_FULL}" >"${full_log}" 2>&1 &
  full_pid=$!
  run_condition vanilla "${seed}" "${GPU_VANILLA}" >"${vanilla_log}" 2>&1 &
  vanilla_pid=$!

  run_failed=0
  if wait "${full_pid}"; then
    echo "=== Full seed ${seed} completed; log: ${full_log} ==="
  else
    echo "=== Full seed ${seed} failed; log: ${full_log} ===" >&2
    tail -n 80 "${full_log}" >&2 || true
    run_failed=1
  fi
  if wait "${vanilla_pid}"; then
    echo "=== Vanilla seed ${seed} completed; log: ${vanilla_log} ==="
  else
    echo "=== Vanilla seed ${seed} failed; log: ${vanilla_log} ===" >&2
    tail -n 80 "${vanilla_log}" >&2 || true
    run_failed=1
  fi
  (( run_failed == 0 )) || die "At least one run failed for seed ${seed}; see the logs above"
done

echo "=== PubMed multi-seed runs completed ==="
echo "Runs and predictions: ${RUN_ROOT}"
echo "Full condition: AFMR bridge + grounded copy"
echo "Vanilla condition: direct projection + no grounded copy"
