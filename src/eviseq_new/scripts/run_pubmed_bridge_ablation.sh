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

ABLATION="${1:-}"
case "${ABLATION}" in
  no_multi_depth)
    MULTI_DEPTH=false
    ADAPTIVE_RESIDUALS=true
    SOURCE_BIAS=true
    ;;
  no_adaptive_residuals)
    MULTI_DEPTH=true
    ADAPTIVE_RESIDUALS=false
    SOURCE_BIAS=true
    ;;
  no_source_bias)
    MULTI_DEPTH=true
    ADAPTIVE_RESIDUALS=true
    SOURCE_BIAS=false
    ;;
  *)
    die "Usage: bash scripts/run_pubmed_bridge_ablation.sh {no_multi_depth|no_adaptive_residuals|no_source_bias}"
    ;;
esac
[[ "$#" -eq 1 ]] || die "Pass exactly one bridge component ablation"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/users/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

PPLX_ENCODER="${PPLX_ENCODER:-/workspace/storage-shared/nlp/users/dungdx4/BERT/pplx-embed-v1-0.6b}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-0.6B}"
SEEDS="${SEEDS:-42}"
GPU_IDS="${GPU_IDS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/afmr/pubmed_bridge_${ABLATION}_$(date +%Y%m%d_%H%M%S)}"
RESUME_EXISTING="${RESUME_EXISTING:-false}"

[[ -d "${PPLX_ENCODER}" ]] || die "PPLX encoder not found: ${PPLX_ENCODER}"
[[ -d "${DECODER_MODEL}" ]] || die "Qwen decoder not found: ${DECODER_MODEL}"
for split in train validation test; do
  [[ -s "${ROOT}/datasets/pubmed/${split}.jsonl" ]] || die "Missing PubMed ${split}.jsonl under ${ROOT}/datasets/pubmed"
done
[[ "${EVAL_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_BATCH_SIZE must be positive"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
(( ${#GPU_LIST[@]} == 1 || ${#GPU_LIST[@]} == 2 )) || die "GPU_IDS must name one GPU or two GPUs, e.g. 0 or 0,1"
for gpu in "${GPU_LIST[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "Invalid GPU index: ${gpu}"
done
if (( ${#GPU_LIST[@]} == 2 )) && [[ "${GPU_LIST[0]}" == "${GPU_LIST[1]}" ]]; then
  die "GPU_IDS must name two different GPUs"
fi

read -r -a SEED_LIST <<< "${SEEDS}"
(( ${#SEED_LIST[@]} > 0 )) || die "SEEDS must contain at least one seed"
for seed in "${SEED_LIST[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "Invalid seed '${seed}' in SEEDS"
done

ROUGE_SCRIPT="${ROUGE155_SCRIPT:-${PROJECT_ROOT}/src/rouge155/evaluate_rouge.py}"
if [[ ! -f "${ROUGE_SCRIPT}" ]]; then
  ROUGE_SCRIPT="${PROJECT_ROOT}/src/evaluation/evaluate_rouge.py"
fi
[[ -f "${ROUGE_SCRIPT}" ]] || die "ROUGE evaluator not found; set ROUGE155_SCRIPT"

mkdir -p "${RUN_ROOT}"
MAIN_LOG="${RUN_ROOT}/run_${ABLATION}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "=== PubMed bridge component ablation: ${ABLATION} ==="
echo "=== Seeds, run sequentially: ${SEED_LIST[*]} ==="
echo "=== GPUs: ${GPU_IDS} ==="
echo "=== Encoder: ${PPLX_ENCODER} ==="
echo "=== Decoder: ${DECODER_MODEL} ==="
echo "=== Batch size and gradient accumulation stay from afmr_pubmed.yaml ==="
echo "=== Output root: ${RUN_ROOT} ==="
echo "=== Main log: ${MAIN_LOG} ==="

make_config() {
  local seed="$1"
  local run_dir="$2"
  local config_path="${run_dir}/input_config.yaml"
  mkdir -p "${run_dir}"

  "${PYTHON_BIN}" - "${ROOT}" "${config_path}" "${PPLX_ENCODER}" "${DECODER_MODEL}" \
    "${run_dir}" "${seed}" "${ABLATION}" "${MULTI_DEPTH}" "${ADAPTIVE_RESIDUALS}" \
    "${SOURCE_BIAS}" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

root = Path(sys.argv[1])
config_path = Path(sys.argv[2])
encoder = sys.argv[3]
decoder = sys.argv[4]
run_dir = Path(sys.argv[5])
seed = int(sys.argv[6])
ablation = sys.argv[7]
multi_depth, adaptive_residuals, source_bias = (value == "true" for value in sys.argv[8:11])

config = load_config(root / "configs/afmr_pubmed.yaml")
config.pop("_meta", None)
config["experiment"]["name"] = f"pubmed_bridge_{ablation}_seed{seed}"
config["experiment"]["output_dir"] = str(run_dir)
config["architecture"]["bridge_mode"] = "afmr"
config["architecture"]["multi_depth_retrieval_enabled"] = multi_depth
config["architecture"]["adaptive_residuals_enabled"] = adaptive_residuals
config["architecture"]["learned_source_bias_enabled"] = source_bias
config["model"]["encoder_name"] = encoder
config["model"]["decoder_name"] = decoder
config["training"]["seed"] = seed
if not config["decoder"]["grounded_copy"]["enabled"]:
    raise ValueError("Bridge component ablations require the full SEAM copy route to remain enabled")
for split in ("train", "validation", "test"):
    config["data"][f"{split}_file"] = str(root / "datasets/pubmed" / f"{split}.jsonl")
validate_config(config)
config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(
    f"seed={seed} batch_size={config['training']['batch_size']} "
    f"gradient_accumulation_steps={config['training']['gradient_accumulation_steps']} "
    f"copy={config['decoder']['grounded_copy']['enabled']}"
)
PY
}

run_one_seed() {
  local seed="$1"
  local run_dir="${RUN_ROOT}/seed${seed}"
  local config_path="${run_dir}/input_config.yaml"
  local predictions="${run_dir}/last_test_predictions.jsonl"
  local -a resume_args=()

  if [[ "${RESUME_EXISTING}" =~ ^(1|true|yes)$ ]]; then
    [[ -s "${run_dir}/last.pt" ]] || die "No checkpoint to resume at ${run_dir}/last.pt"
    resume_args=(--resume-checkpoint "${run_dir}/last.pt")
  elif [[ -s "${run_dir}/last.pt" ]]; then
    die "Checkpoint already exists at ${run_dir}/last.pt; use a new RUN_ROOT or RESUME_EXISTING=true"
  fi

  make_config "${seed}" "${run_dir}"
  echo "=== Training ${ABLATION}, seed ${seed}, GPUs ${GPU_IDS} ==="
  if (( ${#GPU_LIST[@]} == 1 )); then
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" train \
      "${config_path}" "${resume_args[@]}" 2>&1 | tee "${run_dir}/train.log"
  else
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone --nproc_per_node=2 "${ROOT}/run_afmr.py" train \
      "${config_path}" "${resume_args[@]}" 2>&1 | tee "${run_dir}/train.log"
  fi

  [[ -s "${run_dir}/resolved_config.yaml" && -s "${run_dir}/last.pt" ]] || \
    die "Training did not produce resolved_config.yaml and last.pt in ${run_dir}"

  if (( ${#GPU_LIST[@]} == 1 )); then
    echo "=== Test evaluation on GPU ${GPU_LIST[0]} ==="
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
      "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${predictions}" \
      --split test --batch-size "${EVAL_BATCH_SIZE}" 2>&1 | tee "${run_dir}/evaluate_test.log"
  else
    echo "=== Test evaluation sharded across GPUs ${GPU_IDS} ==="
    CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
      "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${run_dir}/shard0.jsonl" \
      --split test --batch-size "${EVAL_BATCH_SIZE}" --shard-rank 0 --num-shards 2 \
      >"${run_dir}/eval_shard0.log" 2>&1 &
    local pid0=$!
    CUDA_VISIBLE_DEVICES="${GPU_LIST[1]}" "${PYTHON_BIN}" "${ROOT}/run_afmr.py" evaluate \
      "${run_dir}/resolved_config.yaml" "${run_dir}/last.pt" "${run_dir}/shard1.jsonl" \
      --split test --batch-size "${EVAL_BATCH_SIZE}" --shard-rank 1 --num-shards 2 \
      >"${run_dir}/eval_shard1.log" 2>&1 &
    local pid1=$!
    local eval_failed=0
    wait "${pid0}" || eval_failed=1
    wait "${pid1}" || eval_failed=1
    if (( eval_failed != 0 )); then
      cat "${run_dir}/eval_shard0.log" "${run_dir}/eval_shard1.log" >&2 || true
      die "Test evaluation failed for ${ABLATION}, seed ${seed}"
    fi
    "${PYTHON_BIN}" "${ROOT}/scripts/merge_eval_shards.py" \
      --output "${predictions}" "${run_dir}/shard0.jsonl" "${run_dir}/shard1.jsonl"
  fi

  echo "=== ROUGE evaluation: ${ABLATION}, seed ${seed} ==="
  "${PYTHON_BIN}" "${ROUGE_SCRIPT}" "${predictions}"
  echo "=== Completed ${ABLATION}, seed ${seed}; output: ${run_dir} ==="
}

for seed in "${SEED_LIST[@]}"; do
  run_one_seed "${seed}"
done

echo "=== Completed ${ABLATION}; runs and predictions: ${RUN_ROOT} ==="
