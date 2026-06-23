#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Try to load environment if available
if [[ -f "${SCRIPT_DIR}/scripts/load_env.sh" ]]; then
    source "${SCRIPT_DIR}/scripts/load_env.sh"
fi

# Define paths and configs
HF_REPO_ID="${HF_REPO_ID:-BienKieu/llm2seq-wikilingua}"
PHASE2_HF_PATH="checkpoints/h200_phase2_lora_encoder/best.pt"
LOCAL_PHASE2_DIR="runs/h200_llm2seq_phase2_lora_encoder"

PHASE3_CONFIG="${PHASE3_CONFIG:-configs/phase3_mtp_self_distill_4096.yaml}"
PHASE3_DIR="${PHASE3_DIR:-runs/h200_llm2seq_phase3_mtp_self_distill}"

EVAL_ROOT="${EVAL_ROOT:-eval_outputs}"
PHASE3_EVAL_DIR="${PHASE3_EVAL_DIR:-${EVAL_ROOT}/full_test_phase3_main}"
PHASE3_MTP_EVAL_DIR="${PHASE3_MTP_EVAL_DIR:-${EVAL_ROOT}/full_test_phase3_mtp_verified}"
PHASE3_SPEED_COMPARE_DIR="${PHASE3_SPEED_COMPARE_DIR:-${EVAL_ROOT}/phase3_speed_comparison}"

LOG_DIR="${LOG_DIR:-logs}"
RUN_PHASE_EVAL="${RUN_PHASE_EVAL:-true}"

mkdir -p "${LOG_DIR}"
mkdir -p "${LOCAL_PHASE2_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"

DATA_DIR="data/processed"
WIKI_DIR="wikilingua"

# 0. Prepare dataset if it does not exist
if [[ ! -f "${DATA_DIR}/train.jsonl" ]]; then
    echo "=== Prepare WikiLingua Dataset ==="
    python3 scripts/prepare_wikilingua_json.py \
      --input_dir "${WIKI_DIR}" \
      --output_dir "${DATA_DIR}" \
      --max_train -1 \
      --max_eval -1 \
      --source_joiner "\n" \
      --target_joiner " " \
      2>&1 | tee "${LOG_DIR}/${STAMP}_prepare.log"
fi

# 1. Download Phase 2 model from Hugging Face if not exists locally
if [[ ! -f "${LOCAL_PHASE2_DIR}/best.pt" ]]; then
    echo "=== Downloading Phase 2 best.pt from HuggingFace (${HF_REPO_ID}) ==="
    python3 -c "
from huggingface_hub import hf_hub_download
import os
import shutil

print('Starting download...')
file_path = hf_hub_download(repo_id='${HF_REPO_ID}', filename='${PHASE2_HF_PATH}', local_dir='.', local_dir_use_symlinks=False)
target_path = '${LOCAL_PHASE2_DIR}/best.pt'
os.makedirs(os.path.dirname(target_path), exist_ok=True)
shutil.move(file_path, target_path)
print(f'Successfully downloaded and moved to {target_path}')
"
fi

echo "=== Train Phase 3 ==="
bash scripts/train_phase3.sh "${LOCAL_PHASE2_DIR}/best.pt" "${PHASE3_CONFIG}" \
  2>&1 | tee "${LOG_DIR}/${STAMP}_phase3.log"

if [[ "${RUN_PHASE_EVAL}" == "true" ]]; then
  echo "=== Evaluate Phase 3 (Main - Autoregressive) ==="
  bash scripts/evaluate_phase.sh \
    phase3_main \
    "${PHASE3_CONFIG}" \
    "${PHASE3_DIR}/best.pt" \
    "${PHASE3_EVAL_DIR}" \
    autoregressive \
    "${LOCAL_PHASE2_DIR}/best.pt"

  echo "=== Evaluate Phase 3 (MTP Verified) ==="
  bash scripts/evaluate_phase.sh \
    phase3_mtp \
    "${PHASE3_CONFIG}" \
    "${PHASE3_DIR}/best.pt" \
    "${PHASE3_MTP_EVAL_DIR}" \
    mtp_verified \
    "${LOCAL_PHASE2_DIR}/best.pt"

  echo "=== Compare Speed Metrics ==="
  python3 scripts/compare_speed_metrics.py \
    --main_metrics "${PHASE3_EVAL_DIR}/metrics.json" \
    --mtp_metrics "${PHASE3_MTP_EVAL_DIR}/metrics.json" \
    --output_dir "${PHASE3_SPEED_COMPARE_DIR}" \
    2>&1 | tee "${LOG_DIR}/${STAMP}_speed_compare_phase3.log"
fi

echo "=== Phase 3 Execution Completed ==="
