#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo " Starting all experiments "
echo "=========================================================================="

echo "--------------------------------------------------------------------------"
echo " 1. Fine-tune T5Gemma on WikiLingua "
echo "--------------------------------------------------------------------------"
echo "Setting up environment for T5Gemma..."
python3 -m pip install -r T5Gemma/requirements.txt

echo "Preparing WikiLingua data for T5Gemma..."
python3 T5Gemma/scripts/prepare_wikilingua_json.py --input_dir T5Gemma/datasets/wikilingua --output_dir T5Gemma/data/processed
echo "Training T5Gemma on WikiLingua..."
CONFIG="T5Gemma/configs/wikilingua_lora_3072.yaml" bash T5Gemma/scripts/train.sh
echo "Evaluating T5Gemma on WikiLingua..."
CONFIG="T5Gemma/configs/wikilingua_lora_3072.yaml" bash T5Gemma/scripts/evaluate.sh

echo "--------------------------------------------------------------------------"
echo " 2. Fine-tune T5Gemma on VLSP "
echo "--------------------------------------------------------------------------"
echo "Preparing VLSP data for T5Gemma..."
python3 T5Gemma/scripts/prepare_vlsp_json.py --input_dir T5Gemma/datasets/vlsp --output_dir T5Gemma/data/processed/vlsp
echo "Training T5Gemma on VLSP..."
CONFIG="T5Gemma/configs/vlsp_lora.yaml" bash T5Gemma/scripts/train.sh
echo "Evaluating T5Gemma on VLSP..."
CONFIG="T5Gemma/configs/vlsp_lora.yaml" bash T5Gemma/scripts/evaluate.sh


echo "--------------------------------------------------------------------------"
echo " 3. Fine-tune llm2seq 3 phases on VLSP "
echo "--------------------------------------------------------------------------"
echo "Switching environment for llm2seq (downgrading transformers)..."
python3 -m pip install -r llm2seq/requirements.txt

echo "Preparing VLSP data for llm2seq..."
python3 llm2seq/scripts/prepare_vlsp_json.py --input_dir llm2seq/datasets/vlsp --output_dir llm2seq/data/processed/vlsp
echo "Running llm2seq Phase 1 on VLSP..."
bash llm2seq/scripts/train_phase1.sh "llm2seq/configs/vlsp_phase1.yaml"

echo "Running llm2seq Phase 2 on VLSP..."
bash llm2seq/scripts/train_phase2.sh "runs/llm2seq_phase1_warmup_vlsp/best.pt" "llm2seq/configs/vlsp_phase2.yaml"
echo "Evaluating llm2seq Phase 2 on VLSP..."
bash llm2seq/scripts/evaluate_phase.sh phase2_main "llm2seq/configs/vlsp_phase2.yaml" "runs/llm2seq_phase2_lora_encoder_vlsp/best.pt" "llm2seq/eval_outputs/vlsp_full_test_phase2_main" autoregressive

echo "Running llm2seq Phase 3 on VLSP..."
bash llm2seq/scripts/train_phase3.sh "runs/llm2seq_phase2_lora_encoder_vlsp/best.pt" "llm2seq/configs/vlsp_phase3.yaml"
echo "Evaluating llm2seq Phase 3 on VLSP..."
bash llm2seq/scripts/evaluate_phase.sh phase3_main "llm2seq/configs/vlsp_phase3.yaml" "runs/llm2seq_phase3_mtp_self_distill_vlsp/best.pt" "llm2seq/eval_outputs/vlsp_full_test_phase3_main" autoregressive "runs/llm2seq_phase2_lora_encoder_vlsp/best.pt"
bash llm2seq/scripts/evaluate_phase.sh phase3_mtp "llm2seq/configs/vlsp_phase3.yaml" "runs/llm2seq_phase3_mtp_self_distill_vlsp/best.pt" "llm2seq/eval_outputs/vlsp_full_test_phase3_mtp_verified" mtp_verified "runs/llm2seq_phase2_lora_encoder_vlsp/best.pt"
python3 llm2seq/scripts/compare_speed_metrics.py \
  --main_metrics "llm2seq/eval_outputs/vlsp_full_test_phase3_main/metrics.json" \
  --mtp_metrics "llm2seq/eval_outputs/vlsp_full_test_phase3_mtp_verified/metrics.json" \
  --output_dir "llm2seq/eval_outputs/vlsp_phase3_speed_comparison"


echo "--------------------------------------------------------------------------"
echo " 4. Fine-tune phase 3 llm2seq WikiLingua (Optional / Remaining GPU time) "
echo "--------------------------------------------------------------------------"
echo "Preparing WikiLingua data for llm2seq..."
python3 llm2seq/scripts/prepare_wikilingua_json.py --input_dir llm2seq/datasets/wikilingua --output_dir llm2seq/data/processed
PHASE2_WIKI_DIR="runs/llm2seq_llm2seq_phase2_lora_encoder"
HF_REPO_ID="${HF_REPO_ID:-BienKieu/llm2seq-wikilingua}"
PHASE2_HF_PATH="checkpoints/llm2seq_phase2_lora_encoder/best.pt"

if [[ ! -f "${PHASE2_WIKI_DIR}/best.pt" ]]; then
    echo "=== Downloading Phase 2 best.pt from HuggingFace (${HF_REPO_ID}) ==="
    python3 -c "
from huggingface_hub import hf_hub_download
import os
import shutil

print('Starting download...')
file_path = hf_hub_download(repo_id='${HF_REPO_ID}', filename='${PHASE2_HF_PATH}', local_dir='.', local_dir_use_symlinks=False)
target_path = '${PHASE2_WIKI_DIR}/best.pt'
os.makedirs(os.path.dirname(target_path), exist_ok=True)
shutil.move(file_path, target_path)
print(f'Successfully downloaded and moved to {target_path}')
"
fi

if [[ -f "${PHASE2_WIKI_DIR}/best.pt" ]]; then
    echo "Found best.pt from phase 2 wikilingua. Proceeding to phase 3."
    PHASE3_CONFIG="llm2seq/configs/wikilingua_phase3.yaml"
    PHASE3_DIR="runs/phase3_mtp_self_distill_wiki"
    bash llm2seq/scripts/train_phase3.sh "${PHASE2_WIKI_DIR}/best.pt" "${PHASE3_CONFIG}"
    
    echo "Evaluating phase 3 llm2seq WikiLingua..."
    bash llm2seq/scripts/evaluate_phase.sh phase3_main "${PHASE3_CONFIG}" "${PHASE3_DIR}/best.pt" "llm2seq/eval_outputs/wiki_full_test_phase3_main" autoregressive "${PHASE2_WIKI_DIR}/best.pt"
    bash llm2seq/scripts/evaluate_phase.sh phase3_mtp "${PHASE3_CONFIG}" "${PHASE3_DIR}/best.pt" "llm2seq/eval_outputs/wiki_full_test_phase3_mtp_verified" mtp_verified "${PHASE2_WIKI_DIR}/best.pt"
    
    python3 llm2seq/scripts/compare_speed_metrics.py \
      --main_metrics "llm2seq/eval_outputs/wiki_full_test_phase3_main/metrics.json" \
      --mtp_metrics "llm2seq/eval_outputs/wiki_full_test_phase3_mtp_verified/metrics.json" \
      --output_dir "llm2seq/eval_outputs/wiki_phase3_speed_comparison"
else
    echo "Could not find or download ${PHASE2_WIKI_DIR}/best.pt. Skipping phase 3 for WikiLingua."
fi

echo "=========================================================================="
echo " All tasks completed successfully! "
echo "=========================================================================="
