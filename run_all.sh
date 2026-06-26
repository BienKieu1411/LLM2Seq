#!/usr/bin/env bash
set -euo pipefail

# Setup environment
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PWD}/llm2seq:${PWD}:${PYTHONPATH:-}"

# Load user environment variables (contains HF_REPO_ID, tokens, etc.)
if [ -f "T5Gemma/scripts/load_env.sh" ]; then
    source T5Gemma/scripts/load_env.sh
fi
if [ -f "llm2seq/scripts/load_env.sh" ]; then
    source llm2seq/scripts/load_env.sh
fi
echo "==========================================="
echo "5. QWEN WIKILINGUA PHASE 3 (MTP)"
echo "==========================================="
echo "--> Preparing WikiLingua Data"
"${PYTHON_BIN}" llm2seq/scripts/prepare_wikilingua_json.py --input_dir llm2seq/datasets/wikilingua

echo "--> Downloading Qwen Phase 2 checkpoint from Hugging Face..."
huggingface-cli download BienKieu/llm2seq-wikilingua \
  checkpoints/phase2_lora_encoder_qwen_wikilingua/best.pt \
  --local-dir runs/qwen_wiki_phase2_download

echo "--> Train Qwen Phase 3"
"${PYTHON_BIN}" -m src.training.trainer \
  --config llm2seq/configs/wikilingua_qwen_phase3.yaml \
  --resume runs/qwen_wiki_phase2_download/checkpoints/phase2_lora_encoder_qwen_wikilingua/best.pt

echo "--> Eval Qwen Phase 3 (Autoregressive Baseline)"
"${PYTHON_BIN}" llm2seq/scripts/evaluate_full_test.py \
  --config llm2seq/configs/wikilingua_qwen_phase3.yaml \
  --checkpoint runs/phase3_mtp_self_distill_qwen_wikilingua/best.pt \
  --base_checkpoint runs/qwen_wiki_phase2_download/checkpoints/phase2_lora_encoder_qwen_wikilingua/best.pt \
  --test_file llm2seq/data/processed/test.jsonl \
  --output_dir runs/phase3_mtp_self_distill_qwen_wikilingua/eval_ar \
  --decode_mode autoregressive

echo "--> Eval Qwen Phase 3 (MTP Verified)"
"${PYTHON_BIN}" llm2seq/scripts/evaluate_full_test.py \
  --config llm2seq/configs/wikilingua_qwen_phase3.yaml \
  --checkpoint runs/phase3_mtp_self_distill_qwen_wikilingua/best.pt \
  --base_checkpoint runs/qwen_wiki_phase2_download/checkpoints/phase2_lora_encoder_qwen_wikilingua/best.pt \
  --test_file llm2seq/data/processed/test.jsonl \
  --output_dir runs/phase3_mtp_self_distill_qwen_wikilingua/eval_mtp \
  --decode_mode mtp_verified

echo "--> Comparing Speed Metrics (AR vs MTP)"
"${PYTHON_BIN}" llm2seq/scripts/compare_speed_metrics.py \
  --main_metrics runs/phase3_mtp_self_distill_qwen_wikilingua/eval_ar/metrics.json \
  --mtp_metrics runs/phase3_mtp_self_distill_qwen_wikilingua/eval_mtp/metrics.json \
  --output_dir runs/phase3_mtp_self_distill_qwen_wikilingua/compare

echo "==========================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "==========================================="
