#!/usr/bin/env bash
# ==============================================================================
# Script: Train 3 Epochs on 2 GPUs (DDP) -> Generate Predictions (vLLM Parallel)
# ==============================================================================
set -euo pipefail

# Project root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-$PWD/src/eviseq_new/runs/afmr/train_si_v2}"
TRAIN_CONFIG="${TRAIN_CONFIG:-src/eviseq_new/runs/afmr/train/resolved_config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-src/eviseq_new/runs/afmr/train/epoch_004.pt}"
TRAIN_FILE="${TRAIN_FILE:-src/eviseq_new/datasets/seam_si_v2/train_si.jsonl}"
AFMR_CONFIG="${AFMR_CONFIG:-src/eviseq_new/runs/afmr/train_si/resolved_config.yaml}"

# Resolve evaluator script (supports both src/evaluation and src/rouge155)
EVAL_SCRIPT="src/evaluation/evaluate_rouge_vietnamese_vllm.py"
if [[ ! -f "$EVAL_SCRIPT" && -f "src/rouge155/evaluate_rouge_vietnamese_vllm.py" ]]; then
  EVAL_SCRIPT="src/rouge155/evaluate_rouge_vietnamese_vllm.py"
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo " [PHASE 1] Starting DDP Training (3 Epochs on 2 GPUs: 0, 1)"
echo " Output dir: $OUTPUT_DIR"
echo " Train file: $TRAIN_FILE"
echo " Log file:   ${LOG_DIR}/train.log"
echo "========================================================================"

CUDA_VISIBLE_DEVICES=0,1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python3 -m torch.distributed.run --standalone --nproc_per_node=2 \
  src/eviseq_new/run_afmr.py train \
  "$TRAIN_CONFIG" \
  --init-checkpoint "$INIT_CHECKPOINT" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3 \
  --batch-size 32 \
  --accum 1 2>&1 | tee "${LOG_DIR}/train.log"

echo ">>> Training finished successfully!"

# If newly generated resolved_config exists in train_si_v2, prefer it
if [[ -f "${OUTPUT_DIR}/resolved_config.yaml" ]]; then
  AFMR_CONFIG="${OUTPUT_DIR}/resolved_config.yaml"
fi

echo ""
echo "========================================================================"
echo " [PHASE 2] Parallel Prediction on 2 GPUs (--skip-scoring: no ROUGE computation)"
echo " GPU 0 -> Epoch 001 (Port 8011)"
echo " GPU 1 -> Epoch 002 (Port 8012)"
echo "========================================================================"

# Launch Epoch 001 on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 "$EVAL_SCRIPT" \
  --start-server \
  --server-port 8011 \
  --afmr-config "$AFMR_CONFIG" \
  --afmr-checkpoint "${OUTPUT_DIR}/epoch_001.pt" \
  --model-name seam_si_v2_001 \
  --concurrency 256 \
  --gpu-memory-utilization 0.85 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --allow-missing \
  --attention-backend FLASH_ATTN \
  --disable-flashinfer-sampler \
  --skip-scoring > "${LOG_DIR}/predict_epoch_001.log" 2>&1 &
PID_001=$!
echo ">>> Dispatched Epoch 001 prediction to GPU 0 (PID $PID_001, log: ${LOG_DIR}/predict_epoch_001.log)"

# Launch Epoch 002 on GPU 1
CUDA_VISIBLE_DEVICES=1 python3 "$EVAL_SCRIPT" \
  --start-server \
  --server-port 8012 \
  --afmr-config "$AFMR_CONFIG" \
  --afmr-checkpoint "${OUTPUT_DIR}/epoch_002.pt" \
  --model-name seam_si_v2_002 \
  --concurrency 256 \
  --gpu-memory-utilization 0.85 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --allow-missing \
  --attention-backend FLASH_ATTN \
  --disable-flashinfer-sampler \
  --skip-scoring > "${LOG_DIR}/predict_epoch_002.log" 2>&1 &
PID_002=$!
echo ">>> Dispatched Epoch 002 prediction to GPU 1 (PID $PID_002, log: ${LOG_DIR}/predict_epoch_002.log)"

echo "Waiting for Epoch 001 and Epoch 002 predictions to finish..."
wait "$PID_001"
echo ">>> Epoch 001 prediction finished!"
wait "$PID_002"
echo ">>> Epoch 002 prediction finished!"

echo ""
echo "========================================================================"
echo " [PHASE 3] Sequential Prediction for Epoch 003 on GPU 0 (Port 8013)"
echo "========================================================================"

CUDA_VISIBLE_DEVICES=0 python3 "$EVAL_SCRIPT" \
  --start-server \
  --server-port 8013 \
  --afmr-config "$AFMR_CONFIG" \
  --afmr-checkpoint "${OUTPUT_DIR}/epoch_003.pt" \
  --model-name seam_si_v2_003 \
  --concurrency 256 \
  --gpu-memory-utilization 0.85 \
  --temperature 0 \
  --top-p 1 \
  --top-k 0 \
  --allow-missing \
  --attention-backend FLASH_ATTN \
  --disable-flashinfer-sampler \
  --skip-scoring 2>&1 | tee "${LOG_DIR}/predict_epoch_003.log"

echo ""
echo "========================================================================"
echo " [COMPLETE] All predictions for Epoch 001, 002, 003 have been generated!"
echo " Predictions are saved in:"
echo "   - data_sum/seam_si_v2_001"
echo "   - data_sum/seam_si_v2_002"
echo "   - data_sum/seam_si_v2_003"
echo ""
echo " To score ROUGE later, run:"
echo "   python3 src/evaluation/evaluate_rouge_vietnamese.py --model-name seam_si_v2_001"
echo "   python3 src/evaluation/evaluate_rouge_vietnamese.py --model-name seam_si_v2_002"
echo "   python3 src/evaluation/evaluate_rouge_vietnamese.py --model-name seam_si_v2_003"
echo "========================================================================"
