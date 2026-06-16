#!/bin/bash
# Evaluate LLM2Seq model with all metrics
set -euo pipefail

CHECKPOINT="${1:-runs/llm2seq_kd_mtp_full/best.pt}"
TEST_FILE="${2:-data/processed/test.jsonl}"
USE_MTP="${3:-true}"

echo "=== LLM2Seq Evaluation ==="
echo "Checkpoint: $CHECKPOINT"
echo "Test file:  $TEST_FILE"
echo "Use MTP:    $USE_MTP"
echo "=========================="

python -c "
import torch
import yaml
from llm2seq.src.models.llm2seq_model import LLM2Seq, LLM2SeqConfig
from llm2seq.src.data.dataset import Seq2SeqDataset
from llm2seq.src.data.collator import Seq2SeqCollator
from llm2seq.src.inference.generate import autoregressive_generate
from llm2seq.src.inference.generate_mtp import mtp_generate
from llm2seq.src.eval.eval_bleu import evaluate_bleu
from llm2seq.src.eval.eval_rouge import evaluate_rouge
from llm2seq.src.eval.eval_latency import evaluate_latency
from transformers import AutoTokenizer
import json

# Load checkpoint
ckpt = torch.load('$CHECKPOINT', map_location='cpu')
print('Checkpoint loaded.')

# TODO: Add full evaluation pipeline
# This script serves as a template for evaluation
print('Evaluation script ready. Implement full pipeline as needed.')
"

echo "Evaluation complete."
