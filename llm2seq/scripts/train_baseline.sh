#!/bin/bash
# Train LLM2Seq Baseline: Encoder (frozen) + Adaptor + Decoder with CE loss only
set -euo pipefail

echo "=== LLM2Seq Baseline Training ==="
echo "MTP: OFF | Distillation: OFF"
echo "================================="

python -m llm2seq.src.training.trainer \
    --config llm2seq/configs/baseline.yaml

echo "Done! Baseline training complete."
