#!/bin/bash
# Train LLM2Seq with Knowledge Distillation only (resume from baseline)
set -euo pipefail

echo "=== LLM2Seq KD-Only Training ==="
echo "MTP: OFF | Distillation: ON (top-k KL)"
echo "================================="

python -m llm2seq.src.training.trainer \
    --config llm2seq/configs/kd_only.yaml

echo "Done! KD-only training complete."
