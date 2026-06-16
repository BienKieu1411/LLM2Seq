#!/bin/bash
# Train LLM2Seq Full: KD + MTP (resume from KD-only best)
set -euo pipefail

echo "=== LLM2Seq Full Training (KD + MTP) ==="
echo "MTP: ON (cascaded) | Distillation: ON (top-k KL)"
echo "================================================"

python -m llm2seq.src.training.trainer \
    --config llm2seq/configs/kd_mtp_full.yaml

echo "Done! Full KD+MTP training complete."
