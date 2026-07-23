#!/usr/bin/env bash
set -euo pipefail

# One launcher for the B200 paper runs.
# Run this file from anywhere; paths are resolved relative to this script.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

export PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
MODE="${1:-help}"

run_genbridge() {
  echo "=== Train GenBridge and evaluate best.pt + last.pt ==="
  args=(pipeline --config configs/qwen3_0_6b.yaml)
  if [[ "${OVERWRITE_GENBRIDGE:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash genbridge/run.sh "${args[@]}"
}

eval_genbridge() {
  echo "=== Evaluate existing GenBridge best.pt + last.pt ==="
  bash genbridge/run.sh pipeline \
    --config configs/qwen3_0_6b.yaml \
    --eval-only
}

run_t5gemma() {
  bash T5Gemma/run_pipeline.sh
}

run_t5gemma_4b() {
  bash T5Gemma/run_4b_pipeline.sh
}

compare_models() {
  echo "=== Paired comparison: GenBridge best vs T5Gemma ==="
  bash genbridge/run.sh compare \
    --candidate runs/genbridge/qwen3_0_6b/best_test_predictions.jsonl \
    --baseline ../T5Gemma/eval_outputs/full_test/predictions.jsonl \
    --output runs/comparisons/genbridge_best_vs_t5gemma.json

  echo "=== Paired comparison: GenBridge last vs T5Gemma ==="
  bash genbridge/run.sh compare \
    --candidate runs/genbridge/qwen3_0_6b/last_test_predictions.jsonl \
    --baseline ../T5Gemma/eval_outputs/full_test/predictions.jsonl \
    --output runs/comparisons/genbridge_last_vs_t5gemma.json
}

compare_models_4b() {
  echo "=== Paired comparison: GenBridge best vs T5Gemma 4B-4B ==="
  bash genbridge/run.sh compare \
    --candidate runs/genbridge/qwen3_0_6b/best_test_predictions.jsonl \
    --baseline ../T5Gemma/eval_outputs/full_test_4b_4b/predictions.jsonl \
    --output runs/comparisons/genbridge_best_vs_t5gemma_4b_4b.json

  echo "=== Paired comparison: GenBridge last vs T5Gemma 4B-4B ==="
  bash genbridge/run.sh compare \
    --candidate runs/genbridge/qwen3_0_6b/last_test_predictions.jsonl \
    --baseline ../T5Gemma/eval_outputs/full_test_4b_4b/predictions.jsonl \
    --output runs/comparisons/genbridge_last_vs_t5gemma_4b_4b.json
}

run_ablation_group() {
  local group="$1"
  local model_size="${MODEL_SIZE:-0.6B}"
  echo "=== GenBridge ablation group=${group}, model_size=${model_size} ==="
  args=(ablate --group "${group}" --model-size "${model_size}" --evaluate)
  if [[ "${OVERWRITE_ABLATIONS:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash genbridge/run.sh "${args[@]}"
}

show_parameter_locations() {
  cat <<'EOF'
GENBRIDGE PARAMETER LOCATIONS

1. Shared architecture/training defaults:
   genbridge/configs/base.yaml

   training.epochs                       total epochs (currently 10)
   training.interface_warmup_epochs      frozen-backbone warm-up (currently 2)
   training.batch_size                   physical train batch
   training.gradient_accumulation_steps  gradient accumulation
   training.adapter_warmup_lr             warm-up LR
   training.full_lr                       full-finetune LR
   training.eval_batch_size               validation batch
   training.num_workers                   train dataloader workers
   training.eval_num_workers              validation workers
   bridge.token_num_layers                bidirectional adapter depth (main=4)
   bridge.hidden_size                     adapter width
   bridge.ffn_size                         adapter FFN width
   model.num_summary_tokens               number of plan tokens
   decoder.cross_attention_every          cross-attention frequency
   data.max_source_length                 source-token limit
   data.max_target_length                 training target-token limit
   generation.batch_size                  test inference batch
   generation.max_new_tokens              maximum generated tokens

2. Model-size overrides:
   genbridge/configs/qwen3_0_6b.yaml
   genbridge/configs/qwen3_1_7b.yaml
   genbridge/configs/qwen3_4b.yaml

   These files override model names, decoder layers, physical batch,
   accumulation, full LR, optimizer, and inference batch for each scale.

3. Dataset-specific settings:
   genbridge/configs/datasets/wikilingua.yaml
   genbridge/configs/datasets/lrsum.yaml

4. Ablation definitions:
   genbridge/configs/ablations/*.yaml

   adapter_2layers.yaml / adapter_8layers.yaml change adapter depth.
   lamate_style.yaml, causal_ed.yaml, hierarchical.yaml,
   no_salience_loss.yaml, no_memory_curriculum.yaml,
   no_salience_attention_bias.yaml, no_plan_evidence_alignment.yaml,
   plan_only.yaml, concat_memory.yaml and no_adapter_rope.yaml isolate
   individual contributions.

5. T5Gemma WikiLingua:
   T5Gemma/configs/wikilingua_full_3072.yaml

   training.num_train_epochs               epochs
   training.per_device_train_batch_size    physical batch
   training.gradient_accumulation_steps    accumulation
   training.learning_rate                  full-finetune LR
   data.max_source_length                  source limit
   data.max_target_length                  target limit
   generation.eval_batch_size              inference batch
   generation.max_new_tokens               generation limit

6. T5Gemma paths/tokens/overwrite:
   T5Gemma/env.txt

Do not change parameters in resolved_config.yaml after training starts. It is
the immutable record of the run, not the source configuration.
EOF
}

usage() {
  cat <<'EOF'
Usage: bash run.sh MODE

Modes:
  setup       Install dependencies and run environment/tests.
  genbridge   Train GenBridge, then evaluate best.pt and last.pt.
  eval        Evaluate existing GenBridge best.pt and last.pt only.
  t5gemma     Prepare data, full-finetune, and evaluate T5Gemma.
  t5gemma-4b  Full-finetune and evaluate google/t5gemma-2-4b-4b.
  compare     Compare both GenBridge checkpoints with T5Gemma.
  compare-4b  Compare both GenBridge checkpoints with T5Gemma 4B-4B.
  all         Run GenBridge, both T5Gemma scales, then comparisons (no ablations).
  ablation-pilot     Fast architecture gate: 5 essential configurations.
  ablation-main      Main paper ablation table: 11 configurations.
  ablation-analysis  Adapter-depth/RoPE/fusion analysis configurations.
  ablation-all       Every unique registered ablation configuration.
  params      Print exactly where batch size, epochs, LR, etc. are configured.

Before T5Gemma/all:
  cp T5Gemma/env.example.txt T5Gemma/env.txt
  # Edit HF_TOKEN and PYTHON_BIN in T5Gemma/env.txt.

Intentional GenBridge rerun:
  OVERWRITE_GENBRIDGE=true bash run.sh genbridge

T5Gemma overwrite is controlled by OVERWRITE_OUTPUT_DIR in T5Gemma/env.txt.

Ablation examples:
  MODEL_SIZE=0.6B bash run.sh ablation-pilot
  MODEL_SIZE=0.6B bash run.sh ablation-main
  MODEL_SIZE=0.6B bash run.sh ablation-analysis
  OVERWRITE_ABLATIONS=true MODEL_SIZE=0.6B bash run.sh ablation-main

MODEL_SIZE may be 0.6B, 1.7B, or 4B. Run 0.6B ablations first; the main 1.7B
or 4B paper model does not require repeating every expensive ablation.
EOF
}

case "${MODE}" in
  setup)
    setup
    ;;
  genbridge)
    run_genbridge
    ;;
  eval)
    eval_genbridge
    ;;
  t5gemma)
    run_t5gemma
    ;;
  t5gemma-4b)
    run_t5gemma_4b
    ;;
  compare)
    compare_models
    ;;
  compare-4b)
    compare_models_4b
    ;;
  ablation-pilot)
    run_ablation_group pilot
    ;;
  ablation-main)
    run_ablation_group main
    ;;
  ablation-analysis)
    run_ablation_group analysis
    ;;
  ablation-all)
    run_ablation_group all
    ;;
  params)
    show_parameter_locations
    ;;
  all)
    run_genbridge
    run_t5gemma
    run_t5gemma_4b
    compare_models
    compare_models_4b
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
