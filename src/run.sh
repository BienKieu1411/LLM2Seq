#!/usr/bin/env bash
set -euo pipefail

# One launcher for the B200 paper runs.
# Run this file from anywhere; paths are resolved relative to this script.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

DEFAULT_PYTHON="/Users/kieugiangbien/bienkieu_env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
fi
export PYTHON_BIN
MODE="${1:-help}"

run_genbridge() {
  local config="${1:-${GENBRIDGE_CONFIG:-configs/qwen3_enc1_7b_dec0_6b.yaml}}"
  echo "=== Train GenBridge (${config}) and evaluate best.pt + last.pt ==="
  args=(pipeline --config "${config}")
  if [[ "${OVERWRITE_GENBRIDGE:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash genbridge/run.sh "${args[@]}"
}

eval_genbridge() {
  local config="${1:-${GENBRIDGE_CONFIG:-configs/qwen3_enc1_7b_dec0_6b.yaml}}"
  echo "=== Evaluate existing GenBridge (${config}) best.pt + last.pt ==="
  bash genbridge/run.sh pipeline \
    --config "${config}" \
    --eval-only
}

run_t5gemma_4b() {
  FORCE_EVAL=true bash T5Gemma/run_4b_pipeline.sh
}

run_t5gemma_cnndm() {
  local scale="${1:-all}"
  bash T5Gemma/run_cnndm_pipeline.sh "${scale}"
}

run_llm2seq_v2() {
  local mode="$1"
  local -a args
  echo "=== LLM2Seq-v2: ${mode} ==="
  args=("${mode}")
  if [[ "${OVERWRITE_LLM2SEQ_V2:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash llm2seq_v2/run.sh "${args[@]}"
}

run_llm2seq_v2_main() {
  # Each pipeline saves only last.pt and evaluates that checkpoint on the
  # complete test split before the next model configuration starts.
  run_llm2seq_v2 pipeline
  run_llm2seq_v2 decoder-1.7b
}

run_llm2seq_v2_all() {
  run_llm2seq_v2_main
  run_llm2seq_v2 ablation-all
}

run_llm2seq_v3() {
  local mode="$1"
  local -a args
  echo "=== LLM2Seq-v3: ${mode} ==="
  args=("${mode}")
  if [[ "${OVERWRITE_LLM2SEQ_V3:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash llm2seq_v3/run.sh "${args[@]}"
}

run_llm2seq_v4() {
  local mode="$1"
  local -a args
  echo "=== LLM2Seq-v4 PSB: ${mode} ==="
  args=("${mode}")
  if [[ "${OVERWRITE_LLM2SEQ_V4:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash llm2seq_v4/run.sh "${args[@]}"
}

run_llm2seq_v5() {
  local mode="$1"
  local -a args
  echo "=== LLM2Seq-v5: ${mode} ==="
  args=("${mode}")
  if [[ "${OVERWRITE_LLM2SEQ_V5:-false}" =~ ^(true|1|yes)$ ]]; then
    args+=(--overwrite-output-dir)
  fi
  bash llm2seq_v5/run.sh "${args[@]}"
}

run_paper_sequence() {
  # Fail-fast order for the B200 allocation:
  # CNN/DailyMail runs separately via run_cnndm_t5gemma.sh on another GPU.
  #   1. Qwen3-Embedding 0.6B -> Qwen3 0.6B LLM2Seq-v2 + evaluation.
  #   2. T5Gemma 4B-4B WikiLingua full fine-tune + evaluation.
  #   3. Qwen3-Embedding 0.6B -> Qwen3 1.7B LLM2Seq-v2 + evaluation.
  #   4. Every LLM2Seq-v2 ablation, each followed by evaluation.
  run_llm2seq_v2 pipeline
  run_t5gemma_4b
  run_llm2seq_v2 decoder-1.7b
  run_llm2seq_v2 ablation-all
}

run_legacy_remaining() {
  # Completed already: GenBridge 0.6B-0.6B and T5Gemma 1B-1B.
  # Each GenBridge pipeline evaluates its own best.pt and last.pt before the
  # next configuration starts. T5Gemma 4B-4B is forced to evaluate final_model.
  run_genbridge configs/qwen3_enc1_7b_dec0_6b.yaml
  # Run every unique 0.6B ablation and evaluate its validation-selected best.pt.
  # Skip "genbridge" because the completed qwen3_0_6b run is the identical
  # full-model control after applying the shared 0.6B profile.
  run_ablation_group all genbridge
  run_genbridge configs/qwen3_enc0_6b_dec1_7b.yaml
  run_genbridge configs/qwen3_1_7b.yaml
  run_genbridge configs/qwen3_4b.yaml
  run_t5gemma_4b
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
  shift
  local model_size="${MODEL_SIZE:-0.6B}"
  echo "=== GenBridge ablation group=${group}, model_size=${model_size} ==="
  args=(ablate --group "${group}" --model-size "${model_size}" --evaluate)
  local skipped
  for skipped in "$@"; do
    args+=(--skip "${skipped}")
  done
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
   genbridge/configs/qwen3_enc1_7b_dec0_6b.yaml
   genbridge/configs/qwen3_enc0_6b_dec1_7b.yaml

   These files override model names, decoder layers, physical batch,
   accumulation, full LR, optimizer, and inference batch for each scale.
   Mixed-config filenames always list encoder size before decoder size.

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
  genbridge   Train the default remaining GenBridge 1.7B-0.6B, then evaluate
              its own best.pt and last.pt.
  genbridge-1.7-0.6  Qwen3 encoder 1.7B + decoder 0.6B.
  genbridge-0.6-1.7  Qwen3 encoder 0.6B + decoder 1.7B.
  genbridge-1.7-1.7  Qwen3 encoder 1.7B + decoder 1.7B.
  genbridge-4-4      Qwen3 encoder 4B + decoder 4B.
  eval        Evaluate existing GenBridge best.pt and last.pt only.
  t5gemma-4b  Full-finetune and evaluate google/t5gemma-2-4b-4b.
  t5gemma-cnndm-1b
              Full-finetune/evaluate T5Gemma 1B-1B on CNN/DM at 4096 tokens.
  t5gemma-cnndm-4b
              Full-finetune/evaluate T5Gemma 4B-4B on CNN/DM at 4096 tokens.
  t5gemma-cnndm-all
              Copy/prepare CNN/DM once, then run both T5Gemma scales.
  llm2seq-v2-0.6-0.6
              Full-train encoder 0.6B + decoder 0.6B, then evaluate last.pt.
  llm2seq-v2-smoke-hiroute
              Train 100 seen examples to check the new architecture end to end.
  llm2seq-v2-hiroute
              Full-train the 8-layer hierarchical depth-routed 0.6B-0.6B model,
              then evaluate last.pt.
  llm2seq-v2-0.6-1.7
              Full-train encoder 0.6B + decoder 1.7B, then evaluate last.pt.
  llm2seq-v2-main
              Run both main LLM2Seq-v2 configurations sequentially.
  llm2seq-v2-ablation-all
              Run all five LLM2Seq-v2 ablations only.
  llm2seq-v2-all
              Run both main configurations, then all five ablations.
  llm2seq-v3  Main contrastive + output-routed HiRoute run, then last.pt eval.
  llm2seq-v3-smoke
              Train/evaluate 100 examples to verify the v3 flow.
  llm2seq-v3-hiroute
              Explicit alias for the main output-routed HiRoute run.
  llm2seq-v3-single-bank
              Full six-layer single-bank v3 control.
  llm2seq-v3-pilot-all
              Run matched 2k HiRoute/single-bank pilots and compare held-out results.
  llm2seq-v3-pilot-compare
              Compare two completed v3 pilot runs without retraining.
  llm2seq-v3-ablation-all
              Run all registered v3 ablations, each followed by evaluation.
  llm2seq-v4-test
              Run all V4 synthetic checks offline; never download a model.
  llm2seq-v4-smoke
              Train/evaluate the 100-example Prospective Summary Bridge flow.
  llm2seq-v4-pilot
              Run the decisive 2k/512 held-out V4 pilot.
  llm2seq-v4
              Full Qwen3-Embedding 0.6B -> Qwen3 0.6B V4, then last.pt eval.
  llm2seq-v4-pilot-ablation-all
              Run main plus four decisive matched V4 pilot ablations.
  llm2seq-v5-test
              Run all V5 offline tests without downloading a model.
  llm2seq-v5-smoke
              Smoke-test the V5 phrase-continuation WikiLingua flow.
  llm2seq-v5-pilot-ablation-all
              Run the matched V5 validation pilot suite.
  llm2seq-v5
              Full WikiLingua V5 train and last.pt test evaluation.
  llm2seq-v5-cnndm-prepare
              Copy/convert CNN/DM from CNNDM_SOURCE_DIR into V5-owned data.
  llm2seq-v5-cnndm-smoke
              Smoke-test V5 on CNN/DM at 4096 source tokens.
  llm2seq-v5-cnndm
              Train V5 on CNN/DM for 1 warm-up + 5 full epochs, then test.
  compare     Compare both GenBridge checkpoints with T5Gemma.
  compare-4b  Compare both GenBridge checkpoints with T5Gemma 4B-4B.
  all         Run the WikiLingua LLM2Seq-v2/T5Gemma sequence and all five
              ablations. CNN/DM runs independently via run_cnndm_t5gemma.sh.
              Every train is followed by full-test evaluation.
  legacy-all  Preserve the previous GenBridge/ablation experiment sequence.
  ablation-pilot     Fast architecture gate; reuse completed full-model control.
  ablation-main      Main paper ablations; reuse completed full-model control.
  ablation-analysis  Adapter-depth/RoPE/fusion analysis configurations.
  ablation-all       Every unfinished unique ablation (14 train+eval runs).
  params      Print exactly where batch size, epochs, LR, etc. are configured.

Before T5Gemma 4B/all:
  cp T5Gemma/env.example.txt T5Gemma/env.txt
  # Edit HF_TOKEN and PYTHON_BIN in T5Gemma/env.txt.

CNN/DailyMail source folder:
  CNNDM_SOURCE_DIR=/absolute/path/to/cnndm bash run.sh t5gemma-cnndm-all
  CNNDM_SOURCE_DIR=/absolute/path/to/cnndm OVERWRITE_LLM2SEQ_V5=true \
    bash run.sh llm2seq-v5-cnndm
  CNNDM_SOURCE_DIR must contain train.txt, val.txt, and test.txt.
  Dedicated other-GPU queue:
  bash run_cnndm_t5gemma.sh /absolute/path/to/cnndm 1

Intentional GenBridge rerun:
  OVERWRITE_GENBRIDGE=true bash run.sh genbridge-1.7-0.6

Explicit GenBridge configuration:
  GENBRIDGE_CONFIG=configs/qwen3_enc1_7b_dec0_6b.yaml bash run.sh genbridge
  GENBRIDGE_CONFIG=configs/qwen3_enc1_7b_dec0_6b.yaml bash run.sh eval

T5Gemma overwrite is controlled by OVERWRITE_OUTPUT_DIR in T5Gemma/env.txt.
LLM2Seq-v2 overwrite:
  OVERWRITE_LLM2SEQ_V2=true bash run.sh all
LLM2Seq-v3 overwrite:
  OVERWRITE_LLM2SEQ_V3=true bash run.sh llm2seq-v3
LLM2Seq-v4 overwrite:
  OVERWRITE_LLM2SEQ_V4=true bash run.sh llm2seq-v4-pilot
LLM2Seq-v5 overwrite:
  OVERWRITE_LLM2SEQ_V5=true bash run.sh llm2seq-v5-cnndm

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
  genbridge-1.7-0.6)
    run_genbridge configs/qwen3_enc1_7b_dec0_6b.yaml
    ;;
  genbridge-0.6-1.7)
    run_genbridge configs/qwen3_enc0_6b_dec1_7b.yaml
    ;;
  genbridge-1.7-1.7)
    run_genbridge configs/qwen3_1_7b.yaml
    ;;
  genbridge-4-4)
    run_genbridge configs/qwen3_4b.yaml
    ;;
  eval)
    eval_genbridge
    ;;
  t5gemma-4b)
    run_t5gemma_4b
    ;;
  t5gemma-cnndm-1b)
    run_t5gemma_cnndm 1b
    ;;
  t5gemma-cnndm-4b)
    run_t5gemma_cnndm 4b
    ;;
  t5gemma-cnndm-all)
    run_t5gemma_cnndm all
    ;;
  llm2seq-v2-0.6-0.6)
    run_llm2seq_v2 pipeline
    ;;
  llm2seq-v2-smoke-hiroute)
    run_llm2seq_v2 smoke-hiroute
    ;;
  llm2seq-v2-hiroute)
    run_llm2seq_v2 hiroute
    ;;
  llm2seq-v2-0.6-1.7)
    run_llm2seq_v2 decoder-1.7b
    ;;
  llm2seq-v2-main)
    run_llm2seq_v2_main
    ;;
  llm2seq-v2-ablation-all)
    run_llm2seq_v2 ablation-all
    ;;
  llm2seq-v2-all)
    run_llm2seq_v2_all
    ;;
  llm2seq-v3)
    run_llm2seq_v3 pipeline
    ;;
  llm2seq-v3-smoke)
    run_llm2seq_v3 smoke
    ;;
  llm2seq-v3-hiroute)
    run_llm2seq_v3 hiroute
    ;;
  llm2seq-v3-single-bank)
    run_llm2seq_v3 single-bank
    ;;
  llm2seq-v3-pilot-all)
    run_llm2seq_v3 pilot-all
    ;;
  llm2seq-v3-pilot-compare)
    run_llm2seq_v3 pilot-compare
    ;;
  llm2seq-v3-ablation-all)
    run_llm2seq_v3 ablation-all
    ;;
  llm2seq-v4-test)
    run_llm2seq_v4 test
    ;;
  llm2seq-v4-smoke)
    run_llm2seq_v4 smoke
    ;;
  llm2seq-v4-pilot)
    run_llm2seq_v4 pilot
    ;;
  llm2seq-v4)
    run_llm2seq_v4 qwen
    ;;
  llm2seq-v4-pilot-ablation-all)
    run_llm2seq_v4 pilot-ablation-all
    ;;
  llm2seq-v5-test)
    run_llm2seq_v5 test
    ;;
  llm2seq-v5-smoke)
    run_llm2seq_v5 smoke
    ;;
  llm2seq-v5-pilot-ablation-all)
    run_llm2seq_v5 pilot-ablation-all
    ;;
  llm2seq-v5)
    run_llm2seq_v5 full
    ;;
  llm2seq-v5-cnndm-prepare)
    if [[ -z "${CNNDM_SOURCE_DIR:-}" ]]; then
      echo "Set CNNDM_SOURCE_DIR=/absolute/path/to/cnndm." >&2
      exit 2
    fi
    bash llm2seq_v5/run.sh cnndm-prepare "$CNNDM_SOURCE_DIR"
    ;;
  llm2seq-v5-cnndm-smoke)
    run_llm2seq_v5 cnndm-smoke
    ;;
  llm2seq-v5-cnndm)
    run_llm2seq_v5 cnndm
    ;;
  compare)
    compare_models
    ;;
  compare-4b)
    compare_models_4b
    ;;
  ablation-pilot)
    run_ablation_group pilot genbridge
    ;;
  ablation-main)
    run_ablation_group main genbridge
    ;;
  ablation-analysis)
    run_ablation_group analysis
    ;;
  ablation-all)
    run_ablation_group all genbridge
    ;;
  params)
    show_parameter_locations
    ;;
  all)
    run_paper_sequence
    ;;
  legacy-all)
    run_legacy_remaining
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
