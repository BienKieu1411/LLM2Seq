#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEFAULT_PYTHON="/Users/kieugiangbien/bienkieu_env/bin/python"
if [[ -x "$DEFAULT_PYTHON" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

# Unit tests are deliberately synthetic. Make accidental Hub access fail
# immediately instead of silently downloading a checkpoint to the workstation.
if [[ "$MODE" == "test" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

MAIN_CONFIG="${CONFIG:-configs/qwen3_0_6b_hiroute.yaml}"
SMOKE_CONFIG="configs/smoke_hiroute_100.yaml"
SINGLE_BANK_CONFIG="configs/qwen3_0_6b.yaml"
SINGLE_BANK_SMOKE_CONFIG="configs/smoke_100.yaml"
ROUGE155_SCRIPT="../rouge155/evaluate_rouge.py"

run_rouge155() {
  local predictions="$1"
  local config="${2:-$MAIN_CONFIG}"
  if [[ -z "${PYROUGE_HOME_DIR:-}" ]]; then
    echo "Skipping Perl ROUGE-1.5.5: export PYROUGE_HOME_DIR first." >&2
    echo "Then run: bash run.sh rouge155 ${predictions}" >&2
    return 0
  fi
  local scores_file="${predictions%.*}.rouge155.json"
  "$PYTHON_BIN" "$ROUGE155_SCRIPT" "$predictions" --output "$scores_file"
  "$PYTHON_BIN" -m llm2seq_v3.paper_compare \
    --config "$config" \
    --scores "$scores_file" \
    --candidate-metrics "${predictions%.*}.metrics.json" \
    --output "$(dirname "$predictions")/t5gemma_paper_gap_report.json"
}

run_final_audit() {
  local baseline_scores="${1:?baseline Perl ROUGE JSON is required}"
  local baseline_metrics="${2:?baseline diagnostic metrics JSON is required}"
  local candidate_dir="${3:-}"
  if [[ -z "$candidate_dir" ]]; then
    candidate_dir="$("$PYTHON_BIN" - "$MAIN_CONFIG" <<'PY'
import sys
from llm2seq_v3.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
  fi
  local audit_config="$MAIN_CONFIG"
  if [[ -f "$candidate_dir/resolved_config.yaml" ]]; then
    audit_config="$candidate_dir/resolved_config.yaml"
  fi
  "$PYTHON_BIN" -m llm2seq_v3.final_audit \
    --config "$audit_config" \
    --candidate-scores "$candidate_dir/last_test_predictions.rouge155.json" \
    --candidate-metrics "$candidate_dir/last_test_predictions.metrics.json" \
    --baseline-scores "$baseline_scores" \
    --baseline-metrics "$baseline_metrics" \
    --output "$candidate_dir/final_t5gemma2_1b_1b_audit.json"
}

run_pipeline() {
  local config="$1"
  shift
  local output_dir
  output_dir="$("$PYTHON_BIN" - "$config" <<'PY'
import sys
from llm2seq_v3.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
  "$PYTHON_BIN" -m llm2seq_v3.training --config "$config" "$@"
  "$PYTHON_BIN" -m llm2seq_v3.evaluate \
    --config "$output_dir/resolved_config.yaml" \
    --checkpoint "$output_dir/last.pt" \
    --output "$output_dir/last_test_predictions.jsonl"
  run_rouge155 "$output_dir/last_test_predictions.jsonl" "$output_dir/resolved_config.yaml"
}

run_smoke() {
  local config="$1"
  shift
  local output_dir
  output_dir="$("$PYTHON_BIN" - "$config" <<'PY'
import sys
from llm2seq_v3.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
  run_pipeline "$config" "$@"
  "$PYTHON_BIN" -m llm2seq_v3.smoke_gate --run-dir "$output_dir" --expected-examples 20
}

case "$MODE" in
  setup)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q tests
    ;;
  check-model)
    "$PYTHON_BIN" -m llm2seq_v3.architecture_check "$@"
    ;;
  count-params)
    "$PYTHON_BIN" -m llm2seq_v3.count_parameters "$@"
    ;;
  train)
    "$PYTHON_BIN" -m llm2seq_v3.training --config "$MAIN_CONFIG" "$@"
    ;;
  eval)
    OUTPUT_DIR="$("$PYTHON_BIN" - "$MAIN_CONFIG" <<'PY'
import sys
from llm2seq_v3.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
    "$PYTHON_BIN" -m llm2seq_v3.evaluate \
      --config "$OUTPUT_DIR/resolved_config.yaml" \
      --checkpoint "$OUTPUT_DIR/last.pt" \
      --output "$OUTPUT_DIR/last_test_predictions.jsonl" \
      "$@"
    ;;
  rouge155)
    PREDICTIONS="${1:-}"
    PAPER_CONFIG="${2:-$MAIN_CONFIG}"
    if [[ -z "$PREDICTIONS" ]]; then
      OUTPUT_DIR="$("$PYTHON_BIN" - "$MAIN_CONFIG" <<'PY'
import sys
from llm2seq_v3.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
      PREDICTIONS="$OUTPUT_DIR/last_test_predictions.jsonl"
      if [[ -f "$OUTPUT_DIR/resolved_config.yaml" ]]; then
        PAPER_CONFIG="$OUTPUT_DIR/resolved_config.yaml"
      fi
    fi
    run_rouge155 "$PREDICTIONS" "$PAPER_CONFIG"
    ;;
  final-audit)
    if [[ $# -lt 2 ]]; then
      echo "Usage: bash run.sh final-audit BASELINE_ROUGE155_JSON BASELINE_METRICS_JSON [CANDIDATE_RUN_DIR]" >&2
      exit 2
    fi
    run_final_audit "$1" "$2" "${3:-}"
    ;;
  pipeline)
    run_pipeline "$MAIN_CONFIG" "$@"
    ;;
  smoke)
    run_smoke "$SMOKE_CONFIG" "$@"
    ;;
  hiroute)
    run_pipeline configs/qwen3_0_6b_hiroute.yaml "$@"
    ;;
  smoke-hiroute)
    run_smoke configs/smoke_hiroute_100.yaml "$@"
    ;;
  single-bank)
    run_pipeline "$SINGLE_BANK_CONFIG" "$@"
    ;;
  smoke-single-bank)
    run_smoke "$SINGLE_BANK_SMOKE_CONFIG" "$@"
    ;;
  pilot)
    run_pipeline configs/pilot_hiroute_2000.yaml "$@"
    ;;
  pilot-single-bank)
    run_pipeline configs/pilot_single_bank_2000.yaml "$@"
    ;;
  pilot-compare)
    "$PYTHON_BIN" -m llm2seq_v3.pilot_compare \
      --main-dir runs/llm2seq_v3/pilot_hiroute_2000 \
      --control-dir runs/llm2seq_v3/pilot_single_bank_2000 \
      --output runs/llm2seq_v3/pilot_comparison.json
    ;;
  pilot-all)
    run_pipeline configs/pilot_hiroute_2000.yaml "$@"
    run_pipeline configs/pilot_single_bank_2000.yaml "$@"
    "$PYTHON_BIN" -m llm2seq_v3.pilot_compare \
      --main-dir runs/llm2seq_v3/pilot_hiroute_2000 \
      --control-dir runs/llm2seq_v3/pilot_single_bank_2000 \
      --output runs/llm2seq_v3/pilot_comparison.json
    ;;
  ablation-no-contrastive)
    run_pipeline configs/ablations/no_contrastive.yaml "$@"
    ;;
  ablation-no-prompt-alignment)
    run_pipeline configs/ablations/no_prompt_alignment.yaml "$@"
    ;;
  ablation-no-source-swap)
    run_pipeline configs/ablations/no_source_swap.yaml "$@"
    ;;
  ablation-cyclic-source-swap)
    run_pipeline configs/ablations/cyclic_source_swap.yaml "$@"
    ;;
  ablation-cross-gate-0.1)
    run_pipeline configs/ablations/cross_gate_0_1.yaml "$@"
    ;;
  ablation-adapter-4layers)
    run_pipeline configs/ablations/adapter_4layers.yaml "$@"
    ;;
  ablation-v2-control)
    run_pipeline configs/ablations/v2_exact_control.yaml "$@"
    ;;
  ablation-pre-attention-routing)
    run_pipeline configs/ablations/pre_attention_memory_routing.yaml "$@"
    ;;
  ablation-static-output-routing)
    run_pipeline configs/ablations/static_output_routing.yaml "$@"
    ;;
  ablation-no-routing-balance)
    run_pipeline configs/ablations/no_routing_balance.yaml "$@"
    ;;
  ablation-no-branch-context)
    run_pipeline configs/ablations/no_branch_global_context.yaml "$@"
    ;;
  ablation-no-hiroute)
    run_pipeline configs/ablations/no_hiroute_memory.yaml "$@"
    ;;
  ablation-mean-only-pooling)
    run_pipeline configs/ablations/mean_only_contrastive_pooling.yaml "$@"
    ;;
  ablation-no-label-smoothing)
    run_pipeline configs/ablations/no_label_smoothing.yaml "$@"
    ;;
  ablation-no-fusion)
    run_pipeline configs/ablations/no_layer_fusion.yaml "$@"
    ;;
  ablation-no-bidirectional)
    run_pipeline configs/ablations/no_bidirectional_adapter.yaml "$@"
    ;;
  ablation-random-cross)
    run_pipeline configs/ablations/random_cross_attention.yaml "$@"
    ;;
  ablation-no-salience)
    run_pipeline configs/ablations/no_salience.yaml "$@"
    ;;
  ablation-sparse-cross)
    run_pipeline configs/ablations/sparse_cross_attention.yaml "$@"
    ;;
  ablation-all)
    run_pipeline configs/ablations/no_contrastive.yaml "$@"
    run_pipeline configs/ablations/no_prompt_alignment.yaml "$@"
    run_pipeline configs/ablations/no_source_swap.yaml "$@"
    run_pipeline configs/ablations/cyclic_source_swap.yaml "$@"
    run_pipeline configs/ablations/cross_gate_0_1.yaml "$@"
    run_pipeline configs/ablations/adapter_4layers.yaml "$@"
    run_pipeline configs/ablations/v2_exact_control.yaml "$@"
    run_pipeline configs/ablations/pre_attention_memory_routing.yaml "$@"
    run_pipeline configs/ablations/static_output_routing.yaml "$@"
    run_pipeline configs/ablations/no_routing_balance.yaml "$@"
    run_pipeline configs/ablations/no_branch_global_context.yaml "$@"
    run_pipeline configs/ablations/no_hiroute_memory.yaml "$@"
    run_pipeline configs/ablations/mean_only_contrastive_pooling.yaml "$@"
    run_pipeline configs/ablations/no_label_smoothing.yaml "$@"
    run_pipeline configs/ablations/no_layer_fusion.yaml "$@"
    run_pipeline configs/ablations/no_bidirectional_adapter.yaml "$@"
    run_pipeline configs/ablations/random_cross_attention.yaml "$@"
    run_pipeline configs/ablations/no_salience.yaml "$@"
    run_pipeline configs/ablations/sparse_cross_attention.yaml "$@"
    ;;
  *)
    cat <<'EOF'
Usage: bash run.sh MODE [arguments]

Modes:
  setup                       Install into bienkieu_env (or PYTHON_BIN).
  test                        Run unit tests without downloading models.
  check-model                 Load both real checkpoints and run a tiny forward pass.
  count-params                Count both profiles without loading model weights.
  smoke --overwrite-output-dir
                              Flow-check the main contrastive HiRoute-v3 model.
  smoke-hiroute --overwrite-output-dir
                              Explicit alias for the main HiRoute smoke run.
  smoke-single-bank --overwrite-output-dir
                              Flow-check the six-layer single-bank control.
  pilot --overwrite-output-dir
                              2k held-out generalization pilot for main HiRoute.
  pilot-single-bank --overwrite-output-dir
                              Capacity-matched 2k single-bank pilot control.
  pilot-all --overwrite-output-dir
                              Run both pilots and write pilot_comparison.json.
  pilot-compare              Compare two completed pilot runs only.
  pipeline --overwrite-output-dir
                              Main: contrastive output-routed HiRoute, then test last.pt.
  hiroute --overwrite-output-dir
                              Explicit alias for the main full run.
  single-bank --overwrite-output-dir
                              Full six-layer single-bank control run.
  train [--overwrite-output-dir]
  eval [--max-samples N]      Evaluate the existing last.pt only.
  rouge155 [predictions.jsonl] Calculate paper scores with Perl ROUGE-1.5.5.
  final-audit BASELINE_ROUGE155_JSON BASELINE_METRICS_JSON [CANDIDATE_RUN_DIR]
                              Require same split/backend, actual smaller parameter
                              count, and strictly higher R-1/R-2/R-L than T5Gemma.
  ablation-no-contrastive     Ablation: disable contrastive learning.
  ablation-no-prompt-alignment
  ablation-no-source-swap
  ablation-cyclic-source-swap
  ablation-cross-gate-0.1
  ablation-adapter-4layers
  ablation-v2-control         Controlled reproduction of the v2 recipe.
  ablation-pre-attention-routing
                              Ablation: legacy bank mixing before attention.
  ablation-static-output-routing
                              Ablation: post-attention route fixed per depth.
  ablation-no-routing-balance Ablation: remove global anti-collapse loss.
  ablation-no-branch-context  Ablation: leave lexical/semantic banks causal-only.
  ablation-no-hiroute        Ablation: remove only the three-bank memory/router.
  ablation-mean-only-pooling Ablation: remove final/EOS contrastive pooling.
  ablation-no-label-smoothing Ablation: disable label smoothing.
  ablation-no-fusion
  ablation-no-bidirectional
  ablation-random-cross
  ablation-no-salience
  ablation-sparse-cross
  ablation-all

Override a config for train/pipeline/eval:
  CONFIG=configs/qwen3_0_6b.yaml bash run.sh pipeline --overwrite-output-dir

There is no push/upload command and no best checkpoint.
EOF
    ;;
esac
