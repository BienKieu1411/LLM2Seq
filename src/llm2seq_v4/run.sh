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
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-help}"
[[ $# -gt 0 ]] && shift

# Synthetic tests must never download a checkpoint on a workstation.
if [[ "$MODE" == "test" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

MAIN_CONFIG="${CONFIG:-configs/qwen3_embedding_0_6b_psb.yaml}"
SMOKE_CONFIG="configs/smoke_qwen3_embedding_100.yaml"
ROUGE155_SCRIPT="../rouge155/evaluate_rouge.py"

output_dir_for() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from llm2seq_v4.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
}

run_rouge155() {
  local predictions="$1"
  local config="$2"
  if [[ -z "${PYROUGE_HOME_DIR:-}" ]]; then
    echo "Perl ROUGE skipped: export PYROUGE_HOME_DIR, then run rouge155." >&2
    return 0
  fi
  local scores="${predictions%.*}.rouge155.json"
  "$PYTHON_BIN" "$ROUGE155_SCRIPT" "$predictions" --output "$scores"
  "$PYTHON_BIN" -m llm2seq_v4.paper_compare \
    --config "$config" \
    --scores "$scores" \
    --candidate-metrics "${predictions%.*}.metrics.json" \
    --output "$(dirname "$predictions")/t5gemma_paper_gap_report.json"
}

run_pipeline() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(output_dir_for "$config")"
  "$PYTHON_BIN" -m llm2seq_v4.training --config "$config" "$@"
  "$PYTHON_BIN" -m llm2seq_v4.evaluate \
    --config "$output_dir/resolved_config.yaml" \
    --checkpoint "$output_dir/last.pt" \
    --output "$output_dir/last_test_predictions.jsonl"
  run_rouge155 "$output_dir/last_test_predictions.jsonl" "$output_dir/resolved_config.yaml"
}

run_smoke() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(output_dir_for "$config")"
  run_pipeline "$config" "$@"
  "$PYTHON_BIN" -m llm2seq_v4.smoke_gate --run-dir "$output_dir" --expected-examples 20
}

case "$MODE" in
  setup)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q tests
    ;;
  check-model)
    "$PYTHON_BIN" -m llm2seq_v4.architecture_check --config "$MAIN_CONFIG" "$@"
    ;;
  count-params)
    "$PYTHON_BIN" -m llm2seq_v4.count_parameters --config "$MAIN_CONFIG" "$@"
    ;;
  train)
    "$PYTHON_BIN" -m llm2seq_v4.training --config "$MAIN_CONFIG" "$@"
    ;;
  eval)
    OUTPUT_DIR="$(output_dir_for "$MAIN_CONFIG")"
    "$PYTHON_BIN" -m llm2seq_v4.evaluate \
      --config "$OUTPUT_DIR/resolved_config.yaml" \
      --checkpoint "$OUTPUT_DIR/last.pt" \
      --output "$OUTPUT_DIR/last_test_predictions.jsonl" "$@"
    ;;
  rouge155)
    OUTPUT_DIR="$(output_dir_for "$MAIN_CONFIG")"
    PREDICTIONS="${1:-$OUTPUT_DIR/last_test_predictions.jsonl}"
    PAPER_CONFIG="${2:-$OUTPUT_DIR/resolved_config.yaml}"
    [[ -f "$PAPER_CONFIG" ]] || PAPER_CONFIG="$MAIN_CONFIG"
    run_rouge155 "$PREDICTIONS" "$PAPER_CONFIG"
    ;;
  final-audit)
    "$PYTHON_BIN" -m llm2seq_v4.final_audit "$@"
    ;;
  pipeline|qwen)
    run_pipeline "$MAIN_CONFIG" "$@"
    ;;
  qwen-base)
    run_pipeline configs/qwen3_base_0_6b_psb.yaml "$@"
    ;;
  pplx)
    run_pipeline configs/pplx_embed_v1_0_6b_psb.yaml "$@"
    ;;
  smoke)
    run_smoke "$SMOKE_CONFIG" "$@"
    ;;
  pilot)
    run_pipeline configs/pilot_qwen3_embedding_2000.yaml "$@"
    ;;
  pilot-pplx)
    run_pipeline configs/pilot_pplx_embed_2000.yaml "$@"
    ;;
  ablation-no-prefix)
    run_pipeline configs/ablations/no_summary_prefix.yaml "$@"
    ;;
  ablation-no-alignment)
    run_pipeline configs/ablations/no_ordered_alignment.yaml "$@"
    ;;
  ablation-no-plan-only)
    run_pipeline configs/ablations/no_plan_only.yaml "$@"
    ;;
  ablation-no-oracle)
    run_pipeline configs/ablations/no_oracle_evidence.yaml "$@"
    ;;
  ablation-no-salience)
    run_pipeline configs/ablations/no_salience.yaml "$@"
    ;;
  ablation-cross-gate-0.30)
    run_pipeline configs/ablations/cross_gate_0_30.yaml "$@"
    ;;
  ablation-label-smoothing-0.10)
    run_pipeline configs/ablations/label_smoothing_0_10.yaml "$@"
    ;;
  ablation-slots-8)
    run_pipeline configs/ablations/summary_slots_8.yaml "$@"
    ;;
  ablation-slots-32)
    run_pipeline configs/ablations/summary_slots_32.yaml "$@"
    ;;
  pilot-ablation-all)
    run_pipeline configs/pilot_qwen3_embedding_2000.yaml "$@"
    run_pipeline configs/pilot_ablations/no_summary_prefix_2000.yaml "$@"
    run_pipeline configs/pilot_ablations/no_ordered_alignment_2000.yaml "$@"
    run_pipeline configs/pilot_ablations/no_plan_only_2000.yaml "$@"
    run_pipeline configs/pilot_ablations/no_oracle_evidence_2000.yaml "$@"
    ;;
  ablation-all)
    run_pipeline configs/ablations/no_summary_prefix.yaml "$@"
    run_pipeline configs/ablations/no_ordered_alignment.yaml "$@"
    run_pipeline configs/ablations/no_plan_only.yaml "$@"
    run_pipeline configs/ablations/no_oracle_evidence.yaml "$@"
    ;;
  *)
    cat <<'EOF'
Usage: bash run.sh MODE [arguments]

Core:
  test                         Synthetic/offline tests; never downloads models.
  setup                        Install requirements into bienkieu_env/PYTHON_BIN.
  smoke --overwrite-output-dir 100-train/20-test flow check.
  pilot --overwrite-output-dir 2k/512 held-out decision run.
  pipeline --overwrite-output-dir
  qwen                         Full Qwen3-Embedding 0.6B -> Qwen3 0.6B PSB.
  qwen-base                    Full causal Qwen3-Base encoder control.
  pplx                         Full PPLX 0.6B encoder score candidate.
  train | eval | rouge155 | final-audit | check-model | count-params

Ablation:
  pilot-ablation-all           Main + four decisive 2k pilots.
  ablation-no-prefix           Remove the prospective-summary slots.
  ablation-no-alignment        Keep slots, remove ordered response alignment.
  ablation-no-plan-only        Remove memory-path curriculum.
  ablation-no-oracle           Use predicted salience throughout training.
  ablation-no-salience
  ablation-cross-gate-0.30
  ablation-label-smoothing-0.10
  ablation-slots-8 | ablation-slots-32 | ablation-all

Override any single core mode with:
  CONFIG=configs/...yaml bash run.sh pipeline --overwrite-output-dir

Only last.pt is saved/evaluated. There is no Hub upload or push command.
EOF
    ;;
esac
