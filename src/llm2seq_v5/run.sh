#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  fi
fi
if [[ -z "${PYTHON_BIN:-}" || ! -x "$PYTHON_BIN" ]]; then
  echo "No executable Python found. Activate the intended environment or set PYTHON_BIN." >&2
  exit 2
fi
export PYTHON_BIN
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-help}"
[[ $# -gt 0 ]] && shift

if [[ "$MODE" == "test" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

MAIN_CONFIG="${CONFIG:-configs/qwen3_embedding_0_6b_phrase_continuation.yaml}"
SMOKE_CONFIG="configs/smoke_phrase_continuation_100.yaml"
PILOT_CONFIG="configs/pilot_phrase_continuation_2000.yaml"
CNNDM_CONFIG="configs/cnndm_qwen3_embedding_0_6b_phrase_continuation_4096.yaml"
CNNDM_SMOKE_CONFIG="configs/smoke_cnndm_100.yaml"
PUBMED_CONFIG="configs/pubmed_qwen3_embedding_0_6b_phrase_continuation_4096.yaml"
PUBMED_SMOKE_CONFIG="configs/smoke_pubmed_100.yaml"
ROUGE155_SCRIPT="../rouge155/evaluate_rouge.py"

prepare_cnndm() {
  local source_dir="${1:-${CNNDM_SOURCE_DIR:-}}"
  if [[ -z "$source_dir" ]]; then
    echo "CNN/DailyMail source is required." >&2
    echo "Set CNNDM_SOURCE_DIR=/absolute/path or run: bash run.sh cnndm-prepare /absolute/path" >&2
    exit 2
  fi
  "$PYTHON_BIN" -m llm2seq_v5.prepare_cnndm \
    --input-dir "$source_dir" \
    --raw-copy-dir data/raw/cnndm \
    --output-dir data/cnndm
}

ensure_cnndm_data() {
  local split
  for split in train validation test; do
    if [[ ! -s "data/cnndm/${split}.jsonl" ]]; then
      echo "Missing data/cnndm/${split}.jsonl; run cnndm-prepare first." >&2
      exit 2
    fi
  done
}

prepare_cnndm_if_requested() {
  if [[ -n "${CNNDM_SOURCE_DIR:-}" ]]; then
    prepare_cnndm "$CNNDM_SOURCE_DIR"
  fi
  ensure_cnndm_data
}

prepare_pubmed() {
  local source_dir="${1:-${PUBMED_SOURCE_DIR:-}}"
  if [[ -z "$source_dir" ]]; then
    echo "PubMed source is required." >&2
    echo "Set PUBMED_SOURCE_DIR=/absolute/path/to/pubmet or run: bash run.sh pubmed-prepare /absolute/path" >&2
    exit 2
  fi
  "$PYTHON_BIN" -m llm2seq_v5.prepare_pubmed \
    --input-dir "$source_dir" \
    --raw-copy-dir data/raw/pubmed \
    --output-dir data/pubmed
}

ensure_pubmed_data() {
  local split
  for split in train validation test; do
    if [[ ! -s "data/pubmed/${split}.jsonl" ]]; then
      echo "Missing data/pubmed/${split}.jsonl; run pubmed-prepare first." >&2
      exit 2
    fi
  done
}

prepare_pubmed_if_requested() {
  if [[ -n "${PUBMED_SOURCE_DIR:-}" ]]; then
    prepare_pubmed "$PUBMED_SOURCE_DIR"
  fi
  ensure_pubmed_data
}

output_dir_for() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from llm2seq_v5.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
}

run_rouge155() {
  local predictions="$1"
  if [[ -z "${PYROUGE_HOME_DIR:-}" ]]; then
    echo "Perl ROUGE skipped: export PYROUGE_HOME_DIR and run 'bash run.sh rouge155'." >&2
    return 0
  fi
  "$PYTHON_BIN" "$ROUGE155_SCRIPT" "$predictions" --output "${predictions%.*}.rouge155.json"
}

run_pipeline() {
  local config="$1"
  local split="$2"
  shift 2
  local output_dir
  output_dir="$(output_dir_for "$config")"
  "$PYTHON_BIN" -m llm2seq_v5.training --config "$config" "$@"
  "$PYTHON_BIN" -m llm2seq_v5.evaluate \
    --config "$output_dir/resolved_config.yaml" \
    --checkpoint "$output_dir/last.pt" \
    --split "$split" \
    --output "$output_dir/last_${split}_predictions.jsonl"
  if [[ "$split" == "test" ]]; then
    run_rouge155 "$output_dir/last_test_predictions.jsonl"
  fi
}

evaluate_existing() {
  local config="$1"
  local split="$2"
  shift 2
  local output_dir
  output_dir="$(output_dir_for "$config")"
  "$PYTHON_BIN" -m llm2seq_v5.evaluate \
    --config "$output_dir/resolved_config.yaml" \
    --checkpoint "$output_dir/last.pt" \
    --split "$split" \
    --output "$output_dir/last_${split}_predictions.jsonl" "$@"
}

compare_pilots() {
  local baseline_config="$1"
  local label="$2"
  local candidate_dir baseline_dir
  candidate_dir="$(output_dir_for "$PILOT_CONFIG")"
  baseline_dir="$(output_dir_for "$baseline_config")"
  "$PYTHON_BIN" -m llm2seq_v5.paired_compare \
    --candidate-predictions "$candidate_dir/last_validation_predictions.jsonl" \
    --candidate-metrics "$candidate_dir/last_validation_predictions.metrics.json" \
    --candidate-config "$candidate_dir/resolved_config.yaml" \
    --baseline-predictions "$baseline_dir/last_validation_predictions.jsonl" \
    --baseline-metrics "$baseline_dir/last_validation_predictions.metrics.json" \
    --baseline-config "$baseline_dir/resolved_config.yaml" \
    --output "$candidate_dir/paired_vs_${label}.json"
}

case "$MODE" in
  setup)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q tests
    ;;
  check-model)
    "$PYTHON_BIN" -m llm2seq_v5.architecture_check --config "$MAIN_CONFIG" "$@"
    ;;
  count-params)
    "$PYTHON_BIN" -m llm2seq_v5.count_parameters --config "$MAIN_CONFIG" "$@"
    ;;
  smoke)
    run_pipeline "$SMOKE_CONFIG" validation "$@"
    SMOKE_DIR="$(output_dir_for "$SMOKE_CONFIG")"
    "$PYTHON_BIN" -m llm2seq_v5.smoke_gate --run-dir "$SMOKE_DIR" --expected-examples 20
    ;;
  pilot)
    run_pipeline "$PILOT_CONFIG" validation "$@"
    ;;
  cnndm-prepare)
    prepare_cnndm "${1:-}"
    ;;
  cnndm-smoke)
    prepare_cnndm_if_requested
    run_pipeline "$CNNDM_SMOKE_CONFIG" validation "$@"
    CNNDM_SMOKE_DIR="$(output_dir_for "$CNNDM_SMOKE_CONFIG")"
    "$PYTHON_BIN" -m llm2seq_v5.smoke_gate --run-dir "$CNNDM_SMOKE_DIR" --expected-examples 20
    ;;
  cnndm|cnndm-full)
    prepare_cnndm_if_requested
    run_pipeline "$CNNDM_CONFIG" test "$@"
    ;;
  cnndm-train)
    prepare_cnndm_if_requested
    "$PYTHON_BIN" -m llm2seq_v5.training --config "$CNNDM_CONFIG" "$@"
    ;;
  cnndm-eval)
    ensure_cnndm_data
    evaluate_existing "$CNNDM_CONFIG" test "$@"
    ;;
  pubmed-prepare)
    prepare_pubmed "${1:-}"
    ;;
  pubmed-smoke)
    prepare_pubmed_if_requested
    run_pipeline "$PUBMED_SMOKE_CONFIG" validation "$@"
    PUBMED_SMOKE_DIR="$(output_dir_for "$PUBMED_SMOKE_CONFIG")"
    "$PYTHON_BIN" -m llm2seq_v5.smoke_gate --run-dir "$PUBMED_SMOKE_DIR" --expected-examples 20
    ;;
  pubmed|pubmed-full)
    prepare_pubmed_if_requested
    run_pipeline "$PUBMED_CONFIG" test "$@"
    ;;
  pubmed-train)
    prepare_pubmed_if_requested
    "$PYTHON_BIN" -m llm2seq_v5.training --config "$PUBMED_CONFIG" "$@"
    ;;
  pubmed-eval)
    ensure_pubmed_data
    evaluate_existing "$PUBMED_CONFIG" test "$@"
    PUBMED_OUTPUT_DIR="$(output_dir_for "$PUBMED_CONFIG")"
    run_rouge155 "$PUBMED_OUTPUT_DIR/last_test_predictions.jsonl"
    ;;
  full|pipeline)
    run_pipeline "$MAIN_CONFIG" test "$@"
    ;;
  train)
    "$PYTHON_BIN" -m llm2seq_v5.training --config "$MAIN_CONFIG" "$@"
    ;;
  eval-validation)
    evaluate_existing "$MAIN_CONFIG" validation "$@"
    ;;
  eval-test|eval)
    evaluate_existing "$MAIN_CONFIG" test "$@"
    ;;
  rouge155)
    MAIN_DIR="$(output_dir_for "$MAIN_CONFIG")"
    run_rouge155 "${1:-$MAIN_DIR/last_test_predictions.jsonl}"
    ;;
  paired-compare)
    "$PYTHON_BIN" -m llm2seq_v5.paired_compare "$@"
    ;;
  final-audit)
    "$PYTHON_BIN" -m llm2seq_v5.final_audit "$@"
    ;;
  ablation-v4-psb)
    run_pipeline configs/ablations/v4_psb_control.yaml test "$@"
    ;;
  ablation-no-phrase-prior)
    run_pipeline configs/ablations/no_phrase_prior.yaml test "$@"
    ;;
  ablation-no-continuation)
    run_pipeline configs/ablations/no_continuation.yaml test "$@"
    ;;
  ablation-no-coverage)
    run_pipeline configs/ablations/no_coverage.yaml test "$@"
    ;;
  pilot-ablation-all)
    run_pipeline "$PILOT_CONFIG" validation "$@"
    run_pipeline configs/pilot_ablations/v2_core_single_bank_2000.yaml validation "$@"
    run_pipeline configs/pilot_ablations/v4_psb_control_2000.yaml validation "$@"
    run_pipeline configs/pilot_ablations/no_continuation_2000.yaml validation "$@"
    run_pipeline configs/pilot_ablations/no_phrase_prior_2000.yaml validation "$@"
    run_pipeline configs/pilot_ablations/no_coverage_2000.yaml validation "$@"
    compare_pilots configs/pilot_ablations/v2_core_single_bank_2000.yaml v2_core_single_bank
    compare_pilots configs/pilot_ablations/v4_psb_control_2000.yaml v4_psb
    compare_pilots configs/pilot_ablations/no_continuation_2000.yaml no_continuation
    compare_pilots configs/pilot_ablations/no_phrase_prior_2000.yaml no_phrase_prior
    compare_pilots configs/pilot_ablations/no_coverage_2000.yaml no_coverage
    ;;
  ablation-all)
    run_pipeline configs/ablations/v4_psb_control.yaml test "$@"
    run_pipeline configs/ablations/no_continuation.yaml test "$@"
    run_pipeline configs/ablations/no_phrase_prior.yaml test "$@"
    run_pipeline configs/ablations/no_coverage.yaml test "$@"
    ;;
  *)
    cat <<'EOF'
Usage: bash run.sh MODE [arguments]

Recommended order:
  test
  check-model
  smoke --overwrite-output-dir
  pilot-ablation-all --overwrite-output-dir   # V5 + V2-core/V4/component controls; validation only
  full --overwrite-output-dir                 # full locked test, after freeze

CNN/DailyMail (4096 source tokens, 1 warm-up + 5 full = 6 epochs):
  bash run.sh cnndm-prepare /absolute/path/to/cnndm
  bash run.sh cnndm-smoke --overwrite-output-dir
  bash run.sh cnndm --overwrite-output-dir
  # Or prepare and run in one command:
  CNNDM_SOURCE_DIR=/absolute/path/to/cnndm bash run.sh cnndm --overwrite-output-dir

PubMed (4096 source tokens, 1 warm-up + 5 full = 6 epochs):
  bash run.sh pubmed-prepare /absolute/path/to/pubmet
  bash run.sh pubmed-smoke --overwrite-output-dir
  bash run.sh pubmed --overwrite-output-dir
  PUBMED_SOURCE_DIR=/absolute/path/to/pubmet bash run.sh pubmed --overwrite-output-dir

Core modes:
  test | setup | check-model | count-params
  smoke | pilot | pilot-ablation-all
  train | full | eval-validation | eval-test | rouge155
  cnndm-prepare | cnndm-smoke | cnndm | cnndm-train | cnndm-eval
  pubmed-prepare | pubmed-smoke | pubmed | pubmed-train | pubmed-eval
  paired-compare | final-audit

Full paper ablations (run only after architecture selection):
  ablation-v4-psb
  ablation-no-continuation
  ablation-no-phrase-prior
  ablation-no-coverage
  ablation-all

Override the main config with CONFIG=configs/...yaml. Only last.pt is saved.
There is no Hugging Face upload or push command.
EOF
    ;;
esac
