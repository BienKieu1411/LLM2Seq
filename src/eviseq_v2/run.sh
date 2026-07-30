#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$PACKAGE_ROOT/.." && pwd)"
cd "$SRC_ROOT"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$VIRTUAL_ENV/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
export PYTHONPATH="$PACKAGE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

WIKI_CONFIG="$PACKAGE_ROOT/configs/wikilingua.yaml"
CNN_CONFIG="$PACKAGE_ROOT/configs/cnndm.yaml"
PUBMED_CONFIG="$PACKAGE_ROOT/configs/pubmed.yaml"
PPLX_PUBMED_CONFIG="$PACKAGE_ROOT/configs/encoders/pplx_0_6b_pubmed.yaml"
SMOKE_CONFIG="$PACKAGE_ROOT/configs/smoke_100.yaml"

config_value() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
from eviseq_v2.config import load_config
value = load_config(sys.argv[1])
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

select_committed_checkpoint() {
  local output_dir="$1"
  local split="$2"
  if [[ ! -f "$output_dir/COMPLETE" ]]; then
    echo "ERROR: run is not complete: $output_dir (missing COMPLETE)." >&2
    return 1
  fi

  local checkpoint_dir="$output_dir"
  if [[ -f "$output_dir/ranking/COMPLETE" ]]; then
    checkpoint_dir="$output_dir/ranking"
  fi
  local required
  for required in last.pt resolved_config.yaml data_manifest.json parameter_manifest.json; do
    if [[ ! -f "$checkpoint_dir/$required" ]]; then
      echo "ERROR: committed checkpoint is missing $checkpoint_dir/$required" >&2
      return 1
    fi
  done

  SELECTED_CHECKPOINT="$checkpoint_dir/last.pt"
  SELECTED_RESOLVED="$checkpoint_dir/resolved_config.yaml"
  SELECTED_PREDICTIONS="$checkpoint_dir/last_${split}_predictions.jsonl"
}

train_and_validate() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(config_value "$config" experiment.output_dir)"
  "$PYTHON_BIN" -m eviseq_v2.training --config "$config" "$@"
  select_committed_checkpoint "$output_dir" validation
  "$PYTHON_BIN" -m eviseq_v2.evaluate \
    --config "$SELECTED_RESOLVED" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --output "$SELECTED_PREDICTIONS" \
    --split validation
}

paper_test() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(config_value "$config" experiment.output_dir)"
  select_committed_checkpoint "$output_dir" test
  "$PYTHON_BIN" -m eviseq_v2.evaluate \
    --config "$SELECTED_RESOLVED" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --output "$SELECTED_PREDICTIONS" \
    --split test --paper-test "$@"
}

wiki_dev_table() {
  local c0_config="$PACKAGE_ROOT/configs/ablations/c0_causal.yaml"
  local c2_config="$PACKAGE_ROOT/configs/ablations/c2_dec2enc.yaml"
  local c3_no_cl_config="$PACKAGE_ROOT/configs/ablations/c3_no_contrastive.yaml"
  local c0_dir c2_dir c3_no_cl_dir c3_dir
  c0_dir="$(config_value "$c0_config" experiment.output_dir)"
  c2_dir="$(config_value "$c2_config" experiment.output_dir)"
  c3_no_cl_dir="$(config_value "$c3_no_cl_config" experiment.output_dir)"
  c3_dir="$(config_value "$WIKI_CONFIG" experiment.output_dir)"

  local directory
  for directory in "$c0_dir" "$c2_dir" "$c3_no_cl_dir" "$c3_dir"; do
    "$PYTHON_BIN" "$SRC_ROOT/rouge155/evaluate_rouge.py" \
      "$directory/last_validation_predictions.jsonl"
  done

  "$PYTHON_BIN" -m eviseq_v2.dev_table \
    --c0-config "$c0_dir/resolved_config.yaml" \
    --c0-rouge "$c0_dir/last_validation_predictions.rouge155.json" \
    --c0-metrics "$c0_dir/last_validation_predictions.metrics.json" \
    --c2-config "$c2_dir/resolved_config.yaml" \
    --c2-rouge "$c2_dir/last_validation_predictions.rouge155.json" \
    --c2-metrics "$c2_dir/last_validation_predictions.metrics.json" \
    --c3-no-cl-config "$c3_no_cl_dir/resolved_config.yaml" \
    --c3-no-cl-rouge "$c3_no_cl_dir/last_validation_predictions.rouge155.json" \
    --c3-no-cl-metrics "$c3_no_cl_dir/last_validation_predictions.metrics.json" \
    --c3-config "$c3_dir/resolved_config.yaml" \
    --c3-rouge "$c3_dir/last_validation_predictions.rouge155.json" \
    --c3-metrics "$c3_dir/last_validation_predictions.metrics.json" \
    --output "runs/eviseq_v2/wikilingua_dev_ablation_table.json"
}

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$MODE" in
  test)
    TEST_PATHS=("$PACKAGE_ROOT/tests")
    if [[ -d "$SRC_ROOT/direct_qwen/tests" ]]; then
      TEST_PATHS+=("$SRC_ROOT/direct_qwen/tests")
      export PYTHONPATH="$SRC_ROOT/direct_qwen:$SRC_ROOT/llm2seq_v2:$SRC_ROOT/eviseq:$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if [[ -d "$SRC_ROOT/rouge155/tests" ]]; then
      TEST_PATHS+=("$SRC_ROOT/rouge155/tests")
      export PYTHONPATH="$SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" -m pytest -q "${TEST_PATHS[@]}"
    ;;
  inspect)
    "$PYTHON_BIN" -m eviseq_v2.inspect_config --config "${1:-$WIKI_CONFIG}"
    ;;
  audit-data)
    "$PYTHON_BIN" -m eviseq_v2.data_integrity --config "${1:-$WIKI_CONFIG}"
    ;;
  smoke)
    train_and_validate "$SMOKE_CONFIG" "$@"
    ;;
  wiki)
    train_and_validate "$WIKI_CONFIG" "$@"
    ;;
  cnndm)
    train_and_validate "$CNN_CONFIG" "$@"
    ;;
  pubmed)
    train_and_validate "$PUBMED_CONFIG" "$@"
    ;;
  pplx-pubmed)
    train_and_validate "$PPLX_PUBMED_CONFIG" "$@"
    ;;
  c0|c2|c3-no-cl)
    case "$MODE" in
      c0) ABLATION="c0_causal" ;;
      c2) ABLATION="c2_dec2enc" ;;
      c3-no-cl) ABLATION="c3_no_contrastive" ;;
    esac
    train_and_validate "$PACKAGE_ROOT/configs/ablations/$ABLATION.yaml" "$@"
    ;;
  c1)
    train_and_validate "$PACKAGE_ROOT/configs/appendix/c1_hard_full.yaml" "$@"
    ;;
  c3)
    train_and_validate "$WIKI_CONFIG" "$@"
    ;;
  ablation-all)
    train_and_validate "$PACKAGE_ROOT/configs/ablations/c0_causal.yaml" "$@"
    train_and_validate "$PACKAGE_ROOT/configs/ablations/c2_dec2enc.yaml" "$@"
    train_and_validate "$PACKAGE_ROOT/configs/ablations/c3_no_contrastive.yaml" "$@"
    ;;
  dev-table-wiki)
    wiki_dev_table
    ;;
  pplx)
    train_and_validate "$PACKAGE_ROOT/configs/encoders/pplx_0_6b.yaml" "$@"
    ;;
  nemotron)
    train_and_validate "$PACKAGE_ROOT/configs/encoders/nemotron_1b.yaml" "$@"
    ;;
  eval-validation)
    CONFIG="${CONFIG:-$WIKI_CONFIG}"
    OUTPUT_DIR="$(config_value "$CONFIG" experiment.output_dir)"
    select_committed_checkpoint "$OUTPUT_DIR" validation
    "$PYTHON_BIN" -m eviseq_v2.evaluate \
      --config "$SELECTED_RESOLVED" \
      --checkpoint "$SELECTED_CHECKPOINT" \
      --output "$SELECTED_PREDICTIONS" \
      --split validation "$@"
    ;;
  paper-test-wiki)
    paper_test "$WIKI_CONFIG" "$@"
    ;;
  paper-test-cnndm)
    paper_test "$CNN_CONFIG" "$@"
    ;;
  paper-test-pubmed)
    paper_test "$PUBMED_CONFIG" "$@"
    ;;
  paper-test-pplx-pubmed)
    paper_test "$PPLX_PUBMED_CONFIG" "$@"
    ;;
  paper-test)
    CONFIG="${1:-$WIKI_CONFIG}"
    if [[ $# -gt 0 ]]; then shift; fi
    paper_test "$CONFIG" "$@"
    ;;
  rouge155)
    PREDICTIONS="${1:?Pass a full-test predictions.jsonl file}"
    shift
    "$PYTHON_BIN" "$SRC_ROOT/rouge155/evaluate_rouge.py" "$PREDICTIONS" "$@"
    ;;
  bootstrap)
    CANDIDATE_HEADLINE="${1:?Pass candidate .rouge155.json}"
    BASELINE_HEADLINE="${2:?Pass baseline .rouge155.json}"
    OUTPUT="${3:?Pass a new output JSON path}"
    "$PYTHON_BIN" -m rouge155.paired_bootstrap \
      "$CANDIDATE_HEADLINE" "$BASELINE_HEADLINE" --output "$OUTPUT"
    ;;
  compare-paper)
    CONFIG="${1:?Pass resolved config}"
    CANDIDATE_ROUGE="${2:?Pass candidate .rouge155.json}"
    CANDIDATE_METRICS="${3:?Pass candidate .metrics.json}"
    BASELINE_ROUGE="${4:?Pass T5Gemma .rouge155.json}"
    BASELINE_METRICS="${5:?Pass T5Gemma metrics.json}"
    OUTPUT="${6:-}"
    COMMAND=(
      "$PYTHON_BIN" -m eviseq_v2.paper_compare
      --config "$CONFIG"
      --candidate-rouge "$CANDIDATE_ROUGE"
      --candidate-metrics "$CANDIDATE_METRICS"
      --baseline-rouge "$BASELINE_ROUGE"
      --baseline-metrics "$BASELINE_METRICS"
    )
    if [[ -n "$OUTPUT" ]]; then COMMAND+=(--output "$OUTPUT"); fi
    "${COMMAND[@]}"
    ;;
  prepare-cnndm)
    SOURCE_DIR="${1:?Pass the local CNN/DM directory}"
    "$PYTHON_BIN" -m eviseq_v2.prepare_cnndm \
      --input-dir "$SOURCE_DIR" \
      --raw-copy-dir "$PACKAGE_ROOT/data/raw/cnndm" \
      --output-dir "$PACKAGE_ROOT/data/cnndm"
    ;;
  prepare-pubmed)
    SOURCE_DIR="${1:?Pass the local PubMed directory}"
    "$PYTHON_BIN" -m eviseq_v2.prepare_pubmed \
      --input-dir "$SOURCE_DIR" \
      --raw-copy-dir "$PACKAGE_ROOT/data/raw/pubmed" \
      --output-dir "$PACKAGE_ROOT/data/pubmed"
    ;;
  *)
    cat <<'EOF'
EviSeq (run from this folder or from src; no upload/push command exists)

  bash eviseq_v2/run.sh test
  bash eviseq_v2/run.sh smoke --overwrite-output-dir
  bash eviseq_v2/run.sh wiki --overwrite-output-dir
  bash eviseq_v2/run.sh pplx --overwrite-output-dir
  bash eviseq_v2/run.sh nemotron --overwrite-output-dir
  bash eviseq_v2/run.sh c0|c2|c3-no-cl --overwrite-output-dir
  bash eviseq_v2/run.sh ablation-all --overwrite-output-dir
  bash eviseq_v2/run.sh dev-table-wiki
  bash eviseq_v2/run.sh c1 --overwrite-output-dir

Data and other datasets:
  bash eviseq_v2/run.sh prepare-cnndm /absolute/path/to/cnndm
  bash eviseq_v2/run.sh prepare-pubmed /absolute/path/to/pubmed
  bash eviseq_v2/run.sh cnndm --overwrite-output-dir
  bash eviseq_v2/run.sh pubmed --overwrite-output-dir
  bash eviseq_v2/run.sh pplx-pubmed --overwrite-output-dir

Final test is intentionally separate and allowed only on the complete split:
  bash eviseq_v2/run.sh paper-test-wiki
  bash eviseq_v2/run.sh rouge155 runs/.../last_test_predictions.jsonl --details
  bash eviseq_v2/run.sh bootstrap CANDIDATE_ROUGE155 BASELINE_ROUGE155 OUTPUT_JSON
  bash eviseq_v2/run.sh compare-paper RESOLVED_CONFIG CANDIDATE_ROUGE \
    CANDIDATE_METRICS T5GEMMA_ROUGE T5GEMMA_METRICS [OUTPUT_JSON]

Override a local checkpoint path, batch, LR, or epoch in the YAML config. The
runner uses the active virtual environment (activate bienkieu_env first).
EOF
    ;;
esac
