#!/usr/bin/env bash
set -euo pipefail

CALLER_CWD="$(pwd -P)"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$PROJECT_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
RUNNER="$PROJECT_ROOT/run.py"

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-$VIRTUAL_ENV/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
PYTHON_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
if [[ -z "${TORCHRUN_BIN:-}" && -x "$PYTHON_DIR/torchrun" ]]; then
  TORCHRUN_BIN="$PYTHON_DIR/torchrun"
else
  TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
fi
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

WIKI_CONFIG="$PROJECT_ROOT/configs/tasks/wikilingua.yaml"
CNN_CONFIG="$PROJECT_ROOT/configs/tasks/cnndm.yaml"
PUBMED_CONFIG="$PROJECT_ROOT/configs/tasks/pubmed.yaml"
ARXIV_CONFIG="$PROJECT_ROOT/configs/tasks/arxiv.yaml"
PPLX_PUBMED_CONFIG="$PROJECT_ROOT/configs/models/pplx_pubmed.yaml"
PCEB_PUBMED_CONFIG="$PROJECT_ROOT/configs/models/pplx_pubmed_pceb.yaml"
PCEB_PUBMED_CORRECTED_CONFIG="$PROJECT_ROOT/configs/models/pplx_pubmed_pceb_corrected.yaml"
DUALBRIDGE_PUBMED_CONFIG="$PROJECT_ROOT/configs/models/pplx_pubmed_dualbridge.yaml"
SMOKE_CONFIG="$PROJECT_ROOT/configs/tasks/smoke.yaml"

absolute_path() {
  if [[ "$1" = /* ]]; then
    printf '%s\n' "$1"
  else
    printf '%s\n' "$CALLER_CWD/$1"
  fi
}

config_value() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
from core.config import load_config
value = load_config(sys.argv[1])
for key in sys.argv[2].split('.'):
    value = value[key]
print(value)
PY
}

select_committed_checkpoint() {
  local output_dir="$1"
  local split="$2"
  local required
  for required in last.pt resolved_config.yaml; do
    if [[ ! -f "$output_dir/$required" ]]; then
      echo "ERROR: run is missing $output_dir/$required" >&2
      return 1
    fi
  done

  SELECTED_CHECKPOINT="$output_dir/last.pt"
  SELECTED_RESOLVED="$output_dir/resolved_config.yaml"
  SELECTED_PREDICTIONS="$output_dir/last_${split}_predictions.jsonl"
}

resolved_output_dir() {
  # Training accepts ``--output-dir``.  The post-training validation must
  # inspect that same directory rather than the YAML default; otherwise a
  # successful override can silently evaluate a stale experiment.
  local configured="$1"
  shift
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --output-dir)
        [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires a value" >&2; return 2; }
        absolute_path "$2"
        return 0
        ;;
      --output-dir=*)
        absolute_path "${1#--output-dir=}"
        return 0
        ;;
    esac
    shift
  done
  printf '%s\n' "$configured"
}

train_and_validate() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(config_value "$config" experiment.output_dir)"
  output_dir="$(resolved_output_dir "$output_dir" "$@")"
  "$PYTHON_BIN" "$RUNNER" train --config "$config" "$@"
  select_committed_checkpoint "$output_dir" validation
  "$PYTHON_BIN" "$RUNNER" evaluate \
    --config "$SELECTED_RESOLVED" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --output "$SELECTED_PREDICTIONS" \
    --split validation
}

train_and_validate_ddp() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(config_value "$config" experiment.output_dir)"
  output_dir="$(resolved_output_dir "$output_dir" "$@")"
  local processes="${DDP_NPROC_PER_NODE:-2}"
  if [[ "$processes" -lt 2 ]]; then
    echo "ERROR: DDP_NPROC_PER_NODE must be at least 2" >&2
    return 2
  fi
  "$TORCHRUN_BIN" --standalone --nproc_per_node="$processes" "$RUNNER" train --config "$config" "$@"
  select_committed_checkpoint "$output_dir" validation
  # Generation is deliberately single-process: the complete last.pt is
  # written by DDP rank 0 and greedy outputs remain deterministic.
  "$PYTHON_BIN" "$RUNNER" evaluate \
    --config "$SELECTED_RESOLVED" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --output "$SELECTED_PREDICTIONS" \
    --split validation
}

evaluate_test() {
  local config="$1"
  shift
  local output_dir
  output_dir="$(config_value "$config" experiment.output_dir)"
  select_committed_checkpoint "$output_dir" test
  "$PYTHON_BIN" "$RUNNER" evaluate \
    --config "$SELECTED_RESOLVED" \
    --checkpoint "$SELECTED_CHECKPOINT" \
    --output "$SELECTED_PREDICTIONS" \
    --split test "$@"
}

run_online_kd() {
  local config="$1"
  local checkpoint="$2"
  local output_dir="$3"
  shift 3
  "$PYTHON_BIN" -m core.training.online_kd \
    --config "$config" \
    --init-checkpoint "$(absolute_path "$checkpoint")" \
    --output-dir "$output_dir" \
    "$@"
}

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi

case "$MODE" in
  train)
    CONFIG="$(absolute_path "${1:?Pass a task config YAML}")"
    shift
    "$PYTHON_BIN" "$RUNNER" train --config "$CONFIG" "$@"
    ;;
  train-ddp)
    CONFIG="$(absolute_path "${1:?Pass a task config YAML}")"
    shift
    train_and_validate_ddp "$CONFIG" "$@"
    ;;
  evaluate)
    CONFIG="$(absolute_path "${1:?Pass a resolved config YAML}")"
    CHECKPOINT="$(absolute_path "${2:?Pass a checkpoint}")"
    OUTPUT="$(absolute_path "${3:?Pass an output JSONL path}")"
    shift 3
    "$PYTHON_BIN" "$RUNNER" evaluate \
      --config "$CONFIG" --checkpoint "$CHECKPOINT" --output "$OUTPUT" "$@"
    ;;
  validate-data)
    CONFIG="$(absolute_path "${1:?Pass a task config YAML}")"
    "$PYTHON_BIN" "$RUNNER" validate-data --config "$CONFIG"
    ;;
  test)
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" -m pytest -q "$PROJECT_ROOT/tests"
    ;;
  inspect)
    CONFIG="${1:-$WIKI_CONFIG}"
    "$PYTHON_BIN" "$RUNNER" inspect --config "$(absolute_path "$CONFIG")"
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
  arxiv)
    train_and_validate "$ARXIV_CONFIG" "$@"
    ;;
  pplx-pubmed)
    train_and_validate "$PPLX_PUBMED_CONFIG" "$@"
    ;;
  pceb-pubmed)
    train_and_validate "$PCEB_PUBMED_CONFIG" "$@"
    ;;
  pceb-pubmed-corrected)
    train_and_validate "$PCEB_PUBMED_CORRECTED_CONFIG" "$@"
    ;;
  dualbridge-pubmed)
    train_and_validate "$DUALBRIDGE_PUBMED_CONFIG" "$@"
    ;;
  pceb-pubmed-ddp)
    train_and_validate_ddp "$PCEB_PUBMED_CONFIG" "$@"
    ;;
  c3)
    train_and_validate "$WIKI_CONFIG" "$@"
    ;;
  pplx)
    train_and_validate "$PROJECT_ROOT/configs/models/pplx_wikilingua.yaml" "$@"
    ;;
  nemotron)
    train_and_validate "$PROJECT_ROOT/configs/models/nemotron_wikilingua.yaml" "$@"
    ;;
  eval-validation)
    CONFIG="${CONFIG:-$WIKI_CONFIG}"
    OUTPUT_DIR="$(config_value "$CONFIG" experiment.output_dir)"
    select_committed_checkpoint "$OUTPUT_DIR" validation
    "$PYTHON_BIN" "$RUNNER" evaluate \
      --config "$SELECTED_RESOLVED" \
      --checkpoint "$SELECTED_CHECKPOINT" \
      --output "$SELECTED_PREDICTIONS" \
      --split validation "$@"
    ;;
  evaluate-wiki-test)
    evaluate_test "$WIKI_CONFIG" "$@"
    ;;
  evaluate-cnndm-test)
    evaluate_test "$CNN_CONFIG" "$@"
    ;;
  evaluate-pubmed-test)
    evaluate_test "$PUBMED_CONFIG" "$@"
    ;;
  evaluate-arxiv-test)
    evaluate_test "$ARXIV_CONFIG" "$@"
    ;;
  evaluate-pplx-pubmed-test)
    evaluate_test "$PPLX_PUBMED_CONFIG" "$@"
    ;;
  evaluate-pceb-pubmed-test)
    evaluate_test "$PCEB_PUBMED_CONFIG" "$@"
    ;;
  evaluate-pceb-pubmed-corrected-test)
    evaluate_test "$PCEB_PUBMED_CORRECTED_CONFIG" "$@"
    ;;
  evaluate-dualbridge-pubmed-test)
    evaluate_test "$DUALBRIDGE_PUBMED_CONFIG" "$@"
    ;;
  evaluate-test)
    CONFIG="${1:-$WIKI_CONFIG}"
    if [[ $# -gt 0 ]]; then shift; fi
    evaluate_test "$(absolute_path "$CONFIG")" "$@"
    ;;
  rouge155)
    PREDICTIONS="$(absolute_path "${1:?Pass a full-test predictions.jsonl file}")"
    shift
    "$PYTHON_BIN" "$WORKSPACE_ROOT/rouge155/evaluate_rouge.py" "$PREDICTIONS" "$@"
    ;;
  bootstrap)
    CANDIDATE_HEADLINE="$(absolute_path "${1:?Pass candidate .rouge155.json}")"
    BASELINE_HEADLINE="$(absolute_path "${2:?Pass baseline .rouge155.json}")"
    OUTPUT="$(absolute_path "${3:?Pass a new output JSON path}")"
    "$PYTHON_BIN" "$WORKSPACE_ROOT/rouge155/paired_bootstrap.py" \
      "$CANDIDATE_HEADLINE" "$BASELINE_HEADLINE" --output "$OUTPUT"
    ;;
  prepare-cnndm)
    SOURCE_DIR="$(absolute_path "${1:?Pass the local CNN/DM directory}")"
    "$PYTHON_BIN" -m core.data.cnndm \
      --input-dir "$SOURCE_DIR" \
      --raw-copy-dir "$PROJECT_ROOT/datasets/raw/cnndm" \
      --output-dir "$PROJECT_ROOT/datasets/cnndm"
    ;;
  prepare-pubmed)
    SOURCE_DIR="$(absolute_path "${1:?Pass the local PubMed directory}")"
    "$PYTHON_BIN" -m core.data.pubmed \
      --input-dir "$SOURCE_DIR" \
      --raw-copy-dir "$PROJECT_ROOT/datasets/raw/pubmed" \
      --output-dir "$PROJECT_ROOT/datasets/pubmed"
    ;;
  prepare-arxiv)
    SOURCE_DIR="$(absolute_path "${1:?Pass the local ArXiv directory}")"
    "$PYTHON_BIN" -m core.data.arxiv \
      --input-dir "$SOURCE_DIR" \
      --raw-copy-dir "$PROJECT_ROOT/datasets/raw/arxiv" \
      --output-dir "$PROJECT_ROOT/datasets/arxiv"
    ;;
  *)
    cat <<'EOF'
EviSeq (runs directly from source; no local package install required)

  bash eviseq_v2/scripts/run.sh train CONFIG.yaml --overwrite-output-dir
  CUDA_VISIBLE_DEVICES=0,1 bash eviseq_v2/scripts/run.sh train-ddp CONFIG.yaml --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh evaluate RESOLVED.yaml last.pt predictions.jsonl --split test
  bash eviseq_v2/scripts/run.sh validate-data CONFIG.yaml
  bash eviseq_v2/scripts/run.sh test
  bash eviseq_v2/scripts/run.sh smoke --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh wiki --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh arxiv --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh pplx --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh nemotron --overwrite-output-dir

Data and other datasets:
  bash eviseq_v2/scripts/run.sh prepare-cnndm /absolute/path/to/cnndm
  bash eviseq_v2/scripts/run.sh prepare-pubmed /absolute/path/to/pubmed
  bash eviseq_v2/scripts/run.sh prepare-arxiv /absolute/path/to/arxiv
  bash eviseq_v2/scripts/run.sh cnndm --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh pubmed --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh arxiv --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh pplx-pubmed --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh pceb-pubmed --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh pceb-pubmed-corrected --overwrite-output-dir
  bash eviseq_v2/scripts/run.sh dualbridge-pubmed --overwrite-output-dir
  CUDA_VISIBLE_DEVICES=0,1 bash eviseq_v2/scripts/run.sh \
    pceb-pubmed-ddp --overwrite-output-dir

Evaluation commands:
  bash eviseq_v2/scripts/run.sh evaluate-wiki-test
  bash eviseq_v2/scripts/run.sh evaluate-arxiv-test --batch-size 96
  bash eviseq_v2/scripts/run.sh evaluate-pceb-pubmed-test --batch-size 96
  bash eviseq_v2/scripts/run.sh evaluate-test CONFIG.yaml --batch-size 8
  # Add --resume to continue an interrupted JSONL evaluation.
  bash eviseq_v2/scripts/run.sh rouge155 runs/.../last_test_predictions.jsonl --details
  bash eviseq_v2/scripts/run.sh bootstrap CANDIDATE_ROUGE155 BASELINE_ROUGE155 OUTPUT_JSON

Override a local checkpoint path, batch, LR, or epoch in the YAML config. The
runner uses the active virtual environment (activate bienkieu_env first).
EOF
    ;;
esac
