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

MAIN_CONFIG="${CONFIG:-configs/qwen3_0_6b.yaml}"
SMOKE_CONFIG="configs/smoke_100.yaml"
HIROUTE_CONFIG="configs/qwen3_0_6b_hiroute.yaml"
HIROUTE_SMOKE_CONFIG="configs/smoke_hiroute_100.yaml"
DECODER_HEAVY_CONFIG="configs/qwen3_embedding_0_6b_decoder_1_7b.yaml"
DECODER_HEAVY_SMOKE_CONFIG="configs/smoke_decoder_1_7b_100.yaml"

run_pipeline() {
  local config="$1"
  shift
  local output_dir
  output_dir="$("$PYTHON_BIN" - "$config" <<'PY'
import sys
from llm2seq_v2.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
  "$PYTHON_BIN" -m llm2seq_v2.training --config "$config" "$@"
  "$PYTHON_BIN" -m llm2seq_v2.evaluate \
    --config "$output_dir/resolved_config.yaml" \
    --checkpoint "$output_dir/last.pt" \
    --output "$output_dir/last_test_predictions.jsonl"
}

case "$MODE" in
  setup)
    "$PYTHON_BIN" -m pip install -r requirements.txt
    ;;
  test)
    "$PYTHON_BIN" -m pytest -q tests
    ;;
  check-model)
    "$PYTHON_BIN" -m llm2seq_v2.architecture_check "$@"
    ;;
  count-params)
    "$PYTHON_BIN" -m llm2seq_v2.count_parameters "$@"
    ;;
  train)
    "$PYTHON_BIN" -m llm2seq_v2.training --config "$MAIN_CONFIG" "$@"
    ;;
  eval)
    OUTPUT_DIR="$("$PYTHON_BIN" - "$MAIN_CONFIG" <<'PY'
import sys
from llm2seq_v2.config import load_config
print(load_config(sys.argv[1])["experiment"]["output_dir"])
PY
)"
    "$PYTHON_BIN" -m llm2seq_v2.evaluate \
      --config "$OUTPUT_DIR/resolved_config.yaml" \
      --checkpoint "$OUTPUT_DIR/last.pt" \
      --output "$OUTPUT_DIR/last_test_predictions.jsonl" \
      "$@"
    ;;
  pipeline)
    run_pipeline "$MAIN_CONFIG" "$@"
    ;;
  smoke)
    run_pipeline "$SMOKE_CONFIG" "$@"
    ;;
  hiroute)
    run_pipeline "$HIROUTE_CONFIG" "$@"
    ;;
  smoke-hiroute)
    run_pipeline "$HIROUTE_SMOKE_CONFIG" "$@"
    ;;
  decoder-1.7b)
    run_pipeline "$DECODER_HEAVY_CONFIG" "$@"
    ;;
  smoke-decoder-1.7b)
    run_pipeline "$DECODER_HEAVY_SMOKE_CONFIG" "$@"
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
                              Train 100 seen examples, save/test smoke last.pt.
  smoke-hiroute --overwrite-output-dir
                              Flow-check the hierarchical routed architecture.
  pipeline --overwrite-output-dir
                              Full 3-epoch warm-up + 12-epoch full FT, then test last.pt.
  hiroute --overwrite-output-dir
                              Full 8-layer HiRoute run, then test last.pt.
  smoke-decoder-1.7b --overwrite-output-dir
                              Seen-example flow check with the 1.7B decoder.
  decoder-1.7b --overwrite-output-dir
                              Full encoder-0.6B/decoder-1.7B run, then test last.pt.
  train [--overwrite-output-dir]
  eval [--max-samples N]      Evaluate the existing last.pt only.
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
