#!/usr/bin/env bash
# Train with DDP, evaluate two disjoint shards, then merge in dataset order.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-${ROOT}/configs/xov_pubmed.yaml}"
PYTHON="${PYTHON:-python3}"
GPU_IDS="${GPU_IDS:-0,1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
SPLIT="${SPLIT:-test}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

[[ $# -le 1 ]] || { echo "Usage: bash $0 [config.yaml]" >&2; exit 2; }
IFS=',' read -r GPU0 GPU1 EXTRA <<< "$GPU_IDS"
[[ -n "$GPU0" && -n "$GPU1" && "$GPU0" != "$GPU1" && -z "$EXTRA" ]] || {
  echo 'GPU_IDS must contain two distinct GPUs, e.g. 0,1' >&2; exit 2;
}
[[ "$EVAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid EVAL_BATCH_SIZE' >&2; exit 2; }
[[ "$SPLIT" == test || "$SPLIT" == validation ]] || { echo 'SPLIT must be test or validation' >&2; exit 2; }
# Resolve the config against the caller's directory before changing directory.
CONFIG="$("$PYTHON" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$CONFIG")"
cd "$ROOT"
RUN="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from xov.config import load_config, resolve_path
config = load_config(sys.argv[1])
print(resolve_path(config['experiment']['output_dir'], config))
PY
)"

printf 'Training on GPUs %s; output: %s\n' "$GPU_IDS" "$RUN"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 "$ROOT/run_xov.py" train "$CONFIG"

[[ -s "$RUN/last.pt" && -s "$RUN/resolved_config.yaml" ]] || {
  echo "Missing last.pt or resolved_config.yaml in $RUN" >&2; exit 1;
}
# Fresh shard directory avoids reusing predictions from an older checkpoint.
EVAL_DIR="$(mktemp -d "$RUN/${SPLIT}_eval_XXXXXX")"
PIDS=()
cleanup() {
  for pid in ${PIDS[*]:-}; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for rank in 0 1; do
  gpu="$GPU0"
  [[ "$rank" == 0 ]] || gpu="$GPU1"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$ROOT/run_xov.py" evaluate \
    "$RUN/resolved_config.yaml" "$RUN/last.pt" "$EVAL_DIR/shard_${rank}.jsonl" \
    --split "$SPLIT" --batch-size "$EVAL_BATCH_SIZE" \
    --shard-rank "$rank" --num-shards 2 \
    > "$EVAL_DIR/shard_${rank}.log" 2>&1 &
  PIDS+=("$!")
done
printf 'Evaluating on two GPUs, batch %s per GPU. Logs: %s\n' "$EVAL_BATCH_SIZE" "$EVAL_DIR"
STATUS=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || STATUS=1
done
PIDS=()
[[ "$STATUS" == 0 ]] || { echo "Evaluation failed; inspect $EVAL_DIR/shard_*.log" >&2; exit 1; }
"$PYTHON" "$ROOT/scripts/merge_eval_shards.py" \
  --output "$RUN/${SPLIT}_predictions.jsonl" \
  "$EVAL_DIR/shard_0.jsonl" "$EVAL_DIR/shard_1.jsonl"
printf 'Predictions: %s/%s_predictions.jsonl\n' "$RUN" "$SPLIT"
echo 'Merged metrics use Python ROUGE; run ROUGE-1.5.5 separately for reported scores.'
