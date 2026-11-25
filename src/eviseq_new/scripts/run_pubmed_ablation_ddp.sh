#!/usr/bin/env bash
set -Eeuo pipefail

EV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$EV/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
ABLATION_OUTPUT_ROOT="${ABLATION_OUTPUT_ROOT:-$EV/runs/afmr}"
DECODER_MODEL="${DECODER_MODEL:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"
NEMOTRON_ENCODER="${NEMOTRON_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Nemotron-3-Embed-1B-BF16}"
LLAMA_ENCODER="${LLAMA_ENCODER:-/workspace/storage-shared/nlp/dungdx4/BERT/Llama-3.2-1B-Instruct}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-3}"
TRAIN_SEED="${TRAIN_SEED:-42}"
cd "$PROJECT_ROOT"

export PYTHONPATH="$EV${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYROUGE_HOME_DIR="${PYROUGE_HOME_DIR:-/workspace/storage-shared/nlp/dungdx4/textsum_platform_eval/pyrouge-master/tools/ROUGE-1.5.5}"

for model_path in "$DECODER_MODEL" "$NEMOTRON_ENCODER" "$LLAMA_ENCODER"; do
  [[ -d "$model_path" ]] || {
    echo "Model directory not found: $model_path" >&2
    exit 1
  }
done

run_encoder_nocopy() {
  local name="$1"
  local encoder_model="$2"
  local run_dir="$ABLATION_OUTPUT_ROOT/pubmed_${name}_ddp_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$run_dir"

  "$PYTHON_BIN" - "$EV" "$run_dir" "$name" "$encoder_model" "$DECODER_MODEL" \
    "$TRAIN_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS" "$TRAIN_SEED" <<'PY'
import sys
from pathlib import Path

import yaml

from eviseq_afmr.config import load_config, validate_config

ev, run_dir = map(Path, sys.argv[1:3])
name, encoder_model, decoder_model = sys.argv[3:6]
batch_size, gradient_accumulation_steps, seed = map(int, sys.argv[6:9])
config = load_config(ev / "configs/afmr_pubmed.yaml")
config.pop("_meta", None)
config["experiment"]["name"] = f"pubmed_{name}"
config["experiment"]["output_dir"] = str(run_dir)
config["architecture"]["bridge_mode"] = "afmr"
config["decoder"]["grounded_copy"]["enabled"] = False
config["model"]["encoder_name"] = encoder_model
config["model"]["decoder_name"] = decoder_model
for split in ("train", "validation", "test"):
    config["data"][f"{split}_file"] = str(ev / "datasets/pubmed" / f"{split}.jsonl")
config["training"]["batch_size"] = batch_size
config["training"]["gradient_accumulation_steps"] = gradient_accumulation_steps
config["training"]["seed"] = seed
validate_config(config)
(run_dir / "input_config.yaml").write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
print(f"{name}: {run_dir}")
PY

  if [[ "${ABLATION_CONFIG_ONLY:-false}" == true ]]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES=0,1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=2 "$EV/run_afmr.py" train "$run_dir/input_config.yaml"
  [[ -s "$run_dir/resolved_config.yaml" && -s "$run_dir/last.pt" ]] || {
    echo "Training did not produce resolved_config.yaml and last.pt in $run_dir" >&2
    return 1
  }

  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$EV/run_afmr.py" evaluate \
    "$run_dir/resolved_config.yaml" "$run_dir/last.pt" "$run_dir/shard0.jsonl" \
    --split test --batch-size 32 --shard-rank 0 --num-shards 2 &
  local pid0=$!

  CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$EV/run_afmr.py" evaluate \
    "$run_dir/resolved_config.yaml" "$run_dir/last.pt" "$run_dir/shard1.jsonl" \
    --split test --batch-size 32 --shard-rank 1 --num-shards 2 &
  local pid1=$!

  local eval_failed=0
  wait "$pid0" || eval_failed=1
  wait "$pid1" || eval_failed=1
  (( eval_failed == 0 )) || return 1

  "$PYTHON_BIN" "$EV/scripts/merge_eval_shards.py" \
    --output "$run_dir/last_test_predictions.jsonl" \
    "$run_dir/shard0.jsonl" "$run_dir/shard1.jsonl"

  local rouge_script="$PROJECT_ROOT/src/rouge155/evaluate_rouge.py"
  if [[ ! -f "$rouge_script" ]]; then
    rouge_script="$PROJECT_ROOT/src/evaluation/evaluate_rouge.py"
  fi
  [[ -f "$rouge_script" ]] || {
    echo "ROUGE-1.5.5 evaluator not found under $PROJECT_ROOT/src" >&2
    return 1
  }
  "$PYTHON_BIN" "$rouge_script" "$run_dir/last_test_predictions.jsonl"
}

run_encoder_nocopy nemotron "$NEMOTRON_ENCODER"
run_encoder_nocopy llama "$LLAMA_ENCODER"
