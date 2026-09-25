#!/usr/bin/env bash
set -Eeuo pipefail

if (( $# < 1 || $# > 2 )); then
  echo "Usage: bash src/T5Gemma/eval_booksum.sh RUN_DIR [EVAL_DIR]" >&2
  exit 2
fi

[[ -d "$1/final_model" && -s "$1/training_config.yaml" ]] || {
  echo "ERROR: RUN_DIR must contain final_model/ and training_config.yaml: $1" >&2
  exit 1
}

export RUN_MODE=eval
export T5GEMMA_BOOKSUM_RUN_DIR="$(cd "$1" && pwd -P)"
if (( $# == 2 )); then
  export T5GEMMA_BOOKSUM_EVAL_DIR="$2"
fi

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/run_booksum.sh"
