#!/usr/bin/env bash
set -Eeuo pipefail

# Mixed-data recipe using a local EmbeddingGemma-2 text encoder checkpoint.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${EMBEDDINGGEMMA2_MODEL:-}"
args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  case "${args[index]}" in
    --encoder-model)
      if ((index + 1 >= ${#args[@]})); then
        echo "Missing value for --encoder-model" >&2
        exit 2
      fi
      MODEL_PATH="${args[index + 1]}"
      ((index += 1))
      ;;
    --encoder-model=*) MODEL_PATH="${args[index]#*=}" ;;
    -h|--help)
      cat <<'EOF'
Usage: EMBEDDINGGEMMA2_MODEL=/local/path bash scripts/run_mixed_wikilingua_embeddinggemma2.sh [options]
   or: bash scripts/run_mixed_wikilingua_embeddinggemma2.sh --encoder-model /local/path [options]

Runs the existing mixed-summary + WikiLingua training flow with a local
EmbeddingGemma-2 checkpoint as the encoder. All options from
run_mixed_wikilingua.sh are accepted, including --summary-data-dir, --decoder-model,
--train-batch-size, --epochs, --cuda-visible-devices, and --output-dir.

The encoder must already be downloaded on the machine. The runner is offline
and does not fetch model files from Hugging Face.
EOF
      exit 0
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Pass --encoder-model PATH or set EMBEDDINGGEMMA2_MODEL to the local EmbeddingGemma-2 checkpoint directory" >&2
  exit 2
fi

export ENCODER_MODEL="${MODEL_PATH}"
export AFMR_CONFIG="${AFMR_CONFIG:-${ROOT}/configs/afmr_summary_wikilingua.yaml}"
export MIXED_DATA_DIR="${MIXED_DATA_DIR:-${ROOT}/datasets/summary_wikilingua_mixed_embeddinggemma2}"
export AFMR_OUTPUT_DIR="${AFMR_OUTPUT_DIR:-${ROOT}/runs/afmr/summary_wikilingua_mixed_embeddinggemma2}"

exec bash "${ROOT}/scripts/run_mixed_wikilingua.sh" "${args[@]}"
