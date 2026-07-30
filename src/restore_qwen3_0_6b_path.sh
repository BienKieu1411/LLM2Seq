#!/usr/bin/env bash
set -Eeuo pipefail

# Restore the historical Qwen3-0.6B path recorded in resolved_config.yaml and
# Phase-2 checkpoints without copying model weights.
#
# Optional overrides:
#   OLD_QWEN_PATH=/old/path NEW_QWEN_PATH=/new/path bash restore_qwen3_0_6b_path.sh

OLD_QWEN_PATH="${OLD_QWEN_PATH:-/workspace/storage-shared/nlp/text-sum-hub/dungdx4/BERT/Qwen3-0.6B}"
NEW_QWEN_PATH="${NEW_QWEN_PATH:-/workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-0.6B}"

if [[ ! -f "${NEW_QWEN_PATH}/config.json" ]]; then
  echo "ERROR: model is incomplete or NEW_QWEN_PATH is incorrect: ${NEW_QWEN_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OLD_QWEN_PATH}")"
ln -sfn "${NEW_QWEN_PATH}" "${OLD_QWEN_PATH}"

if [[ ! -f "${OLD_QWEN_PATH}/config.json" ]]; then
  echo "ERROR: symlink was created but config.json is not reachable through it." >&2
  exit 1
fi

echo "Qwen model path restored:"
echo "  ${OLD_QWEN_PATH} -> $(readlink -f "${OLD_QWEN_PATH}")"
