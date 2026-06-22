#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${H200_ROOT}"

required_files=(
  "${H200_ROOT}/requirements.txt"
  "${H200_ROOT}/llm2seq/__init__.py"
  "${H200_ROOT}/llm2seq/src/training/trainer.py"
  "${H200_ROOT}/llm2seq/src/inference/generate_mtp.py"
  "${H200_ROOT}/configs/phase1_warmup_4096.yaml"
  "${H200_ROOT}/configs/phase2_lora_encoder_4096.yaml"
  "${H200_ROOT}/configs/phase3_mtp_self_distill_4096.yaml"
  "${H200_ROOT}/scripts/run_h200_pipeline.sh"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    echo "Missing required file: ${file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - <<'PY'
import importlib.util
from pathlib import Path
import os

h200_root = Path(os.environ["H200_ROOT"]).resolve()
spec = importlib.util.find_spec("llm2seq")
if spec is None or spec.origin is None:
    raise SystemExit("Cannot import bundled llm2seq package")
origin = Path(spec.origin).resolve()
if h200_root not in origin.parents:
    raise SystemExit(f"llm2seq resolves outside H200 folder: {origin}")
print(f"Bundled llm2seq import OK: {origin}")
PY

echo "H200 folder smoke check OK."
