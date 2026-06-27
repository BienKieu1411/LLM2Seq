#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"
cd "${LLM2SEQ_ROOT}"

required_files=(
  "${LLM2SEQ_ROOT}/requirements.txt"
  "${LLM2SEQ_ROOT}/src/__init__.py"
  "${LLM2SEQ_ROOT}/src/training/trainer.py"
  "${LLM2SEQ_ROOT}/src/inference/generate_mtp.py"
  "${LLM2SEQ_ROOT}/configs/wikilingua_phase1.yaml"
  "${LLM2SEQ_ROOT}/configs/wikilingua_phase2.yaml"
  "${LLM2SEQ_ROOT}/configs/wikilingua_phase3.yaml"
  "${LLM2SEQ_ROOT}/scripts/run_pipeline.sh"
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

llm2seq_root = Path(os.environ["LLM2SEQ_ROOT"]).resolve()
spec = importlib.util.find_spec("src")
if spec is None or spec.origin is None:
    raise SystemExit("Cannot import bundled llm2seq package")
origin = Path(spec.origin).resolve()
if llm2seq_root not in origin.parents:
    raise SystemExit(f"llm2seq resolves outside LLM2Seq folder: {origin}")
print(f"Bundled llm2seq import OK: {origin}")
PY

echo "LLM2Seq folder smoke check OK."
