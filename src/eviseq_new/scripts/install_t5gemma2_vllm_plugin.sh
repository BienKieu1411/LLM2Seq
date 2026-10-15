#!/usr/bin/env bash
set -euo pipefail

# Installs only the T5Gemma 2 vLLM plugin. --no-deps is intentional: this must
# not replace the server's existing CUDA/PyTorch/vLLM stack.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_REF="${T5GEMMA2_VLLM_PLUGIN_REF:-e43532f575c895edf7427c04a62945182cb887d6}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PLUGIN_SOURCE="git+https://github.com/d0rj/t5gemma2-vllm-plugin.git@${PLUGIN_REF}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the Python in the server's vLLM environment." >&2
  exit 127
fi

VERSION="$(${PYTHON_BIN} -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
if [[ -z "${VERSION}" ]]; then
  echo "ERROR: vLLM is not importable from ${PYTHON_BIN}; activate the existing vLLM environment first." >&2
  exit 1
fi
echo "Using vLLM ${VERSION} from ${PYTHON_BIN}"
if [[ "${VERSION}" != "0.25.1" ]]; then
  echo "WARNING: this install script was prepared for vLLM 0.25.1; detected ${VERSION}." >&2
fi

"${PYTHON_BIN}" -m pip install --no-deps "${PLUGIN_SOURCE}"

"${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import importlib.util

dist = importlib.metadata.distribution("t5gemma2-vllm-plugin")
if importlib.util.find_spec("t5gemma2_vllm_plugin") is None:
    raise SystemExit("Plugin distribution installed, but its module cannot be imported")
print(f"Installed T5Gemma 2 vLLM plugin {dist.version}")
print("The plugin will be discovered through the vllm.general_plugins entry point.")
PY

echo "Plugin install complete. The model checkpoint itself is never downloaded by this script."
echo "Next: bash \"${SCRIPT_DIR}/serve_t5gemma2_vllm.sh\" /path/to/local/t5gemma2-checkpoint"
