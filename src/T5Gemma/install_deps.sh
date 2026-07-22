#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"

# Preserve the configured bienkieu_env by default. CUDA 12.4 wheels predate
# native Blackwell support and must never replace a working B200 installation.
INSTALL_TORCH="${INSTALL_TORCH:-false}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
if [[ "${INSTALL_TORCH,,}" == "true" || "${INSTALL_TORCH}" == "1" || "${INSTALL_TORCH,,}" == "yes" ]]; then
  "${PYTHON_BIN}" -m pip install --upgrade torch --index-url "${TORCH_CUDA_INDEX}"
fi

"${PYTHON_BIN}" -m pip install -r "${T5GEMMA_ROOT}/requirements.txt"

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch. Install a CUDA wheel compatible with the NVIDIA driver.")

name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
arch_list = torch.cuda.get_arch_list()
print("gpu:", name)
print("gpu capability:", capability)
print("supported arch list:", arch_list)

if "B200" in name.upper():
    cuda_version = tuple(int(part) for part in torch.version.cuda.split(".")[:2])
    if capability < (10, 0):
        raise SystemExit(f"Expected B200 compute capability sm_100, got {capability}.")
    if cuda_version < (12, 8):
        raise SystemExit(f"B200 requires a Blackwell-capable wheel (CUDA >= 12.8), got {torch.version.cuda}.")
    if not any(arch in arch_list for arch in ("sm_100", "compute_100")):
        raise SystemExit(f"PyTorch wheel does not advertise sm_100 support: {arch_list}")
PY
