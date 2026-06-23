#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/load_env.sh"

TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.20.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.5.1}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"

"${PYTHON_BIN}" -m pip uninstall -y torch torchvision torchaudio || true
"${PYTHON_BIN}" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_CUDA_INDEX}"

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

if "L40" in name.upper() and capability < (8, 9):
    raise SystemExit(f"Expected an Ada/L40-class GPU capability around sm_89, got {capability}.")
if "L40" in name.upper() and "sm_89" not in arch_list and "compute_89" not in arch_list:
    raise SystemExit("Installed PyTorch wheel does not advertise sm_89 support for L40-class GPUs.")
PY
