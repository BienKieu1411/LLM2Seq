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

"${PYTHON_BIN}" -m pip install -r "${H200_ROOT}/requirements.txt"

"${PYTHON_BIN}" - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch. Install a torch build compatible with the NVIDIA driver.")
print("gpu:", torch.cuda.get_device_name(0))
print("gpu capability:", torch.cuda.get_device_capability(0))
print("supported arch list:", torch.cuda.get_arch_list())
cuda_build = torch.version.cuda
if cuda_build and not cuda_build.startswith("12.4"):
    raise SystemExit(
        f"This H200 setup expects a CUDA 12.4 PyTorch wheel, but torch was built with CUDA {cuda_build}. "
        "Use TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu124 or update the NVIDIA driver."
    )
if torch.cuda.get_device_capability(0)[0] >= 9 and "sm_90" not in torch.cuda.get_arch_list():
    raise SystemExit("Installed PyTorch wheel does not advertise sm_90 support for H100/H200-class GPUs.")
PY
