set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

PYTHON_BIN="${PYTHON:-python3}"
CONFIG="${CONFIG:-${ROOT}/runs/afmr/pubmed_full_vanilla_multiseed_20261126_013950/full_seed43/resolved_config.yaml}"
QWEN06="${QWEN06:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-0.6B}"
QWEN17="${QWEN17:-/workspace/storage-shared/models/Qwen3-1.7B}"
QWEN4="${QWEN4:-/workspace/storage-shared/nlp/users/dungdx4/BERT/Qwen3-4B}"
T5GEMMA="${T5GEMMA:-/workspace/storage-shared/nlp/users/dungdx4/BERT/t5gemma-2-1b-1b}"
SYSTEMS="${SYSTEMS:-seam vanilla qwen3_0_6b qwen3_1_7b qwen3_4b t5gemma2_1b}"
GPU="${GPU:-0}"
REPEATS="${REPEATS:-5}"
PASSES="${PASSES:-2}"
OUT="${OUT:-${ROOT}/runs/inference_cost/$(date +%Y%m%d_%H%M%S)}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

[[ "${GPU}" =~ ^[0-9]+$ ]] || die "GPU must be one GPU index, e.g. GPU=1"
[[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]] || die "REPEATS must be a positive number"
[[ "${PASSES}" == 1 || "${PASSES}" == 2 ]] || die "PASSES must be 1 or 2"
[[ -f "${CONFIG}" ]] || die "Run config not found: ${CONFIG}"

read -r -a FORWARD <<< "${SYSTEMS}"
(( ${#FORWARD[@]} > 0 )) || die "SYSTEMS is empty"
BACKWARD=()
for system in "${FORWARD[@]}"; do
  BACKWARD=("${system}" ${BACKWARD[@]+"${BACKWARD[@]}"})
  case "${system}" in
    seam | vanilla) folder="" ;;
    qwen3_0_6b) folder="${QWEN06}" ;;
    qwen3_1_7b) folder="${QWEN17}" ;;
    qwen3_4b) folder="${QWEN4}" ;;
    t5gemma2_1b) folder="${T5GEMMA}" ;;
    *) die "Unknown system: ${system}" ;;
  esac
  [[ -z "${folder}" || -d "${folder}" ]] || die "Model folder not found for ${system}: ${folder}"
done

for name in pass_a pass_b; do
  [[ ! -e "${OUT}/${name}.jsonl" ]] || die "${OUT} already holds results; choose another OUT"
done

busy="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid,used_memory --format=csv,noheader)" ||
  die "nvidia-smi could not query GPU ${GPU}"
if [[ -n "${busy}" && "${ALLOW_BUSY:-0}" != 1 ]]; then
  die "GPU ${GPU} is in use (pid, memory): ${busy}. Choose another GPU or set ALLOW_BUSY=1"
fi

mkdir -p "${OUT}"
{
  date -u
  hostname || true
  nvidia-smi --id="${GPU}"
  uptime || true
  echo "CPUs: $(getconf _NPROCESSORS_ONLN)"
  git -C "${ROOT}" log -1 --format='commit %h %s' 2>/dev/null || true
  "${PYTHON_BIN}" -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'
} 2>&1 | tee "${OUT}/environment.txt"

FAILED=""
run_pass() {
  local name="$1"
  shift
  local system
  for system in "$@"; do
    echo "=== ${name}: ${system} ==="
    if ! "${PYTHON_BIN}" "${ROOT}/scripts/inference_cost.py" \
      --config "${CONFIG}" --systems "${system}" \
      --model "qwen3_0_6b=${QWEN06}" --model "qwen3_1_7b=${QWEN17}" \
      --model "qwen3_4b=${QWEN4}" --model "t5gemma2_1b=${T5GEMMA}" \
      --repeats "${REPEATS}" --output "${OUT}/${name}.jsonl" 2>&1 | tee -a "${OUT}/${name}.log"; then
      FAILED="${FAILED:+${FAILED}, }${name} ${system}"
    fi
  done
}

RESULTS=("${OUT}/pass_a.jsonl")
run_pass pass_a "${FORWARD[@]}"
if [[ "${PASSES}" == 2 ]]; then
  RESULTS+=("${OUT}/pass_b.jsonl")
  run_pass pass_b "${BACKWARD[@]}"
fi

echo "=== Summary ==="
"${PYTHON_BIN}" "${ROOT}/scripts/compare_inference_cost.py" "${RESULTS[@]}" --expect "${FORWARD[@]}" 2>&1 |
  tee "${OUT}/summary.txt"
echo "Results: ${OUT}"

if [[ -n "${FAILED}" ]]; then
  echo "FAILED: ${FAILED}; see the pass logs in ${OUT}"
  exit 1
fi
if grep -q "is missing" "${OUT}/summary.txt"; then
  echo "Some workloads were not measured; see ${OUT}/summary.txt"
  exit 1
fi
