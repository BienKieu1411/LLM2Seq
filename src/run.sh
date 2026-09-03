#!/usr/bin/env bash
set -Eeuo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SRC_ROOT}"

MODE="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "${MODE}" in
  eviseq)
    exec bash "${SRC_ROOT}/eviseq_v2/scripts/run.sh" "$@"
    ;;
  t5gemma-wiki-1b)
    exec bash "${SRC_ROOT}/T5Gemma/run_pipeline.sh" "$@"
    ;;
  t5gemma-wiki-4b)
    exec bash "${SRC_ROOT}/T5Gemma/run_4b_pipeline.sh" "$@"
    ;;
  t5gemma-cnndm)
    exec bash "${SRC_ROOT}/T5Gemma/run_cnndm_pipeline.sh" "$@"
    ;;
  t5gemma-pubmed)
    exec bash "${SRC_ROOT}/T5Gemma/run_pubmed_pipeline.sh" "$@"
    ;;
  t5gemma-lrsum)
    exec bash "${SRC_ROOT}/T5Gemma/run_lrsum_pipeline.sh" "$@"
    ;;
  help|-h|--help)
    cat <<'EOF'
Project launcher

  bash run.sh eviseq <mode> [arguments]
  bash run.sh t5gemma-wiki-1b
  bash run.sh t5gemma-wiki-4b
  bash run.sh t5gemma-cnndm [all|1b|4b]
  bash run.sh t5gemma-pubmed [all|1b|4b]
  bash run.sh t5gemma-lrsum

The maintained architecture and all task commands are under eviseq_v2.
EOF
    ;;
  *)
    exec bash "${SRC_ROOT}/eviseq_v2/scripts/run.sh" "${MODE}" "$@"
    ;;
esac
