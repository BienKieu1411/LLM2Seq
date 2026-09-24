#!/usr/bin/env bash
set -Eeuo pipefail

export AFMR_DATASET=govreport
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_longform.sh" "$@"
