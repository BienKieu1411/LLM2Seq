#!/usr/bin/env python3
"""Run EviSeq-KD directly from this source directory without installing it."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eviseq_kd.trainer import main  # noqa: E402

if __name__ == "__main__":
    main()
