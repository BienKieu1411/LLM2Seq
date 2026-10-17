#!/usr/bin/env python3
"""Add per-sample gold word counts without loading a tokenizer/model."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT.parent / "eviseq_new", ROOT.parent / "seam_grpo", ROOT):
    sys.path.insert(0, str(path))

from seam_length_grpo.prepare import main

if __name__ == "__main__":
    main()
