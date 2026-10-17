#!/usr/bin/env python3
"""Source-checkout launcher; no plugin installation or model download required."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "eviseq_new"))
sys.path.insert(0, str(ROOT))

from seam_grpo.train import main

if __name__ == "__main__":
    main()
