from __future__ import annotations

import importlib
import sys

try:
    import eviseq
except ModuleNotFoundError:
    eviseq = importlib.import_module("core")
    sys.modules["eviseq"] = eviseq
