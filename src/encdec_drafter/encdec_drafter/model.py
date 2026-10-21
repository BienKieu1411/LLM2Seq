from __future__ import annotations

from typing import Any, Dict

from .adapters.t5gemma2 import T5Gemma2Adapter


def load_adapter(config: Dict[str, Any]) -> T5Gemma2Adapter:
    return T5Gemma2Adapter.load(config)
