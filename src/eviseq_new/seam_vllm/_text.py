"""Small, vLLM-independent helpers for source-text processor inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def normalize_text_payload(data: Any) -> Sequence[str] | None:
    """Preserve processor-cache empty batches instead of inventing an empty source."""

    if data is None:
        return None
    if isinstance(data, str):
        return [data]
    if isinstance(data, (list, tuple)):
        if not data:
            return None
        if all(isinstance(value, str) for value in data):
            return data
    raise TypeError(f"SEAM source must be text or a list of texts, got {type(data)}")
