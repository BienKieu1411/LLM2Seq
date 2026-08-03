"""Model architecture components: encoder, bridge, decoder, and attention."""

from typing import Any

__all__ = ["EviSeq"]


def __getattr__(name: str) -> Any:
    if name == "EviSeq":
        from .architecture import EviSeq

        return EviSeq
    raise AttributeError(name)
