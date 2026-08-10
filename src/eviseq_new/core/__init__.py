"""EviSeq: native encoder conversion with a minimal seq2seq interface."""

from typing import Any

from .configuration import load_config, resolve_data_path

__all__ = ["EviSeq", "load_config", "resolve_data_path"]


def __getattr__(name: str) -> Any:
    if name == "EviSeq":
        from .modeling.architecture import EviSeq

        return EviSeq
    raise AttributeError(name)
