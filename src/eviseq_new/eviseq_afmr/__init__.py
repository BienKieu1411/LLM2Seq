"""AFMR: Adaptive Full-Memory Residual bridge for text summarization."""

from .config import load_config

__all__ = ["EviSeqAFMR", "load_config"]


def __getattr__(name):
    if name == "EviSeqAFMR":
        from .modeling.model import EviSeqAFMR

        return EviSeqAFMR
    raise AttributeError(name)
