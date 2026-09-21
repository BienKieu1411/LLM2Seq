"""XOV: Cross-Tokenizer Ordered Value Bridge for text summarization."""

from .config import load_config

__all__ = ["EviSeqXOV", "load_config"]


def __getattr__(name):
    if name == "EviSeqXOV":
        from .modeling.model import EviSeqXOV

        return EviSeqXOV
    raise AttributeError(name)
