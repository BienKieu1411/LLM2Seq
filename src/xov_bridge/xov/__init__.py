"""XOV: Cross-Tokenizer Ordered Value Bridge for text summarization."""

from .config import load_config

__all__ = ["XOVModel", "load_config"]


def __getattr__(name):
    if name == "XOVModel":
        from .modeling.model import XOVModel

        return XOVModel
    raise AttributeError(name)
