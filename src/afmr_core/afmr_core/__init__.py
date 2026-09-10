"""AFMR: Adaptive Full-Memory Residual bridge for text summarization."""

from .config import load_config

__all__ = ["AFMRModel", "load_config"]


def __getattr__(name):
    if name == "AFMRModel":
        from .modeling.model import AFMRModel

        return AFMRModel
    raise AttributeError(name)
