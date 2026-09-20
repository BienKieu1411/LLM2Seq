"""AFMR: Adaptive Full-Memory Residual bridge for text summarization."""

from .config import load_config

__all__ = ["SideMemorySummarizer", "load_config"]


def __getattr__(name):
    if name == "SideMemorySummarizer":
        from .modeling.model import SideMemorySummarizer

        return SideMemorySummarizer
    raise AttributeError(name)
