"""Relational-key bridge experiment for text summarization."""

from .config import load_config

__all__ = ["RelationalKeyModel", "load_config"]


def __getattr__(name):
    if name == "RelationalKeyModel":
        from .modeling.model import RelationalKeyModel

        return RelationalKeyModel
    raise AttributeError(name)
