"""Two-stage training, objectives, and checkpoint utilities."""

from typing import Any

__all__ = ["train"]


def __getattr__(name: str) -> Any:
    if name == "train":
        from .trainer import train

        return train
    raise AttributeError(name)
