"""Greedy generation, task metrics, and checkpoint evaluation."""

from typing import Any

__all__ = ["evaluate"]


def __getattr__(name: str) -> Any:
    if name == "evaluate":
        from .evaluator import evaluate

        return evaluate
    raise AttributeError(name)
