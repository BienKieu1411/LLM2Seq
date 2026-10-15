"""Convenient vLLM entry point for locally trained SEAM/AFMR checkpoints."""

from __future__ import annotations

__all__ = ["SEAMLLM"]


def __getattr__(name: str):
    if name == "SEAMLLM":
        from .api import SEAMLLM

        return SEAMLLM
    raise AttributeError(name)
