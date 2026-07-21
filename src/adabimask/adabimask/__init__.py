"""AdaBiMask: budgeted layer-wise bidirectionalization for seq2seq models.

Submodules are intentionally not imported eagerly: configuration and launcher
tools should remain usable before heavyweight training dependencies are loaded.
"""

STANDALONE_API_VERSION = 2

__all__ = ["STANDALONE_API_VERSION", "config", "mask_policy", "model"]
