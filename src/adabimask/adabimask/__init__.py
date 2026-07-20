"""AdaBiMask: budgeted layer-wise bidirectionalization for seq2seq models.

Submodules are intentionally not imported eagerly: configuration and launcher
tools should remain usable before heavyweight training dependencies are loaded.
"""

__all__ = ["config", "mask_policy", "model"]
