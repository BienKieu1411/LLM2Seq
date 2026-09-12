"""Single-GPU decoder-only summarization baselines.

The package keeps model loading, prompt construction, fine-tuning and
evaluation in one small, reproducible folder.  It is intentionally separate
from the EviSeq architecture so that the baselines cannot change the AFMR
training graph.
"""

__all__ = ["config", "data", "evaluate", "metrics", "suite", "train"]
