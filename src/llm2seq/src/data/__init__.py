"""LLM2Seq data utilities."""

from .dataset import Seq2SeqDataset
from .collator import Seq2SeqCollator


def preprocess_and_save(*args, **kwargs):
    from .preprocess import preprocess_and_save as _preprocess_and_save

    return _preprocess_and_save(*args, **kwargs)

__all__ = [
    "Seq2SeqDataset",
    "Seq2SeqCollator",
    "preprocess_and_save",
]
