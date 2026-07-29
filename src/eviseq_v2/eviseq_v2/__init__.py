"""EviSeq: native encoder conversion with a minimal seq2seq interface."""

from .config import architecture_contract, load_config
from .model import EviSeq

__all__ = ["EviSeq", "architecture_contract", "load_config"]
