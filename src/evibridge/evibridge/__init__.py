"""EviBridge: evidence-planned decoder-only to encoder-decoder summarization."""

from .bridge import EvidenceBridge
from .model import EviBridgeSeq2Seq

__all__ = ["EvidenceBridge", "EviBridgeSeq2Seq"]
