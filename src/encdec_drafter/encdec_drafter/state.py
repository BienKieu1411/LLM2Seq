from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class SourceState:
    source_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    encoder_outputs: Any
    fingerprint: str
    draft_memory: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def hidden_states(self) -> torch.Tensor:
        return self.encoder_outputs.last_hidden_state

    @property
    def valid_length(self) -> int:
        return int(self.attention_mask[0].sum().item())


@dataclass
class DecodeState:
    source: SourceState
    decoder_ids: List[int]
    target_cache: Any
    target_cache_length: int
    feature_buffer: torch.Tensor
    feature_length: int = 0
    generated_count: int = 0
    finished: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def pending_token(self) -> int:
        if not self.decoder_ids:
            raise RuntimeError("Decode state has no pending token")
        return self.decoder_ids[-1]

    def append_processed_features(self, features: torch.Tensor) -> None:
        if features.ndim != 3 or features.shape[0] != 1:
            raise ValueError("T5Gemma2 single-request features must have shape [1, time, width]")
        end = self.feature_length + features.shape[1]
        if end > self.feature_buffer.shape[1]:
            raise RuntimeError("Target feature buffer exceeded the configured output-token capacity")
        self.feature_buffer[:, self.feature_length : end].copy_(features)
        self.feature_length = end

    def assert_nonterminal_invariant(self) -> None:
        if self.finished:
            return
        expected = len(self.decoder_ids) - 1
        if self.target_cache_length != expected:
            raise RuntimeError(
                f"Target cache length {self.target_cache_length} does not match committed prefix invariant {expected}"
            )
        if self.feature_length != self.target_cache_length:
            raise RuntimeError(
                f"Accepted target feature length {self.feature_length} differs from target cache length "
                f"{self.target_cache_length}"
            )
