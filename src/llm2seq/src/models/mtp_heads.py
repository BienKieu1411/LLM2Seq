"""
Parallel Multi-Token Prediction (MTP) Heads for LLM2Seq.

Each head independently predicts a future token y_{t+k} from the same
decoder hidden state, allowing speculative multi-token generation.

Architecture:
    Main head: predict y_t        (shared LM head)
    MTP head 1: predict y_{t+1}   (independent linear head)
    MTP head 2: predict y_{t+2}
    ...
    MTP head K: predict y_{t+K}

Loss:
    L_MTP = Σ_k α_k * CE(P_k, y_{t+k})
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ParallelMTPHeads(nn.Module):
    """
    Parallel multi-token prediction heads.

    Each head is an independent linear projection from decoder hidden states
    to vocabulary logits, predicting a future token at offset k.

    Args:
        hidden_size: Decoder hidden size (d_dec).
        vocab_size: Vocabulary size.
        num_heads: Number of MTP heads (predicting y_{t+1}, ..., y_{t+K}).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # Each MTP head: LayerNorm + Linear
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, vocab_size, bias=False),
            )
            for _ in range(num_heads)
        ])

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        decoder_states: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass: each head projects decoder states to vocab logits.

        Args:
            decoder_states: [B, T, d_dec] — decoder hidden states.
            decoder_input_ids: [B, T] — decoder input IDs (unused for parallel).

        Returns:
            List of K tensors, each [B, T, vocab_size] — logits for future tokens.
        """
        mtp_logits = []
        for head in self.heads:
            logits_k = head(decoder_states)  # [B, T, vocab_size]
            mtp_logits.append(logits_k)
        return mtp_logits

    def get_draft_tokens(
        self,
        decoder_states: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Get greedy draft tokens from each MTP head (for inference).

        Args:
            decoder_states: [B, 1, d_dec] — decoder state at current position.

        Returns:
            List of K tensors, each [B, 1] — predicted token IDs.
        """
        draft_tokens = []
        for head in self.heads:
            logits = head(decoder_states)  # [B, 1, vocab_size]
            tokens = logits.argmax(dim=-1)  # [B, 1]
            draft_tokens.append(tokens)
        return draft_tokens

    def get_draft_tokens_with_confidence(
        self,
        decoder_states: torch.Tensor,
    ) -> List[dict]:
        """
        Get draft tokens with confidence scores (for confidence-adaptive decoding).

        Args:
            decoder_states: [B, 1, d_dec] — decoder state at current position.

        Returns:
            List of K dicts, each with "token_ids" [B, 1] and "confidence" [B, 1].
        """
        results = []
        for head in self.heads:
            logits = head(decoder_states)  # [B, 1, vocab_size]
            probs = torch.softmax(logits, dim=-1)
            confidence, token_ids = probs.max(dim=-1)  # [B, 1]
            results.append({
                "token_ids": token_ids,
                "confidence": confidence,
            })
        return results
