"""
Cascaded / Sequential Multi-Token Prediction (MTP) for LLM2Seq.

Each depth receives information from the previous depth's hidden state
combined with the predicted future token embedding, maintaining a causal
chain across MTP depths.

Architecture:
    Depth 0: decoder state → y_t (main head)
    Depth 1: combine(h_0, emb(y_t)) → TF block → y_{t+1}
    Depth 2: combine(h_1, emb(y_{t+1})) → TF block → y_{t+2}
    ...

Key advantage over parallel MTP: keeps autoregressive dependency
between predicted future tokens, potentially higher acceptance rate.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadedMTPBlock(nn.Module):
    """
    Single block in the cascaded MTP chain.

    Takes hidden state from previous depth + future token embedding,
    projects them together, runs through a small Transformer block,
    and produces hidden states for the next depth.

    Args:
        hidden_size: Decoder hidden size.
        num_heads: Number of attention heads.
        ffn_size: FFN intermediate size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 12,
        ffn_size: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # Combine previous hidden state + future embedding
        self.concat_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.proj_norm = nn.LayerNorm(hidden_size)

        # Self-attention + FFN block (simplified single-layer transformer)
        self.self_attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_size, hidden_size, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        prev_hidden: torch.Tensor,
        future_embed: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for one cascaded MTP depth.

        Args:
            prev_hidden: [B, T, D] — hidden states from previous depth.
            future_embed: [B, T, D] — embedding of the future token at this depth.
            causal_mask: [T, T] — causal attention mask.

        Returns:
            h_out: [B, T, D] — hidden states for this depth.
        """
        # Concatenate and project
        combined = torch.cat([prev_hidden, future_embed], dim=-1)  # [B, T, 2D]
        h = self.concat_proj(combined)  # [B, T, D]
        h = self.proj_norm(h)

        # Self-attention with residual
        residual = h
        h_norm = self.self_attn_norm(h)
        if not self.training and h_norm.size(1) == 1 and causal_mask is None:
            # Exact length-1 fast path for inference. With a single query/key,
            # attention weight is 1, so MHA reduces to V projection + out_proj.
            embed_dim = self.hidden_size
            in_proj_weight = self.self_attn.in_proj_weight
            in_proj_bias = self.self_attn.in_proj_bias
            v_weight = in_proj_weight[2 * embed_dim : 3 * embed_dim]
            v_bias = None if in_proj_bias is None else in_proj_bias[2 * embed_dim : 3 * embed_dim]
            h_attn = F.linear(h_norm, v_weight, v_bias)
            h_attn = self.self_attn.out_proj(h_attn)
        else:
            h_attn, _ = self.self_attn(
                h_norm, h_norm, h_norm,
                attn_mask=causal_mask,
                need_weights=False,
            )
        h = residual + h_attn

        # FFN with residual
        residual = h
        h = residual + self.ffn(self.ffn_norm(h))

        return h


class CascadedMTP(nn.Module):
    """
    Cascaded Multi-Token Prediction module.

    Maintains autoregressive dependency across MTP depths:
    each depth uses the prediction from the previous depth
    as additional context.

    Args:
        hidden_size: Decoder hidden size.
        vocab_size: Vocabulary size.
        num_heads: Number of MTP depths (future tokens to predict).
        attn_heads: Number of attention heads in each block.
        ffn_size: FFN intermediate size.
        dropout: Dropout rate.
        embedding_layer: Shared embedding layer (from decoder).
        lm_head: Shared LM head (from main model).
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_heads: int = 4,
        attn_heads: int = 12,
        ffn_size: int = 3072,
        dropout: float = 0.1,
        embedding_layer: Optional[nn.Embedding] = None,
        lm_head: Optional[nn.Linear] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        # Cascaded blocks
        self.blocks = nn.ModuleList([
            CascadedMTPBlock(
                hidden_size=hidden_size,
                num_heads=attn_heads,
                ffn_size=ffn_size,
                dropout=dropout,
            )
            for _ in range(num_heads)
        ])

        # Shared embedding for looking up future token representations
        if embedding_layer is not None:
            self.embed_tokens = embedding_layer
        else:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Shared or dedicated LM head for predicting tokens
        if lm_head is not None:
            self.lm_head = lm_head
        else:
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize non-shared params
        for block in self.blocks:
            block.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        decoder_states: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Forward pass for cascaded MTP (training mode).

        During training, we use teacher-forced future tokens for the embeddings
        at each depth, shifted appropriately.

        Args:
            decoder_states: [B, T, D] — main decoder hidden states.
            decoder_input_ids: [B, T] — decoder input token IDs (for teacher-forcing).

        Returns:
            List of K tensors, each [B, T, vocab_size] — logits at each depth.
        """
        bsz, seq_len, _ = decoder_states.shape
        device = decoder_states.device

        # Causal mask for self-attention in MTP blocks
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )

        h_prev = decoder_states
        mtp_logits = []

        for k, block in enumerate(self.blocks):
            # Get future token embedding (shifted by k+1 positions)
            # During training: use ground-truth shifted target IDs
            shift = k + 1
            if shift < seq_len:
                # Shift the input IDs to get future tokens
                future_ids = torch.zeros_like(decoder_input_ids)
                future_ids[:, :-shift] = decoder_input_ids[:, shift:]
                # Pad the last `shift` positions with zeros (padding token)
                future_embed = self.embed_tokens(future_ids)
            else:
                # All positions are beyond sequence — use zero embeddings
                future_embed = torch.zeros_like(decoder_states)

            # Run through cascaded block
            h_k = block(h_prev, future_embed, causal_mask)

            # Predict token at this depth
            logits_k = self.lm_head(h_k)  # [B, T, vocab_size]
            mtp_logits.append(logits_k)

            # Pass hidden states to next depth
            h_prev = h_k

        return mtp_logits

    @torch.no_grad()
    def generate_draft(
        self,
        decoder_states: torch.Tensor,
        main_token_id: torch.Tensor,
    ) -> List[dict]:
        """
        Generate draft tokens cascadedly for inference.

        Args:
            decoder_states: [B, 1, D] — decoder state at current position.
            main_token_id: [B, 1] — main head's predicted token.

        Returns:
            List of K dicts with "token_ids" and "confidence".
        """
        h_prev = decoder_states
        prev_token = main_token_id
        drafts = []

        for block in self.blocks:
            future_embed = self.embed_tokens(prev_token)  # [B, 1, D]
            h_k = block(h_prev, future_embed, causal_mask=None)

            logits_k = self.lm_head(h_k)  # [B, 1, vocab_size]
            # Skip expensive softmax since we only need argmax for verified decoding
            token_ids = logits_k.argmax(dim=-1)
            
            drafts.append({
                "token_ids": token_ids,
                "confidence": None,  # Not used in verified mode
            })

            # Next depth uses this prediction
            h_prev = h_k
            prev_token = token_ids

        return drafts
