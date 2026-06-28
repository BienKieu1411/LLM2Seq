"""
Lightweight Transformer Decoder for LLM2Seq.

A custom Transformer decoder designed from scratch, featuring:
- Causal self-attention with RoPE (Rotary Position Embeddings).
- Cross-attention to encoder memory (from adaptor).
- RMSNorm (Pre-LN style) for stable training.
- KV-cache support for efficient autoregressive inference.

Architecture per layer:
    RMSNorm → CausalSelfAttention → Residual
    RMSNorm → CrossAttention → Residual
    RMSNorm → FFN → Residual
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# RMSNorm — lightweight alternative to LayerNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for sequence positions."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len

        # Precompute cos/sin cache
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for positions [0, seq_len)."""
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    seq_len = q.shape[2]
    cos = cos[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[position_offset : position_offset + seq_len].unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Multi-Head Attention (Self and Cross)
# ---------------------------------------------------------------------------


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting both causal self-attention and cross-attention.

    Args:
        hidden_size: Model hidden size.
        num_heads: Number of attention heads.
        dropout: Attention dropout.
        is_cross_attention: If True, K and V come from encoder memory.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_cross_attention: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention.

        Args:
            hidden_states: [B, T, D] — query source.
            encoder_hidden_states: [B, S, D] — key/value source for cross-attention.
            attention_mask: Broadcastable mask (causal or encoder padding).
            cos, sin: RoPE embeddings (only for self-attention).
            position_offset: Position offset for KV-cache continuation.
            kv_cache: Previous (K, V) cache tensors.
            use_cache: Whether to return updated KV cache.

        Returns:
            output: [B, T, D]
            kv_cache: Updated (K, V) if use_cache, else None.
        """
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_cross_attention:
            # Cross-attention: K, V from encoder memory
            kv_source = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            if kv_cache is not None:
                # Reuse cached encoder KV (doesn't change across decode steps)
                k, v = kv_cache
            else:
                kv_len = kv_source.size(1)
                k = self.k_proj(kv_source).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(kv_source).view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # Self-attention
            k = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Apply RoPE to queries and keys
            if cos is not None and sin is not None:
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_offset)

            # Append to KV cache
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        new_cache = (k, v) if use_cache else None
        return attn_output, new_cache


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """SwiGLU-style feed-forward network."""

    def __init__(self, hidden_size: int, ffn_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down_proj = nn.Linear(ffn_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------


class DecoderLayer(nn.Module):
    """
    Single Transformer decoder layer with:
    1. Pre-norm causal self-attention (with RoPE)
    2. Pre-norm cross-attention to encoder memory
    3. Pre-norm SwiGLU FFN

    Args:
        hidden_size: Layer hidden size.
        num_heads: Number of attention heads.
        ffn_size: FFN intermediate dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        self.self_attn_norm = RMSNorm(hidden_size)
        self.self_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            is_cross_attention=False,
        )

        # Cross-attention
        self.cross_attn_norm = RMSNorm(hidden_size)
        self.cross_attn = MultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            is_cross_attention=True,
        )

        # FFN
        self.ffn_norm = RMSNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, ffn_size, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        self_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        Forward pass for a single decoder layer.

        Returns:
            hidden_states: [B, T, D]
            cache: Dict with "self" and "cross" KV caches if use_cache.
        """
        # 1. Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states, new_self_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=self_attn_mask,
            cos=cos,
            sin=sin,
            position_offset=position_offset,
            kv_cache=self_kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 2. Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states, new_cross_cache = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=cross_attn_mask,
            kv_cache=cross_kv_cache,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # 3. FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        cache = None
        if use_cache:
            cache = {"self": new_self_cache, "cross": new_cross_cache}

        return hidden_states, cache


# ---------------------------------------------------------------------------
# Lightweight Decoder
# ---------------------------------------------------------------------------


class LightweightDecoder(nn.Module):
    """
    Complete lightweight Transformer decoder.

    Features:
    - Token + RoPE position embeddings.
    - Stack of DecoderLayers with self-attention, cross-attention, and FFN.
    - Final RMSNorm before output projection.
    - KV-cache for efficient autoregressive generation.

    Args:
        vocab_size: Vocabulary size for token embeddings.
        hidden_size: Hidden dimension (d_dec).
        num_layers: Number of decoder layers.
        num_heads: Number of attention heads.
        ffn_size: FFN intermediate size.
        max_seq_len: Maximum target sequence length.
        dropout: Dropout rate.
        tie_embeddings: If True, share input embedding with output LM head weight.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 6,
        num_heads: int = 12,
        ffn_size: int = 3072,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # RoPE
        self.rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            max_seq_len=max_seq_len,
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_size=ffn_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final norm
        self.final_norm = RMSNorm(hidden_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Xavier uniform initialization for linear layers, normal for embeddings."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create causal attention mask: [1, 1, T, T]."""
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _expand_encoder_mask(
        encoder_attention_mask: torch.Tensor,
        tgt_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expand encoder attention mask for cross-attention.

        Input: [B, S] with 1 for real tokens, 0 for padding.
        Output: [B, 1, T, S] with 0 for attend, -inf for ignore.
        """
        bsz, src_len = encoder_attention_mask.size()
        expanded = encoder_attention_mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted = (1.0 - expanded) * torch.finfo(dtype).min
        return inverted

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, Tuple[torch.Tensor, torch.Tensor]]]]]:
        """
        Forward pass through the decoder.

        Args:
            input_ids: [B, T] — decoder input token IDs.
            encoder_hidden_states: [B, S, d_dec] — memory from adaptor.
            encoder_attention_mask: [B, S] — encoder padding mask.
            attention_mask: [B, T] — decoder attention mask (unused if causal).
            past_key_values: KV-cache from previous step.
            use_cache: Whether to return updated KV cache.

        Returns:
            hidden_states: [B, T, d_dec] — decoder output.
            present_key_values: Updated KV cache (if use_cache=True).
        """
        bsz, seq_len = input_ids.size()

        # Determine position offset from cache
        position_offset = 0
        if past_key_values is not None and past_key_values[0] is not None:
            # Self-attn cache: past_key_values[0]["self"][0] has shape [B, H, cached_len, D]
            if past_key_values[0].get("self") is not None:
                position_offset = past_key_values[0]["self"][0].shape[2]

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embed_dropout(hidden_states)

        # RoPE
        cos, sin = self.rotary_emb(position_offset + seq_len)

        # Causal self-attention mask
        if past_key_values is not None and position_offset > 0:
            # During generation with cache: only attend to all previous + current
            # No explicit causal mask needed for single-token decoding
            self_attn_mask = None
        else:
            self_attn_mask = self._make_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)

        # Cross-attention mask
        cross_attn_mask = None
        if encoder_attention_mask is not None:
            cross_attn_mask = self._expand_encoder_mask(encoder_attention_mask, seq_len, hidden_states.dtype)

        # Forward through layers
        present_key_values = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            layer_past = None
            if past_key_values is not None and i < len(past_key_values):
                layer_past = past_key_values[i]

            hidden_states, layer_cache = layer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                cos=cos,
                sin=sin,
                position_offset=position_offset,
                self_kv_cache=layer_past.get("self") if layer_past else None,
                cross_kv_cache=layer_past.get("cross") if layer_past else None,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(layer_cache)

        # Final norm
        hidden_states = self.final_norm(hidden_states)

        return hidden_states, present_key_values
