"""
Adaptor module for LLM2Seq.

Bridges encoder hidden states to decoder memory space via:
1. LayerFusion: weighted combination of multiple encoder layer outputs.
2. AdaptorMLP: projects from d_enc to d_dec.
3. Optional EncoderStack: additional Transformer encoder layers for refinement.

Pipeline: LayerFusion → MLP → Optional EncStack
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerFusion(nn.Module):
    """
    Learnable weighted sum of hidden states from selected encoder layers.

    Inspired by ELMo-style scalar mixing, but with learnable layer weights
    that are softmax-normalized before summing.

    Args:
        num_layers: Number of layers to fuse (e.g., 4 for layers [-1, -4, -8, -12]).
    """

    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        # Learnable scalar weight per layer, initialized uniformly
        self.layer_weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden_states: Tuple[torch.Tensor, ...], layer_indices: List[int]) -> torch.Tensor:
        """
        Fuse selected layers from encoder hidden states.

        Args:
            hidden_states: Tuple of all encoder layer outputs.
                Each element is [batch_size, seq_len, d_enc].
            layer_indices: Which layers to select (e.g., [-1, -4, -8, -12]).

        Returns:
            Fused representation: [batch_size, seq_len, d_enc].
        """
        assert len(layer_indices) == self.num_layers, (
            f"Expected {self.num_layers} layer indices, got {len(layer_indices)}"
        )

        # Gather selected layer hidden states
        selected = []
        num_total = len(hidden_states)
        for idx in layer_indices:
            # Support negative indexing
            actual_idx = idx if idx >= 0 else num_total + idx
            selected.append(hidden_states[actual_idx].to(dtype=self.layer_weights.dtype))

        # Stack: [num_layers, batch_size, seq_len, d_enc]
        stacked = torch.stack(selected, dim=0)

        # Softmax-normalized weights
        weights = F.softmax(self.layer_weights, dim=0)

        # Weighted sum: [batch_size, seq_len, d_enc]
        fused = torch.einsum("l,lbsd->bsd", weights, stacked)
        return fused


class AdaptorMLP(nn.Module):
    """
    MLP that projects encoder hidden states from d_enc to d_dec.

    Architecture: LayerNorm → Linear → GELU → Dropout → Linear → Dropout

    Args:
        d_enc: Encoder hidden size.
        d_dec: Decoder hidden size.
        dropout: Dropout rate.
    """

    def __init__(self, d_enc: int, d_dec: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_enc)
        self.linear1 = nn.Linear(d_enc, d_dec)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_dec, d_dec)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder representations to decoder space.

        Args:
            x: [batch_size, seq_len, d_enc]

        Returns:
            [batch_size, seq_len, d_dec]
        """
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class EncoderStack(nn.Module):
    """
    Optional stack of Transformer encoder layers for refining adapted representations.

    Uses standard Transformer encoder layers with self-attention and FFN,
    applied after the MLP adaptor for additional processing.

    Args:
        d_model: Model hidden size (should be d_dec).
        num_heads: Number of attention heads.
        ffn_size: Feed-forward intermediate size.
        num_layers: Number of encoder layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.layers = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Refine adapted representations.

        Args:
            x: [batch_size, seq_len, d_dec]
            src_key_padding_mask: [batch_size, seq_len] — True for padded positions.

        Returns:
            [batch_size, seq_len, d_dec]
        """
        return self.layers(x, src_key_padding_mask=src_key_padding_mask)


class Adaptor(nn.Module):
    """
    Full adaptor pipeline: LayerFusion → MLP → Optional EncStack.

    Converts encoder hidden states (possibly from multiple layers) into
    decoder-compatible memory for cross-attention.

    Args:
        d_enc: Encoder hidden size.
        d_dec: Decoder hidden size.
        use_layer_fusion: Whether to fuse multiple encoder layers.
        fuse_layers: Layer indices for fusion (e.g., [-1, -4, -8, -12]).
        use_encstack: Whether to use additional encoder layers.
        encstack_layers: Number of EncStack layers.
        encstack_heads: Number of attention heads in EncStack.
        encstack_ffn: FFN size in EncStack.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_enc: int,
        d_dec: int,
        use_layer_fusion: bool = True,
        fuse_layers: Optional[List[int]] = None,
        use_encstack: bool = False,
        encstack_layers: int = 2,
        encstack_heads: int = 12,
        encstack_ffn: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_enc = d_enc
        self.d_dec = d_dec
        self.use_layer_fusion = use_layer_fusion
        self.fuse_layers = fuse_layers or [-1, -4, -8, -12]
        self.use_encstack = use_encstack

        # Layer fusion module
        if use_layer_fusion:
            self.layer_fusion = LayerFusion(num_layers=len(self.fuse_layers))
        else:
            self.layer_fusion = None

        # MLP projection
        self.mlp = AdaptorMLP(d_enc, d_dec, dropout=dropout)

        # Optional encoder stack
        if use_encstack:
            self.encstack = EncoderStack(
                d_model=d_dec,
                num_heads=encstack_heads,
                ffn_size=encstack_ffn,
                num_layers=encstack_layers,
                dropout=dropout,
            )
        else:
            self.encstack = None

    def forward(
        self,
        encoder_output: dict,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Adapt encoder output to decoder memory.

        Args:
            encoder_output: Dict from EncoderWrapper with:
                - "last_hidden_state": [B, S, d_enc]
                - "hidden_states": Tuple of layer outputs (if layer fusion enabled)
            attention_mask: [B, S] — 1 for real tokens, 0 for padding.

        Returns:
            H_mem: [B, S, d_dec] — memory for decoder cross-attention.
        """
        if self.use_layer_fusion and "hidden_states" in encoder_output:
            h = self.layer_fusion(encoder_output["hidden_states"], self.fuse_layers)
        else:
            h = encoder_output["last_hidden_state"]

        # Project d_enc → d_dec
        h = h.to(dtype=self.mlp.linear1.weight.dtype)
        h = self.mlp(h)

        # Optional refinement
        if self.encstack is not None:
            # TransformerEncoder expects src_key_padding_mask where True = ignore
            padding_mask = None
            if attention_mask is not None:
                padding_mask = attention_mask == 0  # True for padded positions
            h = self.encstack(h, src_key_padding_mask=padding_mask)

        return h
