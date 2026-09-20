"""Direct token bridge plus an independently attended compact side memory."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .outputs import BridgeState, EncoderState


class LearnableEvidenceResampler(nn.Module):
    """Compress the final encoder states without altering the base token bank."""

    def __init__(self, hidden_size: int, num_tokens: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("side_tokens must be positive")
        if num_heads <= 0 or hidden_size % num_heads:
            raise ValueError("side_num_heads must divide decoder hidden size")
        self.hidden_size = int(hidden_size)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.dropout = float(dropout)
        self.latents = nn.Parameter(torch.empty(self.num_tokens, self.hidden_size))
        nn.init.normal_(self.latents, mean=0.0, std=0.02)
        self.query_norm = nn.RMSNorm(self.hidden_size)
        self.memory_norm = nn.RMSNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output_norm = nn.RMSNorm(self.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False),
        )

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if memory.ndim != 3 or memory_mask.shape != memory.shape[:2]:
            raise ValueError("resampler memory/mask shapes do not match")
        batch, source_length, _ = memory.shape
        query_states = self.query_norm(self.latents).unsqueeze(0).expand(batch, -1, -1)
        memory_states = self.memory_norm(memory)
        query = self.q_proj(query_states).view(batch, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(memory_states).view(batch, source_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(memory_states).view(batch, source_length, self.num_heads, self.head_dim).transpose(1, 2)
        valid = memory_mask.bool()
        if not bool(valid.any(-1).all()):
            raise ValueError("side-memory resampling requires at least one source token per example")
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, self.num_tokens, self.hidden_size)
        side = query_states + self.o_proj(attended)
        side = side + self.ffn(self.output_norm(side))
        side_mask = torch.ones(batch, self.num_tokens, dtype=torch.bool, device=memory.device)
        return side, side_mask


class GatedSideMemoryBridge(nn.Module):
    """Keep the exact projected token bank and derive a separate side bank."""

    def __init__(self, encoder_hidden: int, decoder_hidden: int, config: dict, gradient_checkpointing: bool = False):
        super().__init__()
        del gradient_checkpointing
        self.bridge_mode = str(config.get("bridge_mode", "side_memory"))
        if self.bridge_mode not in {"side_memory", "direct_projection"}:
            raise ValueError("architecture.bridge_mode must be side_memory or direct_projection")
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        if self.encoder_hidden == self.decoder_hidden:
            self.base_projection: nn.Module = nn.Identity()
        else:
            self.base_projection = nn.Linear(self.encoder_hidden, self.decoder_hidden, bias=False)
            nn.init.orthogonal_(self.base_projection.weight)
        self.resampler: LearnableEvidenceResampler | None = None
        if self.bridge_mode == "side_memory":
            # Keep the downstream RNG identical to the direct ablation.
            with torch.random.fork_rng(devices=[]):
                self.resampler = LearnableEvidenceResampler(
                    self.decoder_hidden,
                    int(config.get("side_tokens", 24)),
                    int(config.get("side_num_heads", 8)),
                    float(config.get("side_resampler_dropout", 0.0)),
                )
        self.controller_dim = 1

    def forward(
        self,
        encoder_state: EncoderState,
        prompt_embeddings: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
    ) -> BridgeState:
        del prompt_embeddings, prompt_mask, output_budget
        final = encoder_state.final
        if final.ndim != 3:
            raise ValueError("encoder_state.final must be [batch, source_tokens, hidden]")
        if encoder_state.attention_mask.shape != final.shape[:2] or encoder_state.content_mask.shape != final.shape[:2]:
            raise ValueError("encoder masks must match encoder_state.final")
        memory_mask = encoder_state.attention_mask.bool()
        content = encoder_state.content_mask.bool() & memory_mask
        memory = self.base_projection(final.float()).masked_fill(~memory_mask.unsqueeze(-1), 0)
        # Deliberately neutral: salience must not perturb the base-token softmax.
        source_bias = torch.zeros(final.shape[:2], device=final.device, dtype=torch.float32)
        controller = torch.zeros(final.shape[0], 1, device=final.device, dtype=torch.float32)
        side_memory = None
        side_memory_mask = None
        if self.resampler is not None:
            side_memory, side_memory_mask = self.resampler(memory, content)
        return BridgeState(
            memory=memory,
            memory_mask=memory_mask,
            content_mask=content,
            source_bias=source_bias,
            controller=controller,
            value_memory=None,
            side_memory=side_memory,
            side_memory_mask=side_memory_mask,
        )
