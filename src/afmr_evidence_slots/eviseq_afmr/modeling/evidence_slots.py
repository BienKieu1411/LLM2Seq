"""Controller-conditioned latent evidence slots for key-only bridge adaptation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inverse_bounded_init(value: float, maximum: float) -> float:
    if not 0.0 < float(value) < float(maximum):
        raise ValueError("Bounded gates require 0 < init < maximum")
    return math.log(float(value) / (float(maximum) - float(value)))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)


class SlotRefinement(nn.Module):
    """One pre-norm slot self-attention and feed-forward refinement block."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention_norm = nn.RMSNorm(hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.0, batch_first=True)
        self.mlp_norm = nn.RMSNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size, bias=False),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(slots)
        mixed, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        slots = slots + mixed
        return slots + self.mlp(self.mlp_norm(slots))


class EvidenceSlotBridge(nn.Module):
    """Compress global source evidence into slots, then write it back to key tokens.

    The returned residual is token aligned but is intended only for the decoder's
    key memory. Values and grounded-copy state remain anchored to the unmodified
    final encoder representation in the enclosing AFMR bridge.
    """

    def __init__(
        self,
        source_hidden: int,
        decoder_hidden: int,
        controller_dim: int,
        config: dict,
    ):
        super().__init__()
        self.num_slots = int(config["num_slots"])
        self.hidden_size = int(config["dim"])
        self.num_heads = int(config["num_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.rank = int(config["rank"])
        self.gate_max = float(config["gate_max"])
        self.prior_enabled = bool(config["prior_enabled"])
        self.prior_max = float(config["prior_max"])

        self.slot_embeddings = nn.Parameter(torch.empty(self.num_slots, self.hidden_size))
        nn.init.orthogonal_(self.slot_embeddings)
        # Orthogonal rows have unit L2 norm by default (RMS 1/sqrt(D)); scale
        # them to unit RMS so the shared controller condition cannot erase
        # slot identity at initialization.
        with torch.no_grad():
            self.slot_embeddings.mul_(math.sqrt(self.hidden_size))
        self.controller_norm = nn.RMSNorm(controller_dim)
        self.controller_to_slots = nn.Linear(controller_dim, self.hidden_size, bias=False)
        nn.init.zeros_(self.controller_to_slots.weight)

        self.source_norm = nn.RMSNorm(source_hidden)
        self.read_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.read_key = nn.Linear(source_hidden, self.hidden_size, bias=False)
        self.read_value = nn.Linear(source_hidden, self.hidden_size, bias=False)
        self.read_output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.refinement = nn.ModuleList(
            SlotRefinement(self.hidden_size, self.num_heads) for _ in range(int(config["refine_layers"]))
        )
        self.slot_norm = nn.RMSNorm(self.hidden_size)
        self.topdown_query = nn.Linear(source_hidden, self.hidden_size, bias=False)
        self.topdown_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.topdown_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.topdown_output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.key_norm = nn.RMSNorm(self.hidden_size)
        self.key_down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.key_up = nn.Linear(self.rank, decoder_hidden, bias=False)
        # A strictly zero output would postpone all read/write/slot gradients to
        # the second optimizer step. This small initialization preserves an
        # almost-base start while making the complete route train on step one.
        nn.init.normal_(self.key_up.weight, mean=0.0, std=1.0e-3 / math.sqrt(self.rank))
        self.gate_raw = nn.Linear(controller_dim, 1)
        nn.init.zeros_(self.gate_raw.weight)
        nn.init.constant_(self.gate_raw.bias, _inverse_bounded_init(config["gate_init"], self.gate_max))

        if self.prior_enabled:
            self.prior_source = nn.Linear(source_hidden, self.hidden_size, bias=False)
            self.prior_context = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.prior_output = nn.Linear(self.hidden_size, 1, bias=False)
            nn.init.normal_(self.prior_output.weight, mean=0.0, std=1.0e-3 / math.sqrt(self.hidden_size))
            self.prior_raw = nn.Linear(controller_dim, 1)
            nn.init.zeros_(self.prior_raw.weight)
            nn.init.constant_(
                self.prior_raw.bias,
                _inverse_bounded_init(config["prior_init"], self.prior_max),
            )

    def _split_heads(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _competitive_read_weights(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Assign each token across slots, then normalize each slot over tokens.

        Competition on the slot axis prevents every latent slot from becoming
        an independent softmax that can attend to the same source region.
        """
        if mask.ndim != 2 or mask.shape[0] != logits.shape[0] or mask.shape[1] != logits.shape[-1]:
            raise ValueError("attention mask must be [batch, key_tokens]")
        if logits.shape[-1] == 0:
            raise ValueError("evidence-slot attention requires at least one source token")
        valid = mask.bool()
        responsibilities = torch.softmax(logits.float(), dim=-2).masked_fill(~valid[:, None, None, :], 0.0)
        denominator = responsibilities.sum(dim=-1, keepdim=True)
        return responsibilities / denominator.clamp_min(1.0e-6)

    def _read_source(self, source: torch.Tensor, content_mask: torch.Tensor, controller: torch.Tensor) -> torch.Tensor:
        batch = source.shape[0]
        conditioned = (
            self.slot_embeddings[None, :, :]
            + self.controller_to_slots(self.controller_norm(controller.float()))[:, None, :]
        )
        query = self._split_heads(self.read_query(conditioned))
        key = self._split_heads(self.read_key(source))
        value = self._split_heads(self.read_value(source))
        logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        weights = self._competitive_read_weights(logits, content_mask)
        read = (
            torch.matmul(weights.to(value.dtype), value)
            .transpose(1, 2)
            .reshape(batch, self.num_slots, self.hidden_size)
        )
        slots = conditioned + self.read_output(read)
        for layer in self.refinement:
            slots = layer(slots)
        return self.slot_norm(slots)

    def _write_tokens(
        self,
        source: torch.Tensor,
        slots: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, source_length, _ = source.shape
        query = self._split_heads(self.topdown_query(source))
        key = self._split_heads(self.topdown_key(slots))
        value = self._split_heads(self.topdown_value(slots))
        logits = torch.matmul(query.float(), key.float().transpose(-1, -2)) / math.sqrt(self.head_dim)
        routing = torch.softmax(logits, dim=-1)
        written = (
            torch.matmul(routing.to(value.dtype), value).transpose(1, 2).reshape(batch, source_length, self.hidden_size)
        )
        written = self.topdown_output(written).masked_fill(~content_mask[..., None], 0.0)
        return written, routing

    def _routing_prior(
        self,
        normalized_source: torch.Tensor,
        written: torch.Tensor,
        content_mask: torch.Tensor,
        controller: torch.Tensor,
    ) -> torch.Tensor:
        if not self.prior_enabled:
            return written.new_zeros(content_mask.shape, dtype=torch.float32)
        # The slot-written context carries the token-to-slot route. A learned
        # signed compatibility score with H0 is safer than treating routing
        # confidence itself as salience.
        score = (
            self.prior_output(F.silu(self.prior_source(normalized_source) + self.prior_context(written)))
            .squeeze(-1)
            .float()
        )
        centered = score - _masked_mean(score, content_mask)[:, None]
        variance = _masked_mean(centered.square(), content_mask)
        normalized = centered / variance.add(1.0e-6).sqrt()[:, None]
        shaped = torch.tanh(normalized).masked_fill(~content_mask, 0.0)
        shaped = (shaped - _masked_mean(shaped, content_mask)[:, None]).masked_fill(~content_mask, 0.0)
        max_abs = shaped.abs().masked_fill(~content_mask, 0.0).amax(dim=-1, keepdim=True).clamp_min(1.0)
        shaped = shaped / max_abs
        strength = self.prior_max * torch.sigmoid(self.prior_raw(controller.float()).squeeze(-1))
        return (strength[:, None] * shaped).masked_fill(~content_mask, 0.0)

    def forward(
        self,
        source: torch.Tensor,
        content_mask: torch.Tensor,
        controller: torch.Tensor,
        key_anchor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source.ndim != 3 or key_anchor.ndim != 3 or source.shape[:2] != key_anchor.shape[:2]:
            raise ValueError("source and key_anchor must be token-aligned [batch, source_tokens, hidden]")
        if content_mask.shape != source.shape[:2]:
            raise ValueError("content_mask must match source token axes")
        content = content_mask.bool()
        normalized_source = self.source_norm(source.float())
        slots = self._read_source(normalized_source, content, controller)
        written, _ = self._write_tokens(normalized_source, slots, content)
        delta = self.key_up(F.silu(self.key_down(self.key_norm(written)))).float()

        # Smooth relative-RMS cap. The actual residual RMS is at most the
        # controller gate times the key-anchor RMS for every source token.
        base = key_anchor.float()
        base_rms = base.detach().square().mean(dim=-1, keepdim=True).sqrt()
        delta_rms = delta.square().mean(dim=-1, keepdim=True).add(1.0e-12).sqrt()
        scale = base_rms / (base_rms.square() + delta_rms.square() + 1.0e-12).sqrt()
        gate = self.gate_max * torch.sigmoid(self.gate_raw(controller.float()))
        residual = (gate[:, None, :] * scale * delta).masked_fill(~content[..., None], 0.0)
        prior = self._routing_prior(normalized_source, written, content, controller)
        return residual.to(key_anchor.dtype), prior
