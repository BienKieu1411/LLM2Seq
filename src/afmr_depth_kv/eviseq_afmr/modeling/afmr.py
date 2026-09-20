"""Layer-wise coupled-depth key/value bridge.

The direct final-state projection is always the source-memory anchor. In the
full bridge, every decoder layer receives its own token-aligned memory. A
document-conditioned convex router mixes aligned encoder depth taps and a
zero-initialized low-rank output adds a smoothly bounded residual to the
anchor. The same resulting tensor supplies both keys and values.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .controller import FocusController
from .outputs import BridgeState, EncoderState


def _inverse_bounded_init(value: float, maximum: float) -> float:
    if not 0.0 < float(value) < float(maximum):
        raise ValueError("depth gate requires 0 < init < maximum")
    return math.log(float(value) / (float(maximum) - float(value)))


class LayerwiseCoupledDepthBridge(nn.Module):
    """Build one aligned K/V memory per decoder layer.

    ``direct_projection`` returns only the final-state anchor. ``depth_kv``
    additionally returns ``layer_memories`` while leaving ``memory`` untouched
    for grounded copy and for the controlled ablation.
    """

    def __init__(
        self,
        encoder_hidden: int,
        decoder_hidden: int,
        decoder_layers: int,
        config: dict,
    ) -> None:
        super().__init__()
        self.bridge_mode = str(config.get("bridge_mode", "depth_kv"))
        if self.bridge_mode not in {"depth_kv", "direct_projection"}:
            raise ValueError("architecture.bridge_mode must be depth_kv or direct_projection")
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        self.decoder_layers = int(decoder_layers)
        self.controller_dim = int(config.get("controller_dim", 256))
        self.depth_taps = int(config.get("depth_taps", 4))
        self.depth_rank = int(config.get("depth_rank", 128))
        self.depth_gate_max = float(config.get("depth_gate_max", 1.0))
        self.residual_max_relative_rms = float(config.get("residual_max_relative_rms", 0.25))

        if self.decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        if self.depth_taps <= 1:
            raise ValueError("layer-wise depth routing requires depth_taps > 1")
        if self.depth_rank <= 0 or self.controller_dim <= 0:
            raise ValueError("controller_dim and depth_rank must be positive")
        if not 0.0 < self.residual_max_relative_rms <= 1.0:
            raise ValueError("residual_max_relative_rms must lie in (0, 1]")

        # This module exists in both modes and is constructed first. Therefore
        # equal seeds give the full bridge and its direct ablation the same
        # final-state projection.
        if self.encoder_hidden == self.decoder_hidden:
            self.base_projection: nn.Module = nn.Identity()
        else:
            self.base_projection = nn.Linear(self.encoder_hidden, self.decoder_hidden, bias=False)
            nn.init.orthogonal_(self.base_projection.weight)
        shared_rng_state = torch.get_rng_state()

        if self.bridge_mode == "direct_projection":
            return

        self.controller = FocusController(self.encoder_hidden, self.decoder_hidden, self.controller_dim)
        self.depth_norms = nn.ModuleList(nn.RMSNorm(self.encoder_hidden) for _ in range(self.depth_taps))

        # Each decoder layer has a distinct static preference, plus a
        # document/prompt/budget-conditioned correction. Softmax makes every
        # route a convex combination over encoder depths.
        self.depth_route_bias = nn.Parameter(torch.zeros(self.decoder_layers, self.depth_taps))
        anchor_bias = float(config.get("final_tap_bias", 1.5))
        with torch.no_grad():
            self.depth_route_bias[:, -1].fill_(anchor_bias)
        self.depth_route_condition = nn.Linear(self.controller_dim, self.decoder_layers * self.depth_taps, bias=False)
        nn.init.zeros_(self.depth_route_condition.weight)

        self.depth_down = nn.Linear(self.encoder_hidden, self.depth_rank, bias=False)
        self.depth_outputs = nn.ModuleList(
            nn.Linear(self.depth_rank, self.decoder_hidden, bias=False) for _ in range(self.decoder_layers)
        )
        # Exact direct-projection parity at initialization. CE gradients reach
        # every output on the first backward pass; upstream routing starts
        # receiving gradients after those outputs move away from zero.
        for output in self.depth_outputs:
            nn.init.zeros_(output.weight)

        gate_raw = _inverse_bounded_init(
            float(config.get("depth_gate_init", 0.10)),
            self.depth_gate_max,
        )
        self.depth_gate = nn.Linear(self.controller_dim, self.decoder_layers)
        nn.init.zeros_(self.depth_gate.weight)
        nn.init.constant_(self.depth_gate.bias, gate_raw)
        # Preserve downstream sampler/dropout RNG parity with the same-seed
        # direct-projection control without tying the per-layer parameters.
        torch.set_rng_state(shared_rng_state)

    def _route_weights(self, controller: torch.Tensor) -> torch.Tensor:
        conditioned = self.depth_route_condition(controller.float()).view(
            controller.shape[0], self.decoder_layers, self.depth_taps
        )
        return torch.softmax(conditioned + self.depth_route_bias[None], dim=-1)

    def _bounded_residual(self, anchor: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Smoothly cap each token residual relative to its anchor RMS."""
        anchor_rms = anchor.detach().float().square().mean(-1, keepdim=True).sqrt()
        radius = self.residual_max_relative_rms * anchor_rms
        delta_float = delta.float()
        delta_energy = delta_float.square().mean(-1, keepdim=True)
        scale = radius / torch.sqrt(radius.square() + delta_energy + 1.0e-12)
        return (delta_float * scale).to(delta.dtype)

    def forward(
        self,
        encoder_state: EncoderState,
        prompt_embeddings: torch.Tensor,
        prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
    ) -> BridgeState:
        final = encoder_state.final
        if final.ndim != 3:
            raise ValueError("encoder_state.final must be [batch, source_tokens, hidden]")
        if encoder_state.attention_mask.shape != final.shape[:2] or encoder_state.content_mask.shape != final.shape[:2]:
            raise ValueError("encoder masks must match encoder_state.final")
        if prompt_embeddings.shape[:2] != prompt_mask.shape:
            raise ValueError("prompt mask must match prompt embeddings")
        if len(encoder_state.taps) != self.depth_taps:
            raise ValueError(f"expected {self.depth_taps} encoder taps, got {len(encoder_state.taps)}")

        memory_mask = encoder_state.attention_mask.bool()
        content = encoder_state.content_mask.bool() & memory_mask
        anchor = self.base_projection(final.float()).masked_fill(~memory_mask.unsqueeze(-1), 0.0)
        source_bias = torch.zeros(final.shape[:2], device=final.device, dtype=torch.float32)

        if self.bridge_mode == "direct_projection":
            controller = torch.zeros(final.shape[0], self.controller_dim, device=final.device, dtype=torch.float32)
            return BridgeState(anchor, memory_mask, content, source_bias, controller)

        controller = self.controller(final, content, prompt_embeddings, prompt_mask.bool(), output_budget)
        normalized_taps = torch.stack(
            [norm(tap.float()) for norm, tap in zip(self.depth_norms, encoder_state.taps)], dim=2
        )
        weights = self._route_weights(controller)
        final_normalized = normalized_taps[:, :, -1]
        gates = self.depth_gate_max * torch.sigmoid(self.depth_gate(controller.float()))
        layer_memories = []
        for layer_index, output in enumerate(self.depth_outputs):
            mixed = torch.sum(
                normalized_taps * weights[:, None, layer_index, :, None],
                dim=2,
            )
            features = F.silu(self.depth_down(mixed - final_normalized))
            delta = output(features)
            bounded = self._bounded_residual(anchor, delta)
            layer_memory = anchor + gates[:, layer_index, None, None].to(anchor.dtype) * bounded
            layer_memories.append(layer_memory.masked_fill(~memory_mask.unsqueeze(-1), 0.0))

        return BridgeState(
            anchor,
            memory_mask,
            content,
            source_bias,
            controller,
            layer_memories=tuple(layer_memories),
        )


# Keep the historical import name internal callers used, without retaining the
# historical AFMR mechanism.
AdaptiveFullMemoryResidualBridge = LayerwiseCoupledDepthBridge
