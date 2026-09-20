"""Source-conditioned bridge used by the source-film experiment."""

from __future__ import annotations

import torch
import torch.nn as nn

from .controller import FocusController
from .outputs import BridgeState, EncoderState


class SourceConditionedBridge(nn.Module):
    """Project source tokens and return a document-conditioned controller.

    ``direct_projection`` is the controlled ablation.  Both paths expose the
    same projected memory to cross-attention and grounded copy; the full path
    additionally supplies a controller consumed by decoder adaptive norms.
    """

    def __init__(self, encoder_hidden: int, decoder_hidden: int, config: dict, gradient_checkpointing: bool = False):
        super().__init__()
        self.bridge_mode = str(config.get("bridge_mode", "source_film"))
        if self.bridge_mode not in {"source_film", "direct_projection"}:
            raise ValueError("architecture.bridge_mode must be source_film or direct_projection")
        self.encoder_hidden = int(encoder_hidden)
        self.decoder_hidden = int(decoder_hidden)
        self.controller_dim = int(config.get("controller_dim", 256))
        if self.controller_dim <= 0:
            raise ValueError("architecture.controller_dim must be positive")
        if self.encoder_hidden == self.decoder_hidden:
            self.base_projection: nn.Module = nn.Identity()
        else:
            self.base_projection = nn.Linear(self.encoder_hidden, self.decoder_hidden, bias=False)
            nn.init.orthogonal_(self.base_projection.weight)
        self.controller: nn.Module | None = None
        if self.bridge_mode == "source_film":
            self.controller = FocusController(self.encoder_hidden, self.decoder_hidden, self.controller_dim)

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
        memory_mask = encoder_state.attention_mask.bool()
        content = encoder_state.content_mask.bool() & memory_mask
        memory = self.base_projection(final.float()).masked_fill(~memory_mask.unsqueeze(-1), 0)
        source_bias = torch.zeros(final.shape[:2], device=final.device, dtype=torch.float32).masked_fill(~content, 0.0)
        if self.controller is None:
            controller = torch.zeros(final.shape[0], self.controller_dim, device=final.device, dtype=torch.float32)
        else:
            controller = self.controller(final, content, prompt_embeddings, prompt_mask.bool(), output_budget)
        return BridgeState(memory, memory_mask, content, source_bias, controller, None)


# Compatibility alias used by shared runtime/checkpoint code.
AdaptiveFullMemoryResidualBridge = SourceConditionedBridge
