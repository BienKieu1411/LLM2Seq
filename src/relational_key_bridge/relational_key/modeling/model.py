"""End-to-end RelationalKey model graph."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .bridge import RelationalKeyBridge
from .decoder import QwenCrossDecoder
from .encoder import build_encoder, resolve_dtype
from .outputs import BridgeState, RelationalKeyOutput


class RelationalKeyModel(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = build_encoder(config)
        decoder_cfg = config["model"]
        self.decoder = QwenCrossDecoder(
            str(decoder_cfg["decoder_name"]),
            config["decoder"],
            resolve_dtype(str(decoder_cfg.get("dtype", "float32"))),
            bool(decoder_cfg.get("gradient_checkpointing", True)),
            bool(decoder_cfg.get("trust_remote_code", True)),
            str(decoder_cfg.get("attention_implementation", "sdpa")),
        )
        self.bridge = RelationalKeyBridge(
            self.encoder.hidden_size, int(self.decoder.config.hidden_size), config["architecture"]
        )

    def encode_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_content_mask: torch.Tensor,
        decoder_prompt_ids: torch.Tensor,
        decoder_prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
        **alignment_inputs: torch.Tensor,
    ) -> BridgeState:
        state = self.encoder(input_ids, attention_mask, source_content_mask)
        bridge = self.bridge(state, self.decoder.embed_tokens, **alignment_inputs)
        decoder_dtype = self.decoder.embed_tokens.weight.dtype
        if bridge.memory.dtype != decoder_dtype:
            bridge = bridge.to_dtype(decoder_dtype)
        if self.decoder.grounded_copy is not None:
            if not alignment_inputs:
                raise ValueError("Grounded copy is enabled but source-token alignment is missing")
            bridge.copy_state = self.decoder.grounded_copy.prepare(
                bridge.copy_memory,
                bridge.content_mask,
                self.decoder.embed_tokens,
                **{key: value for key, value in alignment_inputs.items() if key != "copy_token_positions"},
            )
        return bridge

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_content_mask: torch.Tensor,
        decoder_prompt_ids: torch.Tensor,
        decoder_prompt_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_budget: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        **alignment_inputs: torch.Tensor,
    ) -> RelationalKeyOutput:
        if output_budget is None:
            output_budget = torch.full(
                (input_ids.shape[0],),
                int(self.config["generation"].get("max_new_tokens", 256)),
                device=input_ids.device,
                dtype=torch.float32,
            )
        bridge = self.encode_source(
            input_ids,
            attention_mask,
            source_content_mask,
            decoder_prompt_ids,
            decoder_prompt_mask,
            output_budget,
            **alignment_inputs,
        )
        logits, _, loss_ce = self.decoder(
            input_ids=decoder_input_ids,
            memory=bridge.memory,
            memory_mask=bridge.memory_mask,
            attention_mask=decoder_attention_mask,
            labels=labels,
            return_logits=return_logits,
            value_memory=bridge.value_memory,
            copy_state=bridge.copy_state,
        )
        return RelationalKeyOutput(logits, loss_ce, loss_ce, bridge)
