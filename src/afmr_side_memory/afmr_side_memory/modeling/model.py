"""End-to-end summarizer with a gated side-memory bridge."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .decoder import QwenCrossDecoder
from .encoder import build_encoder, resolve_dtype
from .outputs import AFMROutput, BridgeState
from .side_memory import GatedSideMemoryBridge


class SideMemorySummarizer(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = build_encoder(config)
        model_config = config["model"]
        self.decoder = QwenCrossDecoder(
            str(model_config["decoder_name"]),
            config["decoder"],
            resolve_dtype(str(model_config.get("dtype", "float32"))),
            bool(model_config.get("gradient_checkpointing", True)),
            bool(model_config.get("trust_remote_code", True)),
            str(model_config.get("attention_implementation", "sdpa")),
            side_memory_enabled=config["architecture"].get("bridge_mode", "side_memory") == "side_memory",
        )
        self.bridge = GatedSideMemoryBridge(
            self.encoder.hidden_size,
            int(self.decoder.config.hidden_size),
            config["architecture"],
            gradient_checkpointing=bool(model_config.get("gradient_checkpointing", True)),
        )

    def encode_source(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        source_content_mask: torch.Tensor,
        decoder_prompt_ids: torch.Tensor,
        decoder_prompt_mask: torch.Tensor,
        output_budget: torch.Tensor,
        **copy_inputs: torch.Tensor,
    ) -> BridgeState:
        state = self.encoder(input_ids, attention_mask, source_content_mask)
        prompt_embeddings = self.decoder.embed_tokens(decoder_prompt_ids)
        bridge = self.bridge(state, prompt_embeddings, decoder_prompt_mask, output_budget)
        decoder_dtype = self.decoder.embed_tokens.weight.dtype
        if bridge.memory.dtype != decoder_dtype:
            bridge.memory = bridge.memory.to(decoder_dtype)
        if bridge.side_memory is not None and bridge.side_memory.dtype != decoder_dtype:
            bridge.side_memory = bridge.side_memory.to(decoder_dtype)
        if self.decoder.grounded_copy is not None:
            if not copy_inputs:
                raise ValueError("Grounded copy is enabled but source-token alignment is missing")
            # Side tokens never enter grounded-copy alignment or normalization.
            bridge.copy_state = self.decoder.grounded_copy.prepare(
                bridge.memory,
                bridge.source_bias,
                bridge.content_mask,
                self.decoder.embed_tokens,
                **copy_inputs,
            )
        elif copy_inputs:
            raise ValueError("Copy alignment supplied to a decoder without grounded copy")
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
        **copy_inputs: torch.Tensor,
    ) -> AFMROutput:
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
            **copy_inputs,
        )
        logits, _, loss_ce = self.decoder(
            decoder_input_ids,
            bridge.memory,
            bridge.memory_mask,
            bridge.source_bias,
            decoder_attention_mask,
            labels=labels,
            return_logits=return_logits,
            copy_state=bridge.copy_state,
            side_memory=bridge.side_memory,
            side_memory_mask=bridge.side_memory_mask,
        )
        # The complete training objective is token-level cross entropy only.
        return AFMROutput(logits, loss_ce, loss_ce, bridge, None)
