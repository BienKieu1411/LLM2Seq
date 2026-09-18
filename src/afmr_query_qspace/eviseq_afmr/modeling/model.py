"""End-to-end AFMR model graph."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import contextual_value_settings, region_query_settings
from .afmr import AdaptiveFullMemoryResidualBridge
from .decoder import QwenCrossDecoder
from .encoder import build_encoder, resolve_dtype
from .outputs import AFMROutput, BridgeState
from .region_query import pool_regions


def salience_valid_rows(
    source_content_mask: torch.Tensor,
    salience_labels: torch.Tensor,
    salience_mask: torch.Tensor,
) -> torch.Tensor:
    """Return rows that contribute to the evidence objective.

    The collator normally precomputes ``salience_mask``.  Rechecking the
    positive/negative support here keeps the loss denominator and the trainer
    accumulation denominator identical even if a hand-built batch supplies an
    inconsistent mask.
    """

    content = source_content_mask.bool()
    positive = salience_labels.gt(0) & content
    negative = (~positive) & content
    return salience_mask.bool() & positive.any(-1) & negative.any(-1)


class EviSeqAFMR(nn.Module):
    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder = build_encoder(config)
        decoder_cfg = config["model"]
        context = contextual_value_settings(config["architecture"])
        region_query = region_query_settings(config["architecture"])
        self.region_query_config = region_query
        self.decoder = QwenCrossDecoder(
            str(decoder_cfg["decoder_name"]),
            config["decoder"],
            resolve_dtype(str(decoder_cfg.get("dtype", "float32"))),
            bool(decoder_cfg.get("gradient_checkpointing", True)),
            bool(decoder_cfg.get("trust_remote_code", True)),
            str(decoder_cfg.get("attention_implementation", "sdpa")),
            value_residual_max_relative_rms=context["max_relative_rms"] if context["enabled"] else 0.0,
            region_query_config=region_query,
        )
        self.bridge = AdaptiveFullMemoryResidualBridge(
            self.encoder.hidden_size,
            int(self.decoder.config.hidden_size),
            config["architecture"],
            gradient_checkpointing=bool(decoder_cfg.get("gradient_checkpointing", True)),
        )
        self.salience_loss_weight = float(config["training"].get("salience_loss_weight", 0.0))
        self.salience_margin = float(config["training"].get("salience_margin", 0.5))

    @staticmethod
    def _salience_ranking_loss(
        source_bias: torch.Tensor,
        source_content_mask: torch.Tensor,
        salience_labels: torch.Tensor,
        salience_mask: torch.Tensor,
        labels: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """Token-weighted positive-vs-negative ranking on the predicted source prior.

        The reference-derived labels only supervise this loss. They never enter
        source encoding or decoding, so inference uses the same predicted prior.
        """
        content = source_content_mask.bool()
        positive = (salience_labels > 0) & content
        negative = (~positive) & content
        valid = salience_valid_rows(content, salience_labels, salience_mask)
        scores = source_bias.float()
        positive_weights = salience_labels.float().clamp_min(0) * content
        pos_mean = (scores * positive_weights).sum(-1) / positive_weights.sum(-1).clamp_min(1)
        neg_mean = (scores * negative).sum(-1) / negative.sum(-1).clamp_min(1)
        row_loss = F.softplus(float(margin) - pos_mean + neg_mean)
        # The trainer weights microbatches by their number of target tokens.
        # The same weighting here makes accumulation and DDP globally exact.
        token_counts = labels[:, 1:].ne(-100).sum(-1).float()
        # Invalid rows must not dilute the auxiliary objective.  The trainer
        # uses the same valid-token count to combine microbatches and DDP
        # ranks, so this is a true mean over supervised target tokens.
        valid_token_counts = token_counts * valid
        return (row_loss * valid_token_counts).sum() / valid_token_counts.sum().clamp_min(1)

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
            bridge = BridgeState(
                bridge.memory.to(decoder_dtype),
                bridge.memory_mask,
                bridge.content_mask,
                bridge.source_bias,
                bridge.controller,
                None if bridge.value_memory is None else bridge.value_memory.to(decoder_dtype),
                value_residual=None if bridge.value_residual is None else bridge.value_residual.to(decoder_dtype),
            )
        if self.region_query_config["enabled"]:
            # In direct_projection, bridge.memory itself is projected H0.
            # This makes direct+region a controlled fourth ablation arm.
            region_values = bridge.value_memory if bridge.value_memory is not None else bridge.memory
            bridge.region_keys, bridge.region_values, bridge.region_mask = pool_regions(
                bridge.memory,
                region_values,
                bridge.content_mask,
                int(self.region_query_config["window_size"]),
                int(self.region_query_config["stride"]),
            )
        if self.decoder.grounded_copy is not None:
            if not copy_inputs:
                raise ValueError("Grounded copy is enabled but source-token alignment is missing")
            bridge.copy_state = self.decoder.grounded_copy.prepare(
                bridge.value_memory if bridge.value_memory is not None else bridge.memory,
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
        source_salience_labels: Optional[torch.Tensor] = None,
        source_salience_mask: Optional[torch.Tensor] = None,
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
            value_memory=bridge.value_memory,
            copy_state=bridge.copy_state,
            value_residual=bridge.value_residual,
            region_keys=bridge.region_keys,
            region_values=bridge.region_values,
            region_mask=bridge.region_mask,
        )
        loss_salience = None
        total_loss = loss_ce
        if labels is not None and self.salience_loss_weight > 0 and self.bridge.bridge_mode == "afmr":
            if source_salience_labels is None or source_salience_mask is None:
                raise ValueError("Salience supervision is enabled but source salience labels are missing")
            loss_salience = self._salience_ranking_loss(
                bridge.source_bias,
                bridge.content_mask,
                source_salience_labels,
                source_salience_mask,
                labels,
                self.salience_margin,
            )
            total_loss = loss_ce + self.salience_loss_weight * loss_salience
        return AFMROutput(logits, loss_ce, total_loss, bridge, loss_salience)
