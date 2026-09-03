"""One encoder, one evidence interface, and one pretrained causal decoder."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.objectives import (
    EvidenceContrastiveHead,
    PromptConditionedEvidenceHead,
    SourcePromptAlignmentHead,
    decoder_sentence_representations,
    decoder_summary_representation,
    evidence_info_nce_loss,
    exact_duplicate_mask,
    info_nce_loss,
    last_prompt_states,
    masked_mean_pool,
    hard_negative_indices,
    pairwise_geometry_preservation_loss,
    per_example_nll,
    sentence_evidence_info_nce_loss,
    source_memory_for_mining,
    source_swap_contrastive_loss,
)
from .attention import pool_units
from .bridge import BridgeOutput, EvidenceBridge, balanced_salience_loss
from .decoder import PretrainedQwenDecoder
from .encoder import build_encoder


def torch_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}")
    return values[name]


class EviSeq(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config["model"]
        dtype = torch_dtype(str(model_config.get("dtype", "float32")))
        checkpointing = bool(model_config.get("gradient_checkpointing", True))
        self.encoder = build_encoder(config, dtype)
        expected_encoder = int(model_config["encoder_hidden_size"])
        if self.encoder.hidden_size != expected_encoder:
            raise ValueError(
                f"Configured encoder hidden size {expected_encoder} != checkpoint {self.encoder.hidden_size}"
            )
        decoder_hidden = int(model_config["decoder_hidden_size"])
        self.adapter = EvidenceBridge(expected_encoder, decoder_hidden, config["bridge"])
        self.decoder = PretrainedQwenDecoder(str(model_config["decoder_name"]), config["decoder"], dtype, checkpointing)
        if int(self.decoder.config.hidden_size) != decoder_hidden:
            raise ValueError("Configured decoder hidden size does not match checkpoint")

        objectives = config.get("objectives", {})

        self.salience_weight = float(objectives.get("salience_weight", 0.1))
        self.bridge_geometry_weight = float(objectives.get("bridge_geometry_weight", 0.0))
        self.bridge_geometry_max_units = int(objectives.get("bridge_geometry_max_units", 12))
        if self.bridge_geometry_weight < 0.0:
            raise ValueError("bridge_geometry_weight must be non-negative")
        if self.bridge_geometry_max_units <= 1:
            raise ValueError("bridge_geometry_max_units must be greater than one")

        self.use_contrastive = bool(objectives.get("use_contrastive", False))
        self.contrastive_weight = float(objectives.get("contrastive_weight", 0.0)) if self.use_contrastive else 0.0
        self.contrastive_temperature = float(objectives.get("contrastive_temperature", 0.07))
        self.contrastive_warmup_epochs = int(objectives.get("contrastive_warmup_epochs", 2))
        self.contrastive_across_accumulation = bool(objectives.get("contrastive_across_accumulation", False))
        self.use_source_swap = self.use_contrastive and bool(objectives.get("use_source_swap", False))
        self.source_swap_weight = float(objectives.get("source_swap_weight", 0.1)) if self.use_source_swap else 0.0
        self.source_swap_margin = float(objectives.get("source_swap_margin", 0.2))
        self.source_swap_temperature = float(objectives.get("source_swap_temperature", 1.0))
        self.source_swap_strategy = str(objectives.get("source_swap_strategy", "hard_in_batch"))
        self.contrastive_pooling = str(objectives.get("contrastive_pooling", "mean_last"))
        self.label_smoothing = float(objectives.get("label_smoothing", 0.0))
        if self.use_contrastive:
            self.alignment_head: Optional[SourcePromptAlignmentHead] = SourcePromptAlignmentHead(
                decoder_hidden,
                projection_size=int(objectives.get("contrastive_projection_size", 256)),
                pooling=str(objectives.get("contrastive_pooling", "mean_last")),
            )
        else:
            self.alignment_head = None

        self.use_evidence_contrastive = bool(objectives.get("use_evidence_contrastive", True))
        self.evidence_contrastive_weight = (
            float(objectives.get("evidence_contrastive_weight", 0.05)) if self.use_evidence_contrastive else 0.0
        )
        self.evidence_contrastive_temperature = float(objectives.get("evidence_contrastive_temperature", 0.07))
        self.evidence_hard_negatives_warmup = int(objectives.get("evidence_hard_negatives", 4))
        self.evidence_hard_negatives_full = int(
            objectives.get("evidence_hard_negatives_full", self.evidence_hard_negatives_warmup)
        )
        self.evidence_hard_negatives = self.evidence_hard_negatives_warmup
        self.evidence_hard_negative_salience_boost = float(objectives.get("evidence_hard_negative_salience_boost", 0.1))
        self.evidence_hard_negative_attention_boost = float(
            objectives.get("evidence_hard_negative_attention_boost", 0.0)
        )
        self.evidence_contrastive_warmup_epochs = int(objectives.get("evidence_contrastive_warmup_epochs", 2))
        self.evidence_contrastive_mode = str(objectives.get("evidence_contrastive_mode", "document"))
        self.evidence_contrastive_salience_bias = float(objectives.get("evidence_contrastive_salience_bias", 0.0))
        self.evidence_contrastive_attention_aligned = bool(
            objectives.get("evidence_contrastive_attention_aligned", False)
        )
        self.prompt_conditioned_inference_bridge = bool(objectives.get("prompt_conditioned_inference_bridge", False))
        self.prompt_bridge_dynamic_salience_mix = float(objectives.get("prompt_bridge_dynamic_salience_mix", 0.5))
        self.prompt_bridge_dynamic_logit_scale = float(objectives.get("prompt_bridge_dynamic_logit_scale", 1.0))
        self.prompt_bridge_dynamic_logit_clip = float(objectives.get("prompt_bridge_dynamic_logit_clip", 2.0))
        self.prompt_bridge_source_probe_layers = int(objectives.get("prompt_bridge_source_probe_layers", 2))
        self.prompt_conditioned_static_neutral_probe = bool(
            objectives.get("prompt_conditioned_static_neutral_probe", False)
        )
        if self.evidence_contrastive_mode not in {"document", "sentence_aligned", "prompt_conditioned"}:
            raise ValueError(
                "evidence_contrastive_mode must be 'document', 'sentence_aligned', or 'prompt_conditioned'"
            )
        if self.evidence_contrastive_salience_bias < 0.0:
            raise ValueError("evidence_contrastive_salience_bias must be non-negative")
        if self.evidence_hard_negative_attention_boost < 0.0:
            raise ValueError("evidence_hard_negative_attention_boost must be non-negative")
        if self.prompt_conditioned_inference_bridge and not self.evidence_contrastive_attention_aligned:
            raise ValueError("DualBridge requires attention-aligned evidence contrastive learning")
        if not 0.0 <= self.prompt_bridge_dynamic_salience_mix <= 1.0:
            raise ValueError("prompt_bridge_dynamic_salience_mix must be in [0, 1]")
        if self.prompt_bridge_dynamic_logit_scale <= 0.0:
            raise ValueError("prompt_bridge_dynamic_logit_scale must be positive")
        if not 0.0 < self.prompt_bridge_dynamic_logit_clip <= 5.0:
            raise ValueError("prompt_bridge_dynamic_logit_clip must be in (0, 5]")
        if (self.prompt_conditioned_inference_bridge or self.prompt_conditioned_static_neutral_probe) and not 0 <= (
            self.prompt_bridge_source_probe_layers
        ) <= len(self.decoder.cross_attention_indices):
            raise ValueError("prompt_bridge_source_probe_layers is outside the copied cross-attention depth")
        self.prompt_conditioned_evidence_head: Optional[PromptConditionedEvidenceHead]
        if self.use_evidence_contrastive and self.evidence_contrastive_mode == "prompt_conditioned":
            self.evidence_contrastive_head = None
            self.prompt_conditioned_evidence_head = PromptConditionedEvidenceHead(
                hidden_size=decoder_hidden,
                projection_size=int(objectives.get("evidence_contrastive_projection_size", 256)),
                context_gate_init=float(objectives.get("evidence_prompt_context_gate_init", 0.5)),
            )
            fusion_init = float(objectives.get("evidence_prompt_bridge_fusion_init", 0.20))
            if not 0.0 < fusion_init < 1.0:
                raise ValueError("evidence_prompt_bridge_fusion_init must be in (0, 1)")
            self.prompt_bridge_fusion_logit = nn.Parameter(
                torch.logit(torch.tensor(fusion_init, dtype=torch.float32)),
                requires_grad=(
                    self.prompt_conditioned_inference_bridge or not self.evidence_contrastive_attention_aligned
                ),
            )
        elif self.use_evidence_contrastive:
            self.evidence_contrastive_head: Optional[EvidenceContrastiveHead] = EvidenceContrastiveHead(
                key_hidden_size=decoder_hidden,
                decoder_hidden_size=decoder_hidden,
                projection_size=int(objectives.get("evidence_contrastive_projection_size", 256)),
            )
            self.prompt_conditioned_evidence_head = None
        else:
            self.evidence_contrastive_head = None
            self.prompt_conditioned_evidence_head = None
        if self.prompt_conditioned_evidence_head is None:
            self.register_parameter("prompt_bridge_fusion_logit", None)

        self._contrastive_scale = 1.0
        self._evidence_contrastive_scale = 1.0
        self._stage = "unconfigured"

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    def set_contrastive_scale(self, scale: float) -> None:
        self._contrastive_scale = max(0.0, min(1.0, float(scale)))

    def set_evidence_contrastive_scale(self, scale: float) -> None:
        self._evidence_contrastive_scale = max(0.0, min(1.0, float(scale)))

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        encoded = self.encoder(input_ids, attention_mask, unit_ids=unit_ids)
        return self.adapter(
            encoded.memory,
            attention_mask,
            unit_ids,
            encoded.unit_logits,
            encoded.valid_units,
            evidence_labels,
        )

    def prompt_bridge_fusion_gate(self) -> torch.Tensor:
        """Weight of the prompt-grounded route in the fused evidence prior."""

        if self.prompt_bridge_fusion_logit is None or (
            not self.prompt_conditioned_inference_bridge and self.evidence_contrastive_attention_aligned
        ):
            return next(self.parameters()).new_zeros((), dtype=torch.float32)
        return torch.sigmoid(self.prompt_bridge_fusion_logit.float())

    def _prompt_conditioned_unit_logits(
        self,
        bridge: BridgeOutput,
        prompt_state: torch.Tensor,
        *,
        sentence_reprs: Optional[torch.Tensor] = None,
        valid_units: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse static encoder salience with a prompt-grounded source score.

        ``prompt_state`` is the state immediately before the first generated
        summary token.  It is causal and therefore contains no reference or
        future generated token.  The method is shared by teacher-forced
        training and greedy generation so the dynamic bridge is not merely a
        training-only contrastive head.
        """

        if self.prompt_conditioned_evidence_head is None or bridge.unit_ids is None or bridge.valid_units is None:
            raise ValueError("Prompt-conditioned routing requires the DualBridge head and unit metadata")
        if bridge.salience_logits is None:
            raise ValueError("Prompt-conditioned routing requires encoder salience logits")
        unit_count = int(bridge.unit_ids.max().item())
        if unit_count <= 0:
            zero = bridge.memory.float().sum() * 0.0
            return bridge.salience_logits, {
                "prompt_context_gate": self.prompt_conditioned_evidence_head.context_gate().detach(),
                "prompt_bridge_fusion_gate": self.prompt_bridge_fusion_gate().detach(),
                "prompt_bridge_dynamic_logit_rms": zero.detach(),
                "prompt_bridge_dynamic_logit_clip_fraction": zero.detach(),
            }
        if (sentence_reprs is None) != (valid_units is None):
            raise ValueError("sentence_reprs and valid_units must be supplied together")
        if sentence_reprs is None:
            sentence_reprs, valid_units = pool_units(bridge.memory, bridge.unit_ids, unit_count)
        elif (
            sentence_reprs.ndim != 3
            or valid_units is None
            or valid_units.ndim != 2
            or sentence_reprs.shape[:2] != valid_units.shape
            or sentence_reprs.shape[0] != bridge.memory.shape[0]
        ):
            raise ValueError("Supplied prompt-bridge unit representations have incompatible shapes")
        assert valid_units is not None
        source_context = masked_mean_pool(sentence_reprs, valid_units)
        query, keys = self.prompt_conditioned_evidence_head(prompt_state, source_context, sentence_reprs)
        raw_dynamic_logits = torch.einsum("bup,bp->bu", keys, query)
        if self.prompt_conditioned_inference_bridge:
            scaled_dynamic_logits = raw_dynamic_logits * self.prompt_bridge_dynamic_logit_scale
            dynamic_logits = scaled_dynamic_logits.clamp(
                -self.prompt_bridge_dynamic_logit_clip,
                self.prompt_bridge_dynamic_logit_clip,
            )
            dynamic_clip_fraction = (
                scaled_dynamic_logits.abs().ge(self.prompt_bridge_dynamic_logit_clip).to(torch.float32)
                * valid_units.to(torch.float32)
            ).sum() / valid_units.sum().clamp_min(1).to(torch.float32)
        else:
            dynamic_logits = raw_dynamic_logits
            dynamic_clip_fraction = raw_dynamic_logits.new_zeros(())
        width = min(bridge.salience_logits.shape[1], dynamic_logits.shape[1], valid_units.shape[1])
        fused_logits = bridge.salience_logits.clone()
        if self.prompt_conditioned_inference_bridge or not self.evidence_contrastive_attention_aligned:
            fused_logits[:, :width] = (
                bridge.salience_logits[:, :width]
                + self.prompt_bridge_fusion_gate().to(dynamic_logits.dtype) * dynamic_logits[:, :width]
            )
        diagnostics = {
            "prompt_context_gate": self.prompt_conditioned_evidence_head.context_gate().detach(),
            "prompt_bridge_fusion_gate": self.prompt_bridge_fusion_gate().detach(),
            "prompt_bridge_dynamic_logit_rms": dynamic_logits.float().square().mean().sqrt().detach(),
            "prompt_bridge_dynamic_logit_clip_fraction": dynamic_clip_fraction.detach(),
            "prompt_bridge_query": query,
            "prompt_bridge_keys": keys,
            "prompt_bridge_sentence_reprs": sentence_reprs,
            "prompt_bridge_valid_units": valid_units,
            "prompt_bridge_dynamic_logits": dynamic_logits,
        }
        return fused_logits, diagnostics

    def _neutral_prompt_probe(
        self,
        bridge: BridgeOutput,
        decoder_seed: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return a target-free task/source state for DualBridge routing.

        The probe neutralizes the *bridge-level*, unit-invariant salience
        prior: all source units retain equal total prior mass. A `qwen_native`
        encoder may still have already transformed its memory through its own
        native evidence view. Only the final configured copied cross-attention layers
        see source memory.  The same immutable layer-start argument is
        replayed under gradient checkpointing and is used verbatim at greedy
        inference.  It is therefore neither a second decoder nor a target
        conditioned pass.
        """

        self.decoder.clear_cross_attention_cache()
        if bridge.salience_logits is None:
            raise ValueError("DualBridge neutral probe requires encoder salience logits")
        probe_layers = self.prompt_bridge_source_probe_layers
        if probe_layers == 0:
            states, _ = self.decoder(
                input_ids=decoder_seed,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=None,
                use_cache=False,
            )
            return states

        neutral_bridge = self.adapter.reroute(bridge, torch.zeros_like(bridge.salience_logits))
        states, _ = self.decoder(
            input_ids=decoder_seed,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=neutral_bridge.memory,
            encoder_attention_mask=neutral_bridge.memory_mask,
            encoder_attention_bias=neutral_bridge.attention_bias,
            use_cache=False,
            encoder_cross_attention_start_layer=self.decoder.cross_attention_probe_start_layer(probe_layers),
        )
        return states

    @staticmethod
    def _effective_prompt_bridge_delta(
        static_bridge: BridgeOutput,
        fused_bridge: BridgeOutput,
    ) -> Dict[str, torch.Tensor]:
        """Measure the *post-cast* dynamic bias that SDPA can actually see.

        The bridge keeps its gate/logit arithmetic in FP32, but a BF16 decoder
        eventually consumes a BF16 additive key bias.  A tiny fused delta can
        therefore round to zero at long-source normalisation offsets.  These
        diagnostics make that failure observable instead of mistaking a
        disconnected-by-quantisation dynamic route.
        They are monitoring values only and never feed back into the loss.
        """

        zero = fused_bridge.memory.float().sum() * 0.0
        if static_bridge.attention_bias is None or fused_bridge.attention_bias is None or fused_bridge.unit_ids is None:
            return {
                "prompt_bridge_effective_delta_rms": zero.detach(),
                "prompt_bridge_effective_delta_nonzero_fraction": zero.detach(),
            }
        source_tokens = fused_bridge.memory_mask.bool() & fused_bridge.unit_ids.gt(0)
        count = source_tokens.sum().clamp_min(1).to(torch.float32)
        delta = fused_bridge.attention_bias.float() - static_bridge.attention_bias.float()
        masked_delta = delta * source_tokens.to(delta.dtype)
        return {
            "prompt_bridge_effective_delta_rms": (masked_delta.square().sum() / count).sqrt().detach(),
            "prompt_bridge_effective_delta_nonzero_fraction": (
                delta.ne(0).to(torch.float32) * source_tokens.to(torch.float32)
            )
            .sum()
            .div(count)
            .detach(),
        }

    @torch.no_grad()
    def prompt_condition_bridge_for_generation(
        self,
        bridge: BridgeOutput,
        decoder_seed: torch.Tensor,
    ) -> BridgeOutput:
        """Materialise the dynamic half of DualBridge before greedy decoding.

        This performs one short neutral, source-aware prompt probe, derives
        the task/source-conditioned sentence prior, then reuses the same
        projected source memory.  Greedy decoding itself remains one decoder
        and one fused cross-attention memory.
        """

        if not self.prompt_conditioned_inference_bridge:
            return bridge
        self.decoder.clear_cross_attention_cache()
        states = self._neutral_prompt_probe(bridge, decoder_seed)
        fused_logits, _ = self._prompt_conditioned_unit_logits(bridge, states[:, -1, :])
        return self.adapter.reroute(bridge, fused_logits)

    def _prompt_condition_bridge_for_training(
        self,
        bridge: BridgeOutput,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        evidence_labels: Optional[torch.Tensor],
    ) -> tuple[BridgeOutput, Dict[str, torch.Tensor]]:
        """Build the inference-equivalent fused bridge before teacher forcing.

        The prompt state is produced by the same short, neutral source probe
        used at greedy inference.  We then reuse the projected source memory,
        replace only its token prior, and run the full teacher-forced decoder
        pass with the fused bridge.  Thus CE directly optimizes the same
        source-conditioning route later used for generation; the dynamic
        route is not merely an auxiliary loss.
        """

        supervised = labels.ne(-100)
        if not bool(supervised.any(dim=1).all()):
            raise ValueError("DualBridge training requires one target token per example")
        first_target = supervised.long().argmax(dim=1)
        if not bool(first_target.eq(first_target[0]).all()):
            raise ValueError("DualBridge requires a common fixed decoder prompt across the batch")
        prompt_length = int(first_target[0].item()) + 1
        if prompt_length <= 0 or prompt_length > decoder_input_ids.shape[1]:
            raise ValueError("Invalid fixed decoder prompt boundary")
        prompt_mask = None if decoder_attention_mask is None else decoder_attention_mask[:, :prompt_length]
        prompt_states = self._neutral_prompt_probe(
            bridge,
            decoder_input_ids[:, :prompt_length],
            prompt_mask,
        )
        fused_logits, diagnostics = self._prompt_conditioned_unit_logits(bridge, prompt_states[:, -1, :])
        routed_bridge = self.adapter.reroute(
            bridge,
            fused_logits,
            evidence_labels=evidence_labels,
        )
        diagnostics["prompt_bridge_fused_logits"] = fused_logits
        diagnostics.update(self._effective_prompt_bridge_delta(bridge, routed_bridge))
        return routed_bridge, diagnostics

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
        target_sentence_ids: Optional[torch.Tensor] = None,
        sentence_evidence_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_source_diagnostics: bool = False,
        contrastive_mode: str = "local",
        return_full_logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run teacher forcing or target-free inference prefill.

        When ``labels`` is omitted, ``decoder_input_ids`` must contain only
        the fixed target-free decoder seed.  Autoregressive generation should
        use :func:`eviseq.evaluation.generation.generate`, which enforces the
        same contract while extending the seed token by token.
        """
        if contrastive_mode not in {"local", "representations_only", "deferred"}:
            raise ValueError(f"Unknown contrastive mode: {contrastive_mode}")

        encoded = self.encoder(input_ids, attention_mask, unit_ids=unit_ids)
        bridge = self.adapter(
            encoded.memory,
            attention_mask,
            unit_ids,
            encoded.unit_logits,
            encoded.valid_units,
            evidence_labels,
        )
        if labels is None:
            if contrastive_mode != "local":
                raise ValueError("Cached contrastive modes require decoder labels to locate the fixed prompt state")
            if self.prompt_conditioned_inference_bridge:
                bridge = self.prompt_condition_bridge_for_generation(bridge, decoder_input_ids)
            states, _ = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=bridge.memory,
                encoder_attention_mask=bridge.memory_mask,
                encoder_attention_bias=bridge.attention_bias,
                use_cache=False,
            )
            return {"logits": self.lm_head(states)}

        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("A training batch must contain supervised decoder labels")

        prompt_bridge: Optional[Dict[str, torch.Tensor]] = None
        active_bridge = bridge
        if self.prompt_conditioned_inference_bridge:
            active_bridge, prompt_bridge = self._prompt_condition_bridge_for_training(
                bridge,
                decoder_input_ids,
                decoder_attention_mask,
                labels,
                evidence_labels,
            )
        states, _ = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=active_bridge.memory,
            encoder_attention_mask=active_bridge.memory_mask,
            encoder_attention_bias=active_bridge.attention_bias,
            use_cache=False,
        )

        representations: Optional[Dict[str, torch.Tensor]] = None
        measure_source_use = self.alignment_head is not None and (
            self.training or bool(compute_source_diagnostics) or contrastive_mode != "local"
        )
        if measure_source_use:
            assert self.alignment_head is not None
            representations = self.alignment_head(
                bridge.memory,
                bridge.memory_mask,
                last_prompt_states(states, labels),
            )

        if contrastive_mode == "representations_only":
            if representations is None:
                raise RuntimeError("representations_only requires enabled contrastive learning")
            return representations

        full_logits = self.lm_head(states) if return_full_logits else None
        logits = full_logits[supervised] if full_logits is not None else self.lm_head(states[supervised])
        loss_ce = F.cross_entropy(
            logits.float(),
            labels[supervised],
            label_smoothing=float(getattr(self, "label_smoothing", 0.0)),
        )

        loss_contrastive = states.float().sum() * 0.0
        prompt_retrieval_accuracy = loss_contrastive.detach()
        if representations is not None and contrastive_mode == "local":
            loss_contrastive, prompt_retrieval_accuracy = info_nce_loss(
                representations["source_repr"],
                representations["prompt_repr"],
                self.contrastive_temperature,
                duplicate_mask=exact_duplicate_mask(input_ids, attention_mask),
            )

        loss_source_swap = states.float().sum() * 0.0
        source_swap_nll_gap = loss_source_swap.detach()
        source_swap_accuracy = loss_source_swap.detach()
        source_swap_negative_similarity = loss_source_swap.detach()
        if self.use_source_swap and measure_source_use and input_ids.shape[0] > 1:
            if self.prompt_conditioned_inference_bridge:
                raise RuntimeError("Source-swap loss is incompatible with prompt-conditioned bridge rerouting")
            if self.source_swap_strategy == "hard_in_batch":
                with torch.no_grad():
                    source_repr = source_memory_for_mining(
                        bridge.memory.detach(),
                        bridge.memory_mask,
                        pooling=self.contrastive_pooling,
                    )
                    permutation, selected_similarity = hard_negative_indices(source_repr)
                    source_swap_negative_similarity = selected_similarity.mean().to(states.dtype)
            elif self.source_swap_strategy == "cyclic":
                permutation = torch.arange(input_ids.shape[0], device=input_ids.device).roll(1)
            else:
                raise ValueError(f"Unknown source-swap strategy: {self.source_swap_strategy}")

            negative_memory = bridge.memory.index_select(0, permutation)
            negative_memory_mask = bridge.memory_mask.index_select(0, permutation)
            negative_attention_bias = (
                bridge.attention_bias.index_select(0, permutation) if bridge.attention_bias is not None else None
            )
            negative_states, _ = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=negative_memory,
                encoder_attention_mask=negative_memory_mask,
                encoder_attention_bias=negative_attention_bias,
                use_cache=False,
            )
            negative_logits = self.lm_head(negative_states[supervised])
            positive_nll = per_example_nll(logits, labels, supervised)
            negative_nll = per_example_nll(negative_logits, labels, supervised)
            loss_source_swap = source_swap_contrastive_loss(
                positive_nll,
                negative_nll,
                margin=self.source_swap_margin,
                temperature=self.source_swap_temperature,
            )
            nll_gap = negative_nll - positive_nll
            source_swap_nll_gap = nll_gap.detach().mean()
            source_swap_accuracy = nll_gap.detach().gt(0.0).float().mean()

        evi_results: Dict[str, torch.Tensor] = {}
        loss_evidence_contrastive = states.float().sum() * 0.0
        if (
            (self.evidence_contrastive_head is not None or self.prompt_conditioned_evidence_head is not None)
            and self.use_evidence_contrastive
            and (self.training or compute_source_diagnostics)
            and unit_ids is not None
            and evidence_labels is not None
            and encoded.unit_logits is not None
            and encoded.valid_units is not None
        ):
            unit_count = int(unit_ids.max().item()) if unit_ids is not None else 0
            if unit_count > 0:
                if (
                    self.evidence_contrastive_mode == "prompt_conditioned"
                    and prompt_bridge is not None
                    and "prompt_bridge_sentence_reprs" in prompt_bridge
                ):
                    sentence_reprs = prompt_bridge["prompt_bridge_sentence_reprs"]
                    valid_units = prompt_bridge["prompt_bridge_valid_units"]
                else:
                    sentence_reprs, valid_units = pool_units(active_bridge.memory, unit_ids, unit_count)
                attention_prior_energy = (
                    self.adapter.unit_attention_energy(active_bridge.salience_logits)
                    if bool(getattr(self, "evidence_contrastive_attention_aligned", False))
                    and active_bridge.salience_logits is not None
                    else None
                )

                if self.evidence_contrastive_mode == "sentence_aligned":
                    if target_sentence_ids is None or sentence_evidence_labels is None:
                        raise ValueError(
                            "sentence_aligned evidence contrastive requires target_sentence_ids and "
                            "sentence_evidence_labels from the dataset"
                        )
                    sentence_count = int(sentence_evidence_labels.shape[1])
                    summary_reprs, summary_valid = decoder_sentence_representations(
                        states,
                        labels,
                        target_sentence_ids,
                        sentence_count,
                    )
                    projected_queries = torch.nn.functional.normalize(
                        self.evidence_contrastive_head.query_projection(summary_reprs).float(), dim=-1
                    )
                    projected_keys = torch.nn.functional.normalize(
                        self.evidence_contrastive_head.key_projection(sentence_reprs).float(), dim=-1
                    )
                    evi_results = sentence_evidence_info_nce_loss(
                        query=projected_queries,
                        query_valid=summary_valid,
                        keys=projected_keys,
                        evidence_labels=sentence_evidence_labels,
                        valid_units=valid_units,
                        temperature=self.evidence_contrastive_temperature,
                        num_hard_negatives=self.evidence_hard_negatives,
                        salience_logits=encoded.unit_logits,
                        salience_boost=self.evidence_hard_negative_salience_boost,
                        salience_logit_bias=self.evidence_contrastive_salience_bias,
                        attention_prior_energy=attention_prior_energy,
                        attention_mining_boost=self.evidence_hard_negative_attention_boost,
                        global_evidence_labels=evidence_labels,
                    )
                elif self.evidence_contrastive_mode == "prompt_conditioned":
                    assert self.prompt_conditioned_evidence_head is not None
                    if prompt_bridge is None:
                        if self.prompt_conditioned_static_neutral_probe:
                            first_target = supervised.long().argmax(dim=1)
                            if not bool(first_target.eq(first_target[0]).all()):
                                raise ValueError("Static PCEB neutral probe requires a common fixed decoder prompt")
                            prompt_length = int(first_target[0].item()) + 1
                            prompt_mask = (
                                None if decoder_attention_mask is None else decoder_attention_mask[:, :prompt_length]
                            )
                            neutral_states = self._neutral_prompt_probe(
                                bridge,
                                decoder_input_ids[:, :prompt_length],
                                prompt_mask,
                            )
                            prompt_repr = neutral_states[:, -1, :]
                        else:
                            prompt_repr = last_prompt_states(states, labels)
                        fused_logits, prompt_bridge = self._prompt_conditioned_unit_logits(
                            bridge,
                            prompt_repr,
                            sentence_reprs=sentence_reprs,
                            valid_units=valid_units,
                        )
                    else:
                        fused_logits = prompt_bridge["prompt_bridge_fused_logits"]
                    q = prompt_bridge["prompt_bridge_query"]
                    k = prompt_bridge["prompt_bridge_keys"]
                    evi_results = evidence_info_nce_loss(
                        query=q,
                        keys=k,
                        evidence_labels=evidence_labels,
                        valid_units=valid_units,
                        temperature=self.evidence_contrastive_temperature,
                        num_hard_negatives=self.evidence_hard_negatives,
                        salience_logits=fused_logits,
                        salience_boost=self.evidence_hard_negative_salience_boost,
                        salience_logit_bias=self.evidence_contrastive_salience_bias,
                        attention_prior_energy=attention_prior_energy,
                        attention_mining_boost=self.evidence_hard_negative_attention_boost,
                    )
                    evi_results.update(
                        {
                            "prompt_context_gate": prompt_bridge["prompt_context_gate"],
                            "prompt_bridge_fusion_gate": prompt_bridge["prompt_bridge_fusion_gate"],
                            "prompt_bridge_dynamic_logit_rms": prompt_bridge["prompt_bridge_dynamic_logit_rms"],
                            "prompt_bridge_dynamic_logit_clip_fraction": prompt_bridge[
                                "prompt_bridge_dynamic_logit_clip_fraction"
                            ],
                            "prompt_bridge_effective_delta_rms": prompt_bridge.get(
                                "prompt_bridge_effective_delta_rms", states.new_zeros(())
                            ),
                            "prompt_bridge_effective_delta_nonzero_fraction": prompt_bridge.get(
                                "prompt_bridge_effective_delta_nonzero_fraction", states.new_zeros(())
                            ),
                        }
                    )
                    if self.prompt_conditioned_inference_bridge:
                        dynamic_salience = balanced_salience_loss(
                            fused_logits,
                            evidence_labels,
                            prompt_bridge["prompt_bridge_valid_units"],
                            ranking_weight=self.adapter.salience_ranking_weight,
                        )
                        evi_results["prompt_bridge_salience_loss"] = dynamic_salience
                else:
                    summary_repr = decoder_summary_representation(states, labels)
                    q, k = self.evidence_contrastive_head(summary_repr, sentence_reprs)
                    evi_results = evidence_info_nce_loss(
                        query=q,
                        keys=k,
                        evidence_labels=evidence_labels,
                        valid_units=valid_units,
                        temperature=self.evidence_contrastive_temperature,
                        num_hard_negatives=self.evidence_hard_negatives,
                        salience_logits=encoded.unit_logits,
                        salience_boost=self.evidence_hard_negative_salience_boost,
                        attention_prior_energy=attention_prior_energy,
                        attention_mining_boost=self.evidence_hard_negative_attention_boost,
                    )
                loss_evidence_contrastive = evi_results.get("evidence_contrastive_loss", loss_evidence_contrastive)

        auxiliary_scale = 1.0 if self.training else 0.0
        prompt_bridge_salience = evi_results.get("prompt_bridge_salience_loss")
        if prompt_bridge_salience is None:
            effective_salience_loss = bridge.loss_salience.float()
        else:
            mix = self.prompt_bridge_dynamic_salience_mix
            effective_salience_loss = (1.0 - mix) * bridge.loss_salience.float() + mix * prompt_bridge_salience.float()
        weighted_salience = self.salience_weight * effective_salience_loss

        loss_bridge_geometry = states.float().sum() * 0.0
        if self.bridge_geometry_weight > 0.0 and unit_ids is not None and encoded.valid_units is not None:
            unit_count = min(int(unit_ids.max().item()), self.bridge_geometry_max_units)
            if unit_count > 1:
                source_units, source_valid = pool_units(encoded.memory, unit_ids, unit_count)
                projected_units, projected_valid = pool_units(bridge.memory, unit_ids, unit_count)
                loss_bridge_geometry = pairwise_geometry_preservation_loss(
                    source_units,
                    projected_units,
                    source_valid & projected_valid,
                )
        weighted_bridge_geometry = self.bridge_geometry_weight * loss_bridge_geometry.float()

        local_contrastive_scale = self._contrastive_scale if contrastive_mode == "local" else 0.0
        weighted_contrastive = self.contrastive_weight * local_contrastive_scale * loss_contrastive.float()
        weighted_source_swap = self.source_swap_weight * local_contrastive_scale * loss_source_swap.float()

        evidence_scale = self._evidence_contrastive_scale
        weighted_evidence = self.evidence_contrastive_weight * evidence_scale * loss_evidence_contrastive.float()

        loss = loss_ce + auxiliary_scale * (
            weighted_salience
            + weighted_contrastive
            + weighted_source_swap
            + weighted_evidence
            + weighted_bridge_geometry
        )
        attention_prior_clip_fraction = states.new_zeros(())
        if active_bridge.salience_logits is not None and active_bridge.valid_units is not None:
            width = min(active_bridge.salience_logits.shape[1], active_bridge.valid_units.shape[1])
            valid_units = active_bridge.valid_units[:, :width].bool()
            attention_prior_clip_fraction = (
                active_bridge.salience_logits[:, :width].float().abs().ge(5.0).to(torch.float32)
                * valid_units.to(torch.float32)
            ).sum() / valid_units.sum().clamp_min(1).to(torch.float32)

        result: Dict[str, torch.Tensor] = {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_salience": effective_salience_loss,
            "weighted_salience": weighted_salience.detach(),
            "loss_bridge_geometry": loss_bridge_geometry.detach(),
            "weighted_bridge_geometry": weighted_bridge_geometry.detach(),
            "bridge_geometry_to_ce_ratio": (
                weighted_bridge_geometry.detach() / loss_ce.detach().float().clamp_min(1.0e-8)
            ),
            "salience_to_ce_ratio": (weighted_salience.detach() / loss_ce.detach().float().clamp_min(1.0e-8)),
            "loss_contrastive": loss_contrastive,
            "weighted_contrastive": weighted_contrastive.detach(),
            "contrastive_to_ce_ratio": (weighted_contrastive.detach() / loss_ce.detach().float().clamp_min(1.0e-8)),
            "prompt_retrieval_accuracy": prompt_retrieval_accuracy,
            "loss_source_swap": loss_source_swap,
            "weighted_source_swap": weighted_source_swap.detach(),
            "source_swap_to_ce_ratio": (weighted_source_swap.detach() / loss_ce.detach().float().clamp_min(1.0e-8)),
            "source_swap_nll_gap": source_swap_nll_gap,
            "source_swap_accuracy": source_swap_accuracy,
            "source_swap_negative_similarity": source_swap_negative_similarity,
            "contrastive_examples": states.new_tensor(input_ids.shape[0] if self.alignment_head is not None else 0),
            "contrastive_scale": states.new_tensor(self._contrastive_scale),
            "loss_evidence_contrastive": loss_evidence_contrastive.detach(),
            "weighted_evidence_contrastive": weighted_evidence.detach(),
            "evidence_contrastive_to_ce_ratio": (
                weighted_evidence.detach() / loss_ce.detach().float().clamp_min(1.0e-8)
            ),
            "evidence_contrastive_scale": states.new_tensor(evidence_scale),
            "evidence_hard_negatives": states.new_tensor(self.evidence_hard_negatives),
            "prompt_context_gate": evi_results.get("prompt_context_gate", states.new_tensor(0.0)),
            "prompt_bridge_fusion_gate": evi_results.get("prompt_bridge_fusion_gate", states.new_tensor(0.0)),
            "prompt_bridge_dynamic_logit_rms": evi_results.get(
                "prompt_bridge_dynamic_logit_rms", states.new_tensor(0.0)
            ),
            "prompt_bridge_dynamic_logit_clip_fraction": evi_results.get(
                "prompt_bridge_dynamic_logit_clip_fraction", states.new_tensor(0.0)
            ),
            "prompt_bridge_effective_delta_rms": evi_results.get(
                "prompt_bridge_effective_delta_rms", states.new_tensor(0.0)
            ),
            "prompt_bridge_effective_delta_nonzero_fraction": evi_results.get(
                "prompt_bridge_effective_delta_nonzero_fraction", states.new_tensor(0.0)
            ),
            "prompt_bridge_probe_layers": states.new_tensor(
                self.prompt_bridge_source_probe_layers
                if (self.prompt_conditioned_inference_bridge or self.prompt_conditioned_static_neutral_probe)
                else 0
            ),
            "evidence_attention_aligned": states.new_tensor(float(self.evidence_contrastive_attention_aligned)),
            "attention_prior_clip_fraction": attention_prior_clip_fraction.detach(),
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": self.decoder.cross_residual_ratio_mean().detach(),
            "bridge_projection_residual_ratio": active_bridge.projection_residual_ratio.detach(),
            "bridge_salience_gate": active_bridge.salience_attention_gate.detach(),
            "positive_attention_prior": active_bridge.positive_attention_prior.detach(),
            "negative_attention_prior": active_bridge.negative_attention_prior.detach(),
            "positive_attention_prior_gap": active_bridge.positive_attention_prior_gap.detach(),
            "bidirectional_gate_mean": encoded.native_gate_mean.detach(),
            "projection_gate": encoded.native_gate_mean.detach(),
            "evidence_view_gate": encoded.native_gate_mean.detach(),
        }
        if full_logits is not None:
            result["logits"] = full_logits

        for key in (
            "evidence_top1_accuracy",
            "positive_similarity",
            "hard_negative_similarity",
            "evidence_similarity_gap",
            "evidence_valid_examples",
        ):
            if key in evi_results:
                result[key] = evi_results[key]
            else:
                result[key] = states.new_tensor(0.0)

        if contrastive_mode == "deferred":
            if representations is None:
                raise RuntimeError("deferred contrastive backward requires enabled contrastive learning")
            result.update(representations)

        if active_bridge.salience_logits is not None and evidence_labels is not None:
            width = min(active_bridge.salience_logits.shape[1], evidence_labels.shape[1])
            valid = evidence_labels[:, :width].ge(0)
            if bool(valid.any()):
                probabilities = torch.sigmoid(active_bridge.salience_logits[:, :width][valid].float())
                predictions = probabilities.ge(0.5)
                gold = evidence_labels[:, :width][valid].gt(0.5)
                tp = (predictions & gold).sum().float()
                result["salience_probability_mean"] = probabilities.mean().detach()
                result["salience_predicted_positive_rate"] = predictions.float().mean().detach()
                result["salience_tp"] = tp.detach()
                result["salience_predicted_count"] = predictions.sum().float().detach()
                result["salience_gold_count"] = gold.sum().float().detach()
                result["salience_precision"] = (tp / predictions.sum().float().clamp_min(1)).detach()
                result["salience_recall"] = (tp / gold.sum().float().clamp_min(1)).detach()
                correct_pairs = probabilities.new_zeros(())
                pair_count = probabilities.new_zeros(())
                sal_logits = active_bridge.salience_logits[:, :width].float()
                for row in range(sal_logits.shape[0]):
                    row_valid = valid[row]
                    row_gold = evidence_labels[row, :width].gt(0.5) & row_valid
                    row_negative = ~evidence_labels[row, :width].gt(0.5) & row_valid
                    positives = sal_logits[row][row_gold]
                    negatives = sal_logits[row][row_negative]
                    if positives.numel() and negatives.numel():
                        differences = positives[:, None] - negatives[None, :]
                        correct_pairs = correct_pairs + differences.gt(0).float().sum()
                        correct_pairs = correct_pairs + 0.5 * differences.eq(0).float().sum()
                        pair_count = pair_count + differences.new_tensor(differences.numel())
                result["salience_correct_pairs"] = correct_pairs.detach()
                result["salience_pair_count"] = pair_count.detach()

        if self.alignment_head is not None and self.alignment_head.pool_gate is not None:
            result["alignment_last_pool_weight"] = torch.sigmoid(self.alignment_head.pool_gate.float()).mean().detach()

        return result

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")

        full = stage == "full_finetune"
        self.evidence_hard_negatives = (
            self.evidence_hard_negatives_full if full else self.evidence_hard_negatives_warmup
        )
        self.encoder.set_trainable(full)
        self.decoder.set_backbone_trainable(full)
        for parameter in self.adapter.parameters():
            parameter.requires_grad = True
        if self.alignment_head is not None:
            for parameter in self.alignment_head.parameters():
                parameter.requires_grad = True
        if self.evidence_contrastive_head is not None:
            for parameter in self.evidence_contrastive_head.parameters():
                parameter.requires_grad = True
        if self.prompt_conditioned_evidence_head is not None:
            for parameter in self.prompt_conditioned_evidence_head.parameters():
                parameter.requires_grad = True
        if self.prompt_bridge_fusion_logit is not None:
            self.prompt_bridge_fusion_logit.requires_grad = (
                self.prompt_conditioned_inference_bridge or not self.evidence_contrastive_attention_aligned
            )

        self._stage = stage

        if stage == "full_finetune":
            training_only_prefixes = (
                "alignment_head.",
                "evidence_contrastive_head.",
                "prompt_conditioned_evidence_head.",
            )
            # Native dual-mask controls are intentionally disabled when their
            # attention branch is not part of the selected variant.  They are
            # therefore allowed to remain frozen even during full backbone
            # fine-tuning; otherwise a valid ``full``/``causal``/``evidence``
            # the selected run fails this invariant before training starts.
            intentionally_frozen_prefixes = []
            encoder_variant = getattr(self.encoder, "variant", None)
            if encoder_variant is not None and encoder_variant != "dec2enc":
                intentionally_frozen_prefixes.append("encoder.generic_token_gate.")
            if encoder_variant is not None and encoder_variant not in {"evidence", "dec2enc"}:
                intentionally_frozen_prefixes.append("encoder.evidence_view_gate")
            allowed_frozen_prefixes = training_only_prefixes + tuple(intentionally_frozen_prefixes)
            frozen = [
                name
                for name, value in self.named_parameters()
                if not value.requires_grad
                and not any(name.startswith(p) for p in allowed_frozen_prefixes)
                and not (
                    name == "prompt_bridge_fusion_logit"
                    and not self.prompt_conditioned_inference_bridge
                    and self.evidence_contrastive_attention_aligned
                )
            ]
            if frozen:
                raise RuntimeError(f"Full fine-tuning left parameters frozen: {frozen[:20]}")

    def parameter_summary(self) -> Dict[str, int]:
        return {
            "encoder": sum(value.numel() for value in self.encoder.parameters()),
            "adapter": sum(value.numel() for value in self.adapter.parameters()),
            "contrastive_head": (
                sum(value.numel() for value in self.alignment_head.parameters())
                if self.alignment_head is not None
                else 0
            ),
            "evidence_contrastive_head": (
                sum(value.numel() for value in self.evidence_contrastive_head.parameters())
                if self.evidence_contrastive_head is not None
                else 0
            ),
            "prompt_conditioned_evidence_head": (
                sum(value.numel() for value in self.prompt_conditioned_evidence_head.parameters())
                + (self.prompt_bridge_fusion_logit.numel() if self.prompt_bridge_fusion_logit is not None else 0)
                if self.prompt_conditioned_evidence_head is not None
                else 0
            ),
            "decoder": sum(value.numel() for value in self.decoder.parameters()),
            "total": sum(value.numel() for value in self.parameters()),
            "trainable": sum(value.numel() for value in self.parameters() if value.requires_grad),
        }
