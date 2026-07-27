"""LLM2Seq-v5: output-centric latent bridge for low-data summarization."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import AdapterOutput, ProspectiveSummaryBridge
from .contrastive import (
    SourceAlignmentHead,
    hard_negative_indices,
    info_nce_loss,
    last_prompt_states,
    per_example_nll,
    source_memory_for_mining,
    source_swap_contrastive_loss,
)
from .decoder import PretrainedQwenDecoder
from .encoder import EmbeddingTokenEncoder
from .phrase_pointer import StatefulPhrasePointer
from .response_alignment import ordered_response_alignment_loss


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}")
    return mapping[name]


class LLM2SeqV5(nn.Module):
    """One encoder -> one prospective-summary bridge -> one causal decoder.

    The adapter keeps dense token memory for coverage and additionally predicts
    a short, ordered sequence of summary latents.  Those latents are injected as
    a source-conditioned soft prefix into the pretrained decoder.  Everything
    used to supervise the latents is training-only; inference still consumes
    only the source document.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        model_config = config.get("model", {})
        dtype = torch_dtype(str(model_config.get("dtype", "float32")))
        gradient_checkpointing = bool(model_config.get("gradient_checkpointing", True))
        fallback_hidden = int(model_config.get("hidden_size", 1024))
        encoder_hidden_size = int(model_config.get("encoder_hidden_size", fallback_hidden))
        decoder_hidden_size = int(model_config.get("decoder_hidden_size", fallback_hidden))
        self.encoder = EmbeddingTokenEncoder(
            str(model_config["encoder_name"]),
            dtype,
            gradient_checkpointing,
            attn_implementation=str(model_config.get("encoder_attn_implementation", "sdpa")),
            expected_hidden_layers=(
                int(model_config["encoder_num_hidden_layers"])
                if model_config.get("encoder_num_hidden_layers") is not None
                else None
            ),
            expected_attention_mode=str(model_config.get("encoder_attention_mode", "auto")),
            trust_remote_code=bool(model_config.get("encoder_trust_remote_code", True)),
            revision=(
                str(model_config["encoder_revision"]) if model_config.get("encoder_revision") is not None else None
            ),
        )
        if self.encoder.hidden_size != encoder_hidden_size:
            raise ValueError(
                f"Configured encoder hidden size {encoder_hidden_size} != checkpoint size {self.encoder.hidden_size}"
            )
        self.adapter = ProspectiveSummaryBridge(
            encoder_hidden_size,
            decoder_hidden_size,
            config.get("adapter", {}),
        )
        expected_banks = 3 if bool(config.get("adapter", {}).get("depth_routed_memory", False)) else 1
        configured_banks = int(config.get("decoder", {}).get("memory_bank_count", 1))
        if configured_banks != expected_banks:
            raise ValueError(
                f"Adapter produces {expected_banks} memory bank(s), but decoder.memory_bank_count={configured_banks}"
            )
        self.decoder = PretrainedQwenDecoder(
            str(model_config["decoder_name"]),
            config.get("decoder", {}),
            dtype,
            gradient_checkpointing,
            revision=(
                str(model_config["decoder_revision"]) if model_config.get("decoder_revision") is not None else None
            ),
        )
        decoder_hidden = int(self.decoder.config.hidden_size)
        if decoder_hidden != decoder_hidden_size:
            raise ValueError(
                f"Configured decoder hidden size {decoder_hidden_size} != checkpoint size {decoder_hidden}"
            )

        # --- Contrastive learning ---
        objectives = config.get("objectives", {})
        self.salience_weight = float(objectives.get("salience_weight", 0.1))
        self.use_contrastive = bool(objectives.get("use_contrastive", True))
        self.use_prompt_alignment = self.use_contrastive and bool(objectives.get("use_prompt_alignment", True))
        self.use_source_swap = self.use_contrastive and bool(objectives.get("use_source_swap", True))
        self.contrastive_weight = float(objectives.get("contrastive_weight", 0.1)) if self.use_prompt_alignment else 0.0
        self.contrastive_temperature = float(objectives.get("contrastive_temperature", 0.07))
        self.source_swap_weight = float(objectives.get("source_swap_weight", 0.1)) if self.use_source_swap else 0.0
        self.source_swap_margin = float(objectives.get("source_swap_margin", 0.2))
        self.source_swap_temperature = float(objectives.get("source_swap_temperature", 1.0))
        self.source_swap_strategy = str(objectives.get("source_swap_strategy", "hard_in_batch"))
        self.contrastive_pooling = str(objectives.get("contrastive_pooling", "mean_last"))
        self.routing_balance_weight = float(objectives.get("routing_balance_weight", 0.0))
        self.label_smoothing = float(objectives.get("label_smoothing", 0.0))
        self.response_alignment_weight = float(objectives.get("response_alignment_weight", 0.0))
        self.response_alignment_temperature = float(objectives.get("response_alignment_temperature", 0.10))

        # V5 realizes source-supported phrases through the normal decoder
        # output distribution. This remains a single encoder -> bridge ->
        # decoder model; the pointer is a compact output head on decoder states,
        # not a second encoder/decoder or an inference-time retriever.
        phrase_config = config.get("phrase_pointer", {})
        self.use_phrase_pointer = bool(phrase_config.get("enabled", False))
        self.phrase_mixture_weight = float(objectives.get("phrase_mixture_weight", 1.0))
        self.phrase_copy_weight = float(objectives.get("phrase_copy_weight", 0.10))
        self.phrase_continue_weight = float(objectives.get("phrase_continue_weight", 0.10))
        self.phrase_label_weight = float(objectives.get("phrase_label_weight", 0.05))
        self.phrase_coverage_weight = float(objectives.get("phrase_coverage_weight", 0.02))
        if self.use_phrase_pointer:
            self.phrase_pointer: Optional[StatefulPhrasePointer] = StatefulPhrasePointer(
                hidden_size=decoder_hidden_size,
                vocabulary_size=int(self.decoder.config.vocab_size),
                rank=int(phrase_config.get("rank", 128)),
                phrase_hidden_size=int(phrase_config.get("phrase_hidden_size", 256)),
                dropout=float(phrase_config.get("dropout", 0.10)),
                phrase_bias_scale=float(phrase_config.get("phrase_bias_scale", 0.5)),
                continuation_strength=float(phrase_config.get("continuation_strength", 1.0)),
                generate_probability_init=float(phrase_config.get("generate_probability_init", 0.98)),
                use_continuation=bool(phrase_config.get("use_continuation", True)),
                detach_recurrent_state=bool(phrase_config.get("detach_recurrent_state", True)),
            )
        else:
            self.phrase_pointer = None

        if self.use_prompt_alignment:
            projection_size = int(objectives.get("contrastive_projection_size", 256))
            self.alignment_head: Optional[SourceAlignmentHead] = SourceAlignmentHead(
                decoder_hidden_size,
                projection_size,
                pooling=str(objectives.get("contrastive_pooling", "mean_last")),
            )
        else:
            self.alignment_head = None

        self._stage = "unconfigured"
        self._contrastive_scale = 1.0  # adjusted during training for warmup
        self._plan_only_probability = 0.0
        self._oracle_evidence_mix = 0.0

    @property
    def lm_head(self) -> nn.Module:
        return self.decoder.lm_head

    def _decoder_embedding_weight(self) -> torch.Tensor:
        """Return decoder token embeddings for response-space supervision.

        Production Qwen exposes ``backbone.embed_tokens``.  The fallbacks keep
        pure-tensor architecture tests and compatible causal backbones honest
        without special-casing the alignment loss itself.
        """

        backbone = getattr(self.decoder, "backbone", None)
        embeddings = getattr(backbone, "embed_tokens", None)
        if embeddings is None:
            embeddings = getattr(self.decoder, "embedding", None)
        if embeddings is None and backbone is not None and hasattr(backbone, "get_input_embeddings"):
            embeddings = backbone.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is None:
            raise AttributeError("Decoder does not expose an input-embedding weight")
        return weight

    def set_contrastive_scale(self, scale: float) -> None:
        """Set the contrastive loss scaling factor (for warmup schedule)."""
        self._contrastive_scale = max(0.0, min(1.0, float(scale)))

    def set_summary_curriculum(
        self,
        *,
        plan_only_probability: float,
        oracle_evidence_mix: float,
    ) -> None:
        """Set train-time bridge curriculum without changing inference state.

        ``plan_only_probability`` stochastically removes dense source memory
        while retaining the source-derived summary prefix.  This prevents the
        decoder from learning the same token-memory shortcut seen in v3.
        ``oracle_evidence_mix`` anneals reference-derived evidence guidance into
        the predicted salience distribution.  Both values must be reset to zero
        for validation/test; the training loop enforces that invariant.
        """

        if not 0.0 <= float(plan_only_probability) <= 1.0:
            raise ValueError("plan_only_probability must be in [0, 1]")
        if not 0.0 <= float(oracle_evidence_mix) <= 1.0:
            raise ValueError("oracle_evidence_mix must be in [0, 1]")
        self._plan_only_probability = float(plan_only_probability)
        self._oracle_evidence_mix = float(oracle_evidence_mix)
        self.adapter.set_oracle_evidence_mix(self._oracle_evidence_mix)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
    ) -> AdapterOutput:
        hidden_states = self.encoder(input_ids, attention_mask)
        return self.adapter(
            hidden_states,
            attention_mask,
            unit_ids=unit_ids,
            evidence_labels=evidence_labels,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        unit_ids: Optional[torch.Tensor] = None,
        evidence_labels: Optional[torch.Tensor] = None,
        phrase_labels: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        compute_source_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        adapter_output = self.encode(
            input_ids,
            attention_mask,
            unit_ids=unit_ids,
            evidence_labels=evidence_labels,
        )

        # On a controlled subset of training examples, remove the dense token
        # path but keep the summary latents.  This is one forward pass (not an
        # expensive second decoder pass) and makes prefix usage identifiable.
        conditioning_memory = adapter_output.memory
        conditioning_bias = adapter_output.attention_bias
        plan_only_mask = torch.zeros(
            conditioning_memory.shape[0],
            dtype=torch.bool,
            device=conditioning_memory.device,
        )
        if (
            self.training
            and labels is not None
            and adapter_output.summary_prefix is not None
            and getattr(self, "_plan_only_probability", 0.0) > 0.0
        ):
            plan_only_mask = torch.rand_like(plan_only_mask, dtype=torch.float32).lt(
                getattr(self, "_plan_only_probability", 0.0)
            )
            if bool(plan_only_mask.any()):
                memory_view = (conditioning_memory.shape[0],) + (1,) * (conditioning_memory.ndim - 1)
                conditioning_memory = conditioning_memory.masked_fill(plan_only_mask.view(memory_view), 0)
                if conditioning_bias is not None:
                    conditioning_bias = conditioning_bias.masked_fill(plan_only_mask[:, None], 0)
        decoder_states, _ = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=conditioning_memory,
            encoder_attention_mask=adapter_output.memory_mask,
            encoder_attention_bias=conditioning_bias,
            summary_prefix=adapter_output.summary_prefix,
            summary_prefix_mask=adapter_output.summary_prefix_mask,
            use_cache=False,
        )
        positive_cross_residual_ratio = self.decoder.cross_residual_ratio_mean().detach()
        # Capture this immediately: source-swap and prefix-swap diagnostics run
        # additional decoder forwards and overwrite the decoder's latest state.
        # This is only a representation self-drift diagnostic; causal prefix use
        # is measured below with the prefix-swap NLL gap.
        positive_prefix_self_drift = self.decoder.prefix_drift_ratio().detach()
        positive_routing_for_loss = self.decoder.memory_routing_mean_for_loss()
        positive_routing = positive_routing_for_loss.detach()
        routing_entropy = self.decoder.memory_routing_entropy().detach()
        adaptive_routing_delta = self.decoder.adaptive_routing_delta_mean().detach()
        if labels is None:
            return {"logits": self.lm_head(decoder_states)}
        supervised = labels.ne(-100)
        if not bool(supervised.any()):
            raise ValueError("A training batch must contain supervised decoder labels")
        logits = self.lm_head(decoder_states[supervised])

        # --- Label smoothing cross-entropy ---
        loss_ce = F.cross_entropy(
            logits.float(),
            labels[supervised],
            label_smoothing=self.label_smoothing,
        )

        decoder_embedding_weight = self._decoder_embedding_weight()
        if adapter_output.summary_prefix is not None and getattr(self, "response_alignment_weight", 0.0) > 0.0:
            alignment = ordered_response_alignment_loss(
                adapter_output.summary_prefix,
                labels,
                decoder_embedding_weight,
                temperature=getattr(self, "response_alignment_temperature", 0.10),
            )
        else:
            alignment_zero = decoder_states.float().sum() * 0.0
            alignment = {
                "loss": alignment_zero,
                "cosine": alignment_zero.detach(),
                "accuracy": alignment_zero.detach(),
                "valid_slots": torch.zeros((), dtype=torch.long, device=decoder_states.device),
            }
        loss_response_alignment = alignment["loss"]

        phrase_zero = decoder_states.float().sum() * 0.0
        phrase_losses = {
            "loss_phrase_mixture": loss_ce,
            "loss_phrase_copy": phrase_zero,
            "loss_phrase_continue": phrase_zero,
            "loss_phrase_labels": phrase_zero,
            "loss_phrase_coverage": phrase_zero,
            "phrase_copyable_rate": phrase_zero.detach(),
            "phrase_continuation_available_rate": phrase_zero.detach(),
            "phrase_mode_generate": decoder_states.new_tensor(1.0),
            "phrase_mode_new": phrase_zero.detach(),
            "phrase_mode_continue": phrase_zero.detach(),
            "phrase_copy_support_accuracy": phrase_zero.detach(),
        }
        phrase_pointer = getattr(self, "phrase_pointer", None)
        if phrase_pointer is not None:
            if unit_ids is None:
                raise ValueError("unit_ids are required when the phrase pointer is enabled")
            source_copy_mask = attention_mask.bool() & unit_ids.gt(0)
            # Preserve the meaning of the V4 plan-only curriculum: examples
            # selected for that intervention must not recover a dense source
            # shortcut through the new pointer path.
            if bool(plan_only_mask.any()):
                source_copy_mask = source_copy_mask & ~plan_only_mask[:, None]
            phrase_losses = phrase_pointer.teacher_forced_loss(
                decoder_states=decoder_states,
                lm_logits=logits,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                source_memory=adapter_output.memory,
                source_token_ids=input_ids,
                source_unit_ids=unit_ids,
                source_copy_mask=source_copy_mask,
                attention_bias=adapter_output.attention_bias,
                phrase_labels=phrase_labels,
            )

        # --- Contrastive loss ---
        loss_contrastive = decoder_states.new_zeros(())
        prompt_retrieval_accuracy = decoder_states.new_zeros(())
        measure_source_use = self.training or bool(compute_source_diagnostics)
        # Capture the differentiable route from the correct source before the
        # counterfactual decoder pass overwrites each layer's latest route.
        loss_routing_balance = decoder_states.new_zeros(())
        if measure_source_use and self.routing_balance_weight > 0.0:
            loss_routing_balance = self.decoder.routing_balance_loss()
        if self.alignment_head is not None and measure_source_use:
            prompt_states = last_prompt_states(decoder_states, labels)
            representations = self.alignment_head(
                adapter_output.memory,
                adapter_output.memory_mask,
                prompt_states,
                bank_weights=positive_routing_for_loss if adapter_output.memory.ndim == 4 else None,
            )
            loss_contrastive = info_nce_loss(
                representations["memory_repr"],
                representations["decoder_repr"],
                self.contrastive_temperature,
            )
            similarity = representations["decoder_repr"] @ representations["memory_repr"].T
            expected = torch.arange(similarity.shape[0], device=similarity.device)
            prompt_retrieval_accuracy = similarity.detach().argmax(dim=1).eq(expected).float().mean()

        # --- Counterfactual source-swap loss ---
        loss_source_swap = decoder_states.new_zeros(())
        source_swap_nll_gap = decoder_states.new_zeros(())
        source_swap_accuracy = decoder_states.new_zeros(())
        source_swap_negative_similarity = decoder_states.new_zeros(())
        prefix_swap_nll_gap = decoder_states.new_zeros(())
        prefix_swap_accuracy = decoder_states.new_zeros(())
        source_permutation: Optional[torch.Tensor] = None
        positive_nll_for_diagnostics: Optional[torch.Tensor] = None
        batch_size = decoder_states.shape[0]
        if measure_source_use and self.source_swap_weight > 0.0 and batch_size > 1:
            if self.source_swap_strategy == "hard_in_batch":
                with torch.no_grad():
                    source_repr = source_memory_for_mining(
                        adapter_output.memory.detach(),
                        adapter_output.memory_mask,
                        bank_weights=positive_routing.detach() if adapter_output.memory.ndim == 4 else None,
                        pooling=self.contrastive_pooling,
                    )
                    permutation, selected_similarity = hard_negative_indices(source_repr)
                    source_swap_negative_similarity = selected_similarity.mean().to(decoder_states.dtype)
            elif self.source_swap_strategy == "cyclic":
                # Deterministic derangement retained as the clean ablation.
                permutation = torch.arange(batch_size, device=decoder_states.device).roll(1)
            else:  # validated at config load; keep a local defensive guard.
                raise ValueError(f"Unknown source-swap strategy: {self.source_swap_strategy}")
            source_permutation = permutation
            negative_memory = adapter_output.memory.index_select(0, permutation)
            negative_memory_mask = adapter_output.memory_mask.index_select(0, permutation)
            negative_attention_bias = (
                adapter_output.attention_bias.index_select(0, permutation)
                if adapter_output.attention_bias is not None
                else None
            )
            negative_prefix = (
                adapter_output.summary_prefix.index_select(0, permutation)
                if adapter_output.summary_prefix is not None
                else None
            )
            negative_prefix_mask = (
                adapter_output.summary_prefix_mask.index_select(0, permutation)
                if adapter_output.summary_prefix_mask is not None
                else None
            )
            if bool(plan_only_mask.any()):
                memory_view = (negative_memory.shape[0],) + (1,) * (negative_memory.ndim - 1)
                negative_memory = negative_memory.masked_fill(plan_only_mask.view(memory_view), 0)
                if negative_attention_bias is not None:
                    negative_attention_bias = negative_attention_bias.masked_fill(plan_only_mask[:, None], 0)
            negative_states, _ = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=negative_memory,
                encoder_attention_mask=negative_memory_mask,
                encoder_attention_bias=negative_attention_bias,
                summary_prefix=negative_prefix,
                summary_prefix_mask=negative_prefix_mask,
                use_cache=False,
            )
            negative_logits = self.lm_head(negative_states[supervised])
            positive_nll = per_example_nll(logits, labels, supervised)
            positive_nll_for_diagnostics = positive_nll
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

        # Validation-only causal diagnostic for the dynamic summary prefix.
        # Keep the correct dense source memory fixed and replace only the prefix
        # with another example's prefix.  A positive NLL gap therefore means the
        # decoder benefits from the source-specific prefix, unlike prefix
        # self-drift, which merely measures how prefix positions are transformed.
        if bool(compute_source_diagnostics) and adapter_output.summary_prefix is not None and batch_size > 1:
            if source_permutation is None:
                source_permutation = torch.arange(batch_size, device=decoder_states.device).roll(1)
            swapped_prefix = adapter_output.summary_prefix.index_select(0, source_permutation)
            swapped_prefix_mask = (
                adapter_output.summary_prefix_mask.index_select(0, source_permutation)
                if adapter_output.summary_prefix_mask is not None
                else None
            )
            prefix_negative_states, _ = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=conditioning_memory,
                encoder_attention_mask=adapter_output.memory_mask,
                encoder_attention_bias=conditioning_bias,
                summary_prefix=swapped_prefix,
                summary_prefix_mask=swapped_prefix_mask,
                use_cache=False,
            )
            prefix_negative_logits = self.lm_head(prefix_negative_states[supervised])
            if positive_nll_for_diagnostics is None:
                positive_nll_for_diagnostics = per_example_nll(logits, labels, supervised)
            prefix_negative_nll = per_example_nll(prefix_negative_logits, labels, supervised)
            prefix_gap = prefix_negative_nll - positive_nll_for_diagnostics
            prefix_swap_nll_gap = prefix_gap.detach().mean()
            prefix_swap_accuracy = prefix_gap.detach().gt(0.0).float().mean()

        # --- Total loss ---
        effective_contrastive_weight = self.contrastive_weight * self._contrastive_scale
        auxiliary_scale = 1.0 if self.training else 0.0
        mixture_weight = getattr(self, "phrase_mixture_weight", 0.0) if phrase_pointer is not None else 0.0
        primary_generation_loss = (1.0 - mixture_weight) * loss_ce + mixture_weight * phrase_losses[
            "loss_phrase_mixture"
        ].float()
        loss = (
            primary_generation_loss
            + self.salience_weight * adapter_output.loss_salience.float()
            + auxiliary_scale * getattr(self, "response_alignment_weight", 0.15) * loss_response_alignment.float()
            + auxiliary_scale * getattr(self, "phrase_copy_weight", 0.0) * phrase_losses["loss_phrase_copy"].float()
            + auxiliary_scale
            * getattr(self, "phrase_continue_weight", 0.0)
            * phrase_losses["loss_phrase_continue"].float()
            + auxiliary_scale * getattr(self, "phrase_label_weight", 0.0) * phrase_losses["loss_phrase_labels"].float()
            + auxiliary_scale
            * getattr(self, "phrase_coverage_weight", 0.0)
            * phrase_losses["loss_phrase_coverage"].float()
            + auxiliary_scale * effective_contrastive_weight * loss_contrastive.float()
            + auxiliary_scale * self.source_swap_weight * self._contrastive_scale * loss_source_swap.float()
            + auxiliary_scale * self.routing_balance_weight * self._contrastive_scale * loss_routing_balance.float()
        )

        summary_prefix_rms = (
            adapter_output.summary_prefix.detach().float().square().mean().sqrt()
            if adapter_output.summary_prefix is not None
            else decoder_states.new_zeros((), dtype=torch.float32)
        )
        with torch.no_grad():
            decoder_token_embeddings = F.embedding(
                decoder_input_ids,
                decoder_embedding_weight.detach(),
            ).float()
            if decoder_attention_mask is None:
                valid_decoder_embeddings = decoder_token_embeddings.reshape(
                    -1,
                    decoder_token_embeddings.shape[-1],
                )
            else:
                valid_decoder_embeddings = decoder_token_embeddings[decoder_attention_mask.bool()]
            decoder_embedding_rms = valid_decoder_embeddings.square().mean().sqrt()
            prefix_to_embedding_rms_ratio = summary_prefix_rms / decoder_embedding_rms.clamp_min(1e-12)

        result = {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_salience": adapter_output.loss_salience,
            "loss_response_alignment": loss_response_alignment,
            "response_alignment_cosine": alignment["cosine"].detach(),
            "response_alignment_accuracy": alignment["accuracy"].detach(),
            "response_alignment_valid_slots": alignment["valid_slots"].detach(),
            **phrase_losses,
            "plan_only_rate": plan_only_mask.float().mean().detach(),
            "plan_only_probability": decoder_states.new_tensor(getattr(self, "_plan_only_probability", 0.0)),
            "oracle_evidence_mix": decoder_states.new_tensor(getattr(self, "_oracle_evidence_mix", 0.0)),
            "loss_contrastive": loss_contrastive,
            "prompt_retrieval_accuracy": prompt_retrieval_accuracy,
            "loss_source_swap": loss_source_swap,
            "source_swap_nll_gap": source_swap_nll_gap,
            "source_swap_accuracy": source_swap_accuracy,
            "source_swap_negative_similarity": source_swap_negative_similarity,
            "prefix_swap_nll_gap": prefix_swap_nll_gap,
            "prefix_swap_accuracy": prefix_swap_accuracy,
            "loss_routing_balance": loss_routing_balance,
            "memory_routing_entropy": routing_entropy,
            "adaptive_routing_delta": adaptive_routing_delta,
            "contrastive_scale": decoder_states.new_tensor(self._contrastive_scale),
            "cross_gate_mean": self.decoder.cross_gate_mean().detach(),
            "cross_residual_ratio": positive_cross_residual_ratio,
            "bidirectional_gate_mean": self.adapter.bidirectional_gate_mean().detach(),
            "branch_context_gate_mean": self.adapter.branch_context_gate_mean().detach(),
            "projection_gate": torch.tanh(self.adapter.projection.residual_gate.float()).detach(),
            "summary_prefix_rms": summary_prefix_rms,
            "decoder_embedding_rms": decoder_embedding_rms,
            "prefix_to_embedding_rms_ratio": prefix_to_embedding_rms_ratio,
            "prefix_drift_ratio": positive_prefix_self_drift,
        }
        routing = positive_routing.detach()
        if routing.numel() == 1:
            result["memory_route_summary"] = routing[0]
        else:
            for index, name in enumerate(("lexical", "semantic", "summary")):
                result[f"memory_route_{name}"] = routing[index]
        if self.adapter.salience_attention_gate is not None:
            result["salience_attention_gate"] = torch.tanh(self.adapter.salience_attention_gate.float()).detach()
        if self.alignment_head is not None and self.alignment_head.pool_gate is not None:
            result["alignment_last_pool_weight"] = torch.sigmoid(self.alignment_head.pool_gate.float()).mean().detach()
        if adapter_output.layer_weights is not None:
            for index, value in enumerate(adapter_output.layer_weights):
                result[f"fusion_weight_{index}"] = value.detach()
        if adapter_output.salience_logits is not None and evidence_labels is not None:
            width = min(adapter_output.salience_logits.shape[1], evidence_labels.shape[1])
            valid = evidence_labels[:, :width].ge(0)
            if bool(valid.any()):
                probabilities = torch.sigmoid(adapter_output.salience_logits[:, :width][valid].float())
                predictions = probabilities.ge(0.5)
                gold = evidence_labels[:, :width][valid].gt(0.5)
                true_positive = (predictions & gold).sum().float()
                result["salience_probability_mean"] = probabilities.mean().detach()
                result["salience_predicted_positive_rate"] = predictions.float().mean().detach()
                result["salience_precision"] = (true_positive / predictions.sum().float().clamp_min(1.0)).detach()
                result["salience_recall"] = (true_positive / gold.sum().float().clamp_min(1.0)).detach()
        return result

    def set_training_stage(self, stage: str) -> None:
        if stage not in {"interface_warmup", "full_finetune"}:
            raise ValueError(f"Unknown training stage: {stage}")
        full = stage == "full_finetune"
        self.encoder.set_trainable(full)
        self.decoder.set_backbone_trainable(full)
        for parameter in self.adapter.parameters():
            parameter.requires_grad = True
        if self.alignment_head is not None:
            for parameter in self.alignment_head.parameters():
                parameter.requires_grad = True
        phrase_pointer = getattr(self, "phrase_pointer", None)
        if phrase_pointer is not None:
            for parameter in phrase_pointer.parameters():
                parameter.requires_grad = True
        self._stage = stage
        if full:
            frozen = [
                name
                for name, parameter in self.named_parameters()
                if parameter.numel() > 0 and not parameter.requires_grad
            ]
            if frozen:
                raise RuntimeError("Full fine-tuning left parameters frozen: " + ", ".join(frozen[:20]))

    def parameter_summary(self) -> Dict[str, int]:
        alignment_params = (
            sum(parameter.numel() for parameter in self.alignment_head.parameters())
            if self.alignment_head is not None
            else 0
        )
        phrase_pointer_params = (
            sum(parameter.numel() for parameter in getattr(self, "phrase_pointer", ()).parameters())
            if getattr(self, "phrase_pointer", None) is not None
            else 0
        )
        return {
            "encoder": sum(parameter.numel() for parameter in self.encoder.parameters()),
            "adapter": sum(parameter.numel() for parameter in self.adapter.parameters()),
            "alignment_head": alignment_params,
            "phrase_pointer": phrase_pointer_params,
            "decoder": sum(parameter.numel() for parameter in self.decoder.parameters()),
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
        }
