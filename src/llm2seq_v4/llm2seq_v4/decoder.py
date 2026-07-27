"""Full pretrained Qwen3 decoder with conventional copied cross-attention."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # pragma: no cover - compatibility guard for older Transformers
    GradientCheckpointingLayer = nn.Module


def _load_causal_lm(model_name: str, dtype: torch.dtype) -> Tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    raw_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = raw_config.get_text_config() if hasattr(raw_config, "get_text_config") else raw_config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    return model, config


class QwenCopiedCrossAttention(nn.Module):
    """Independent source attention initialized from the same decoder layer.

    Unlike merged attention, source keys receive their own softmax exactly as
    in the original LLM2Seq decoder. Unlike the old scratch decoder, Q/K/V/O
    begin in a pretrained Qwen coordinate system. A 4D source memory is
    interpreted as independent banks: each bank receives its own source
    softmax and produces its own attention output. The decoder layer routes
    those outputs only after attention, so lexical/semantic/summary evidence
    cannot cancel before key/value matching.
    """

    def __init__(
        self,
        source_attention: nn.Module,
        input_norm: nn.Module,
        config: Any,
        dropout: float,
        initialize_from_self: bool,
    ):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.q_proj = copy.deepcopy(source_attention.q_proj)
        self.k_proj = copy.deepcopy(source_attention.k_proj)
        self.v_proj = copy.deepcopy(source_attention.v_proj)
        self.o_proj = copy.deepcopy(source_attention.o_proj)
        self.q_norm = copy.deepcopy(getattr(source_attention, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(source_attention, "k_norm", nn.Identity()))
        self.memory_norm = copy.deepcopy(input_norm)
        self.dropout = float(dropout)
        self._memory_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if not initialize_from_self:
            initializer_range = float(getattr(config, "initializer_range", 0.02))
            for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            for norm in (self.q_norm, self.k_norm):
                if hasattr(norm, "weight"):
                    nn.init.ones_(norm.weight)

    def _project_memory(self, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if memory.ndim == 3:
            memory = memory.unsqueeze(1)
        if memory.ndim != 4:
            raise ValueError("Cross-attention memory must be [B, S, D] or [B, K, S, D]")
        batch, bank_count, source_length, _ = memory.shape
        normalized = self.memory_norm(memory)
        key = self.k_proj(normalized).view(
            batch,
            bank_count,
            source_length,
            self.num_kv_heads,
            self.head_dim,
        )
        value = self.v_proj(normalized).view(
            batch,
            bank_count,
            source_length,
            self.num_kv_heads,
            self.head_dim,
        )
        key = self.k_norm(key).transpose(2, 3)  # [B, K, Hkv, S, Dh]
        value = value.transpose(2, 3)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        encoder_attention_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim)
        query = self.q_norm(query).transpose(1, 2).unsqueeze(1)  # [B, 1, Hq, T, Dh]
        cached = self._memory_cache if not self.training else None
        if cached is None:
            key, value = self._project_memory(encoder_hidden_states)
            source_length = key.shape[3]
        else:
            key, value = cached
            source_length = key.shape[3]
            if key.shape[0] != batch:
                raise RuntimeError("Stale cross-attention cache; rebuild it for the current batch")
        bank_count = key.shape[1]
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            key = key.repeat_interleave(repeats, dim=2)
            value = value.repeat_interleave(repeats, dim=2)
        # Flatten bank into the batch dimension so CUDA SDPA keeps its normal
        # 4D Flash/memory-efficient kernel path instead of falling back on a
        # potentially materialized 5D attention matrix.
        query = query.expand(-1, bank_count, -1, -1, -1).reshape(
            batch * bank_count,
            self.num_heads,
            query_length,
            self.head_dim,
        )
        key = key.reshape(batch * bank_count, self.num_heads, source_length, self.head_dim)
        value = value.reshape(batch * bank_count, self.num_heads, source_length, self.head_dim)

        mask = None
        if encoder_attention_bias is not None:
            mask = encoder_attention_bias.to(query.dtype)[:, None, None, :]
            if encoder_attention_mask is not None:
                mask = mask.masked_fill(
                    ~encoder_attention_mask.bool()[:, None, None, :],
                    torch.finfo(query.dtype).min,
                )
        elif encoder_attention_mask is not None:
            mask = encoder_attention_mask.bool()[:, None, None, :]
        if mask is not None and bank_count > 1:
            mask = (
                mask[:, None]
                .expand(-1, bank_count, -1, -1, -1)
                .reshape(
                    batch * bank_count,
                    1,
                    1,
                    source_length,
                )
            )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = (
            attended.reshape(
                batch,
                bank_count,
                self.num_heads,
                query_length,
                self.head_dim,
            )
            .permute(0, 1, 3, 2, 4)
            .reshape(
                batch,
                bank_count,
                query_length,
                self.num_heads * self.head_dim,
            )
        )
        projected = self.o_proj(attended)
        return projected[:, 0] if bank_count == 1 else projected

    @torch.no_grad()
    def prepare_memory_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        self._memory_cache = tuple(value.contiguous() for value in self._project_memory(encoder_hidden_states))

    def clear_memory_cache(self) -> None:
        self._memory_cache = None


class QwenDecoderLayerWithCrossAttention(GradientCheckpointingLayer):
    """Native self-attention -> bank-wise source attention -> output routing -> FFN."""

    def __init__(
        self,
        base_layer: nn.Module,
        config: Any,
        dropout: float,
        gate_init: float,
        initialize_from_self: bool,
        memory_bank_count: int,
        memory_routing_mode: str,
        layer_index: int,
        total_layers: int,
        query_adaptive_routing: bool = False,
        query_router_max_delta: float = 2.0,
    ):
        super().__init__()
        if not 0.0 <= gate_init < 1.0:
            raise ValueError("cross gate initialization must be in [0, 1)")
        self.base_layer = base_layer
        self.cross_attn_norm = copy.deepcopy(base_layer.input_layernorm)
        self.cross_attn = QwenCopiedCrossAttention(
            base_layer.self_attn,
            base_layer.input_layernorm,
            config,
            dropout,
            initialize_from_self,
        )
        reference = base_layer.self_attn.q_proj.weight
        self.cross_attn.to(device=reference.device, dtype=reference.dtype)
        self.cross_gate = nn.Parameter(torch.tensor(math.atanh(gate_init), dtype=torch.float32))
        self.memory_bank_count = int(memory_bank_count)
        self.memory_routing_mode = str(memory_routing_mode)
        self.query_adaptive_routing = bool(query_adaptive_routing)
        self.query_router_max_delta = float(query_router_max_delta)
        if self.query_router_max_delta <= 0.0:
            raise ValueError("query_router_max_delta must be positive")
        if self.memory_routing_mode not in {"attention_output", "memory"}:
            raise ValueError("memory_routing_mode must be 'attention_output' or 'memory'")
        if self.memory_bank_count > 1:
            if self.memory_bank_count != 3:
                raise ValueError("Depth routing currently requires three memory banks")
            depth = float(layer_index) / max(1, total_layers - 1)
            lexical = max(0.0, 1.0 - 2.0 * depth)
            semantic = max(0.0, 1.0 - abs(2.0 * depth - 1.0))
            summary = max(0.0, 2.0 * depth - 1.0)
            prior = torch.tensor([lexical, semantic, summary], dtype=torch.float32)
            prior = prior.clamp_min(0.05)
            prior = prior / prior.sum()
            self.memory_router_logits = nn.Parameter(prior.log())
            if self.query_adaptive_routing:
                # A zero-initialized residual leaves the interpretable depth
                # prior exactly unchanged at step zero. It can then learn a
                # distinct lexical/semantic/summary mixture for every decoder
                # token without destabilizing the copied cross-attention.
                self.memory_router_norm: Optional[nn.Module] = copy.deepcopy(base_layer.input_layernorm)
                self.memory_router_proj: Optional[nn.Linear] = nn.Linear(
                    int(config.hidden_size),
                    self.memory_bank_count,
                    bias=False,
                )
                nn.init.zeros_(self.memory_router_proj.weight)
                self.memory_router_bank_weight: Optional[nn.Parameter] = nn.Parameter(
                    torch.zeros(
                        self.memory_bank_count,
                        int(config.hidden_size),
                        device=reference.device,
                        dtype=reference.dtype,
                    )
                )
                self.memory_router_norm.to(device=reference.device, dtype=reference.dtype)
                self.memory_router_proj.to(device=reference.device, dtype=reference.dtype)
            else:
                self.memory_router_norm = None
                self.memory_router_proj = None
                self.register_parameter("memory_router_bank_weight", None)
        else:
            self.register_parameter("memory_router_logits", None)
            self.memory_router_norm = None
            self.memory_router_proj = None
            self.register_parameter("memory_router_bank_weight", None)
        self.last_memory_routing: Optional[torch.Tensor] = None
        # Differentiable mean route from the latest forward. Diagnostics use
        # the detached copy above; auxiliary losses must use this tensor or
        # they would regularize only the static prior and miss dynamic collapse.
        self.last_memory_routing_for_loss: Optional[torch.Tensor] = None
        self.last_adaptive_routing_delta: Optional[torch.Tensor] = None
        self.last_cross_residual_ratio: Optional[torch.Tensor] = None

    def static_routing_weights(self) -> torch.Tensor:
        if self.memory_router_logits is None:
            return self.cross_gate.new_ones(1)
        return F.softmax(self.memory_router_logits.float(), dim=0)

    def routing_weights(
        self,
        router_states: Optional[torch.Tensor] = None,
        bank_outputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return static [K] or query-adaptive [B, T, K] routing weights."""
        static_logits = self.memory_router_logits
        if static_logits is None:
            return self.cross_gate.new_ones(1)
        if router_states is None or self.memory_router_proj is None:
            return F.softmax(static_logits.float(), dim=0)
        assert self.memory_router_norm is not None
        delta = self.memory_router_proj(self.memory_router_norm(router_states)).float()
        if bank_outputs is not None:
            if bank_outputs.ndim != 4 or bank_outputs.shape[1] != self.memory_bank_count:
                raise ValueError("bank_outputs must be [B, K, T, D] for adaptive routing")
            assert self.memory_router_bank_weight is not None
            normalized_banks = self.memory_router_norm(bank_outputs)
            bank_delta = torch.einsum(
                "bktd,kd->btk",
                normalized_banks,
                self.memory_router_bank_weight,
            ).float() / math.sqrt(normalized_banks.shape[-1])
            delta = delta + bank_delta
        limit = self.query_router_max_delta
        delta = limit * torch.tanh(delta / limit)
        weights = F.softmax(static_logits.float().view(1, 1, -1) + delta, dim=-1)
        static = F.softmax(static_logits.float(), dim=0).view(1, 1, -1)
        self.last_adaptive_routing_delta = (weights.detach() - static.detach()).abs().mean()
        return weights

    def effective_routing_weights(self) -> torch.Tensor:
        """Mean routing used by the most recent forward pass."""
        if self.last_memory_routing is not None:
            return self.last_memory_routing
        return self.static_routing_weights()

    def effective_routing_weights_for_loss(self) -> torch.Tensor:
        """Differentiable mean route used by the latest decoder forward."""
        if self.last_memory_routing_for_loss is not None:
            return self.last_memory_routing_for_loss
        return self.static_routing_weights()

    def route_memory(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim == 3:
            if self.memory_bank_count != 1:
                raise ValueError("Depth-routed decoder expected [B, 3, S, D] memory")
            return memory
        if memory.ndim != 4 or memory.shape[1] != self.memory_bank_count:
            raise ValueError(
                f"Expected [B, {self.memory_bank_count}, S, D] source memory, received {tuple(memory.shape)}"
            )
        weights = self.static_routing_weights().to(memory.dtype)
        self.last_memory_routing = weights.detach().float()
        self.last_memory_routing_for_loss = weights
        self.last_adaptive_routing_delta = weights.new_zeros((), dtype=torch.float32)
        return torch.einsum("k,bksd->bsd", weights, memory)

    def route_attention_outputs(
        self,
        bank_outputs: torch.Tensor,
        router_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse independently normalized bank-attention outputs.

        ``bank_outputs`` is [B, K, T, D]. Each K slice has already received
        its own attention softmax, which preserves complementary bank evidence.
        """
        if bank_outputs.ndim == 3:
            if self.memory_bank_count != 1 and self.memory_routing_mode != "memory":
                raise ValueError("Multi-bank routing expected [B, K, T, D] attention outputs")
            # Legacy pre-attention routing has already collapsed K memories to
            # one source tensor, so its single context needs no second route.
            return bank_outputs
        if bank_outputs.ndim != 4 or bank_outputs.shape[1] != self.memory_bank_count:
            raise ValueError(
                f"Expected [B, {self.memory_bank_count}, T, D] attention outputs, received {tuple(bank_outputs.shape)}"
            )
        weights = self.routing_weights(router_states, bank_outputs).to(bank_outputs.dtype)
        if weights.ndim == 1:
            self.last_memory_routing = weights.detach().float()
            self.last_memory_routing_for_loss = weights
            return torch.einsum("k,bktd->btd", weights, bank_outputs)
        mean_weights = weights.mean(dim=(0, 1))
        self.last_memory_routing = mean_weights.detach().float()
        self.last_memory_routing_for_loss = mean_weights
        return torch.einsum("btk,bktd->btd", weights, bank_outputs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_bias: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.base_layer.input_layernorm(hidden_states)
        self_states, _ = self.base_layer.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + self_states

        if encoder_hidden_states is not None:
            source_memory = encoder_hidden_states
            if (
                self.memory_routing_mode == "memory"
                and source_memory.ndim == 4
                and (self.training or self.cross_attn._memory_cache is None)
            ):
                source_memory = self.route_memory(source_memory)
            bank_cross_states = self.cross_attn(
                self.cross_attn_norm(hidden_states),
                source_memory,
                encoder_attention_mask,
                encoder_attention_bias,
            )
            cross_states = self.route_attention_outputs(
                bank_cross_states,
                router_states=hidden_states,
            )
            scaled = torch.tanh(self.cross_gate.float()).to(hidden_states.dtype) * cross_states
            with torch.no_grad():
                cross_rms = scaled.detach().float().square().mean().sqrt()
                hidden_rms = hidden_states.detach().float().square().mean().sqrt().clamp_min(1e-8)
                self.last_cross_residual_ratio = cross_rms / hidden_rms
            hidden_states = hidden_states + scaled

        residual = hidden_states
        hidden_states = self.base_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.base_layer.mlp(hidden_states)
        return residual + hidden_states

    @torch.no_grad()
    def prepare_cross_attention_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        memory = encoder_hidden_states
        if self.memory_routing_mode == "memory" and memory.ndim == 4:
            memory = self.route_memory(memory)
        self.cross_attn.prepare_memory_cache(memory)


class PretrainedQwenDecoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        config: Dict[str, Any],
        dtype: torch.dtype,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        causal_lm, qwen_config = _load_causal_lm(model_name, dtype)
        self.model_name = str(model_name)
        self.config = qwen_config
        self.backbone = causal_lm.model
        self.lm_head = causal_lm.lm_head
        self.cross_attention_every = max(1, int(config.get("cross_attention_every", 1)))
        self.initialize_cross_from_self = bool(config.get("initialize_cross_from_self", True))
        self.memory_bank_count = int(config.get("memory_bank_count", 1))
        self.memory_routing_mode = str(config.get("memory_routing_mode", "attention_output"))
        self.query_adaptive_routing = bool(config.get("query_adaptive_routing", False))
        self.last_prefix_drift_ratio: Optional[torch.Tensor] = None
        # Old resolved configs intentionally fall back to ``legacy`` so that
        # already-trained checkpoints retain bit-for-bit prefix injection.
        # New configs opt into RMS matching at the decoder input boundary.
        self.summary_prefix_scale_mode = str(config.get("summary_prefix_scale_mode", "legacy")).lower()
        self.summary_prefix_scale_multiplier = float(config.get("summary_prefix_scale_multiplier", 1.0))
        self.summary_prefix_scale_epsilon = float(config.get("summary_prefix_scale_epsilon", 1e-6))
        self.last_summary_prefix_rms: Optional[torch.Tensor] = None
        self.last_decoder_embedding_rms: Optional[torch.Tensor] = None
        self.last_prefix_to_embedding_rms_ratio: Optional[torch.Tensor] = None
        self.query_router_max_delta = float(config.get("query_router_max_delta", 2.0))
        dropout = float(config.get("cross_attention_dropout", 0.0))
        gate_init = float(config.get("cross_gate_init", 0.1))
        wrapped = []
        cross_indices = []
        base_layers = list(self.backbone.layers)
        for index, layer in enumerate(base_layers):
            self_attention = getattr(layer, "self_attn", None)
            if self_attention is None:
                raise ValueError("LLM2Seq-v4 requires a Qwen decoder layer with self_attn")
            if hasattr(self_attention, "layer_idx"):
                self_attention.layer_idx = index
            inject = (index + 1) % self.cross_attention_every == 0 or index == len(self.backbone.layers) - 1
            if inject:
                layer = QwenDecoderLayerWithCrossAttention(
                    layer,
                    qwen_config,
                    dropout,
                    gate_init,
                    self.initialize_cross_from_self,
                    self.memory_bank_count,
                    self.memory_routing_mode,
                    index,
                    len(base_layers),
                    self.query_adaptive_routing,
                    self.query_router_max_delta,
                )
                cross_indices.append(index)
            wrapped.append(layer)
        self.backbone.layers = nn.ModuleList(wrapped)
        self.cross_attention_indices = tuple(cross_indices)
        self._last_generation_routing_per_layer: Optional[torch.Tensor] = None
        self._last_generation_routing_observations = 0
        self.backbone.config.use_cache = False
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    @staticmethod
    def _masked_rms(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return FP32 RMS over valid sequence positions and hidden units."""
        weights = mask.to(device=values.device, dtype=torch.float32).unsqueeze(-1)
        count = weights.sum() * values.shape[-1]
        return ((values.float().square() * weights).sum() / count.clamp_min(1.0)).sqrt()

    def _scale_prefix_at_decoder_boundary(
        self,
        prefix_embeddings: torch.Tensor,
        token_embeddings: torch.Tensor,
        prefix_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Match soft-prefix magnitude to native decoder token embeddings.

        The scale statistics are detached: the adapter still receives gradients
        through the returned prefix, while it cannot game the normalization by
        changing only its output norm.
        """
        with torch.no_grad():
            raw_prefix_rms = self._masked_rms(prefix_embeddings, prefix_mask)
            decoder_embedding_rms = self._masked_rms(token_embeddings, token_mask)

        mode = str(getattr(self, "summary_prefix_scale_mode", "legacy")).lower()
        if mode == "match_embedding_rms":
            epsilon = float(getattr(self, "summary_prefix_scale_epsilon", 1e-6))
            multiplier = float(getattr(self, "summary_prefix_scale_multiplier", 1.0))
            scale = decoder_embedding_rms / raw_prefix_rms.clamp_min(epsilon)
            prefix_embeddings = prefix_embeddings * (scale * multiplier).to(dtype=prefix_embeddings.dtype)

        with torch.no_grad():
            injected_prefix_rms = self._masked_rms(prefix_embeddings, prefix_mask)
            self.last_summary_prefix_rms = injected_prefix_rms.detach()
            self.last_decoder_embedding_rms = decoder_embedding_rms.detach()
            self.last_prefix_to_embedding_rms_ratio = (
                injected_prefix_rms / decoder_embedding_rms.clamp_min(1e-12)
            ).detach()
        return prefix_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_bias: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        summary_prefix: Optional[torch.Tensor] = None,
        summary_prefix_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Decode tokens with an optional source-conditioned soft prefix.

        The prefix is inserted only on the first autoregressive call.  It is
        then retained by the native Qwen KV cache, so cached calls consume
        only the newly generated token.  During teacher forcing the returned
        prefix positions are removed, preserving the one-to-one alignment
        between decoder states and labels.
        """

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [B, T], received {tuple(input_ids.shape)}")
        if summary_prefix is None and summary_prefix_mask is not None:
            raise ValueError("summary_prefix_mask requires summary_prefix")

        first_step_with_prefix = summary_prefix is not None and past_key_values is None
        prefix_length = 0
        backbone_input_ids: Optional[torch.Tensor] = input_ids
        backbone_inputs_embeds: Optional[torch.Tensor] = None
        backbone_attention_mask = attention_mask
        if first_step_with_prefix:
            assert summary_prefix is not None
            if summary_prefix.ndim != 3:
                raise ValueError(f"summary_prefix must be [B, K, D], received {tuple(summary_prefix.shape)}")
            batch, token_length = input_ids.shape
            prefix_batch, prefix_length, prefix_hidden = summary_prefix.shape
            if prefix_batch != batch:
                raise ValueError(f"summary_prefix batch {prefix_batch} does not match input batch {batch}")
            if prefix_length <= 0:
                raise ValueError("summary_prefix must contain at least one latent token")

            embed_tokens = getattr(self.backbone, "embed_tokens", None)
            if embed_tokens is None:
                get_embeddings = getattr(self.backbone, "get_input_embeddings", None)
                embed_tokens = get_embeddings() if get_embeddings is not None else None
            if embed_tokens is None:
                raise AttributeError("Decoder backbone does not expose its input embeddings")
            token_embeddings = embed_tokens(input_ids)
            if prefix_hidden != token_embeddings.shape[-1]:
                raise ValueError(
                    f"summary_prefix hidden size {prefix_hidden} does not match decoder hidden size "
                    f"{token_embeddings.shape[-1]}"
                )
            if summary_prefix.device != token_embeddings.device:
                raise ValueError(
                    f"summary_prefix is on {summary_prefix.device}, but decoder inputs are on {token_embeddings.device}"
                )
            # The adapter may keep numerically sensitive projections in FP32;
            # cast only at the decoder boundary to avoid BF16/FP32 failures.
            prefix_embeddings = summary_prefix.to(dtype=token_embeddings.dtype)

            if attention_mask is None:
                token_mask = torch.ones(
                    (batch, token_length),
                    dtype=torch.long,
                    device=input_ids.device,
                )
            else:
                if attention_mask.ndim != 2 or attention_mask.shape != input_ids.shape:
                    raise ValueError(
                        "On the first prefix call attention_mask must match input_ids; received "
                        f"{tuple(attention_mask.shape)} versus {tuple(input_ids.shape)}"
                    )
                if attention_mask.device != input_ids.device:
                    raise ValueError("attention_mask and input_ids must be on the same device")
                token_mask = attention_mask
            if summary_prefix_mask is None:
                prefix_mask = torch.ones(
                    (batch, prefix_length),
                    dtype=token_mask.dtype,
                    device=token_mask.device,
                )
            else:
                if summary_prefix_mask.ndim != 2 or summary_prefix_mask.shape != (
                    batch,
                    prefix_length,
                ):
                    raise ValueError(
                        "summary_prefix_mask must be [B, K], received "
                        f"{tuple(summary_prefix_mask.shape)} for prefix {(batch, prefix_length)}"
                    )
                if summary_prefix_mask.device != input_ids.device:
                    raise ValueError("summary_prefix_mask and input_ids must be on the same device")
                prefix_mask = summary_prefix_mask.to(dtype=token_mask.dtype)
            prefix_embeddings = self._scale_prefix_at_decoder_boundary(
                prefix_embeddings,
                token_embeddings,
                prefix_mask,
                token_mask,
            )
            backbone_inputs_embeds = torch.cat([prefix_embeddings, token_embeddings], dim=1)
            backbone_input_ids = None
            backbone_attention_mask = torch.cat([prefix_mask, token_mask], dim=1)

        outputs = self.backbone(
            input_ids=backbone_input_ids,
            inputs_embeds=backbone_inputs_embeds,
            attention_mask=backbone_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_attention_bias=encoder_attention_bias,
        )
        hidden_states = outputs.last_hidden_state
        if first_step_with_prefix:
            expected_length = input_ids.shape[1] + prefix_length
            if hidden_states.ndim != 3 or hidden_states.shape[1] != expected_length:
                raise RuntimeError(
                    "Decoder backbone returned hidden states incompatible with the dynamic prefix: "
                    f"expected sequence length {expected_length}, received {tuple(hidden_states.shape)}"
                )
            # How far the prefix positions themselves travel through the backbone.
            # This is a scale-free representation diagnostic, not proof that
            # target-token positions use the prefix.  Causal use is measured by
            # intervening on the prefix and comparing target NLL in model.py.
            #
            # This must be measured as a DIRECTION change, not a magnitude change:
            # last_hidden_state comes out of the backbone's final RMSNorm, which
            # forces every emitted position to RMS ~ 1.0. A relative-L2 drift would
            # therefore read ~1/injected_RMS -- the reciprocal of the prefix scale,
            # carrying no information about movement, and algebraically coupled to
            # prefix_to_embedding_rms_ratio. Cosine distance is scale-free and
            # immune to the final norm.
            with torch.no_grad():
                injected = prefix_embeddings.float()
                emitted = hidden_states[:, :prefix_length, :].float()
                cosine = F.cosine_similarity(emitted, injected, dim=-1)
                self.last_prefix_drift_ratio = (1.0 - cosine).mean().detach()
            hidden_states = hidden_states[:, prefix_length:, :]
        return hidden_states, outputs.past_key_values if use_cache else None

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)
        if not trainable:
            for layer in self.backbone.layers:
                if not isinstance(layer, QwenDecoderLayerWithCrossAttention):
                    continue
                for parameter in layer.cross_attn_norm.parameters():
                    parameter.requires_grad = True
                for parameter in layer.cross_attn.parameters():
                    parameter.requires_grad = True
                layer.cross_gate.requires_grad = True
                if layer.memory_router_logits is not None:
                    layer.memory_router_logits.requires_grad = True
                if layer.memory_router_norm is not None:
                    for parameter in layer.memory_router_norm.parameters():
                        parameter.requires_grad = True
                if layer.memory_router_proj is not None:
                    for parameter in layer.memory_router_proj.parameters():
                        parameter.requires_grad = True
                if layer.memory_router_bank_weight is not None:
                    layer.memory_router_bank_weight.requires_grad = True

    @torch.no_grad()
    def prepare_cross_attention_cache(self, encoder_hidden_states: torch.Tensor) -> None:
        self.clear_cross_attention_cache()
        for layer in self.backbone.layers:
            if isinstance(layer, QwenDecoderLayerWithCrossAttention):
                layer.prepare_cross_attention_cache(encoder_hidden_states)

    def clear_cross_attention_cache(self) -> None:
        for layer in self.backbone.layers:
            if isinstance(layer, QwenDecoderLayerWithCrossAttention):
                layer.cross_attn.clear_memory_cache()

    def set_generation_routing_statistics(
        self,
        per_layer: torch.Tensor,
        observations: int,
    ) -> None:
        self._last_generation_routing_per_layer = per_layer.detach().float()
        self._last_generation_routing_observations = int(observations)

    def generation_routing_statistics(self) -> Tuple[torch.Tensor, int]:
        routing = self._last_generation_routing_per_layer
        if routing is None:
            routing = self.memory_routing_per_layer().detach().float()
        return routing, int(self._last_generation_routing_observations)

    def cross_gate_mean(self) -> torch.Tensor:
        gates = [
            torch.tanh(layer.cross_gate.float())
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention)
        ]
        return torch.stack(gates).mean() if gates else next(self.parameters()).new_zeros(())

    def prefix_drift_ratio(self) -> torch.Tensor:
        """Mean cosine distance travelled by prefix positions through the backbone.

        This describes prefix-state transformation only.  It cannot establish
        whether later target positions attend to or benefit from the prefix;
        use the prefix-swap NLL diagnostic for that causal question.

        Returns NaN when no prefix has been forwarded -- a configuration with no
        summary planner has not measured zero drift, it has not measured at all,
        and reporting 0.0 would read as "the prefix is inert".
        """
        if self.last_prefix_drift_ratio is None:
            return next(self.parameters()).new_full((), float("nan"))
        return self.last_prefix_drift_ratio

    def prefix_input_scale_metrics(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """RMS metrics for the prefix actually injected on the latest first step."""
        if (
            self.last_summary_prefix_rms is None
            or self.last_decoder_embedding_rms is None
            or self.last_prefix_to_embedding_rms_ratio is None
        ):
            missing = next(self.parameters()).new_full((), float("nan"), dtype=torch.float32)
            return missing, missing.clone(), missing.clone()
        return (
            self.last_summary_prefix_rms,
            self.last_decoder_embedding_rms,
            self.last_prefix_to_embedding_rms_ratio,
        )

    def cross_residual_ratio_mean(self) -> torch.Tensor:
        values = [
            layer.last_cross_residual_ratio
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention) and layer.last_cross_residual_ratio is not None
        ]
        return torch.stack(values).mean() if values else next(self.parameters()).new_zeros(())

    def memory_routing_mean(self) -> torch.Tensor:
        values = self.memory_routing_per_layer()
        if values.numel() == 0:
            return next(self.parameters()).new_ones(1)
        return values.mean(dim=0)

    def memory_routing_per_layer(self) -> torch.Tensor:
        values = [
            layer.effective_routing_weights()
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention)
        ]
        if not values:
            return next(self.parameters()).new_empty((0, 1))
        return torch.stack(values)

    def memory_routing_mean_for_loss(self) -> torch.Tensor:
        """Differentiable mean of the actual per-token routes."""
        values = [
            layer.effective_routing_weights_for_loss()
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention)
        ]
        if not values:
            return next(self.parameters()).new_ones(1)
        return torch.stack(values).mean(dim=0)

    def routing_balance_loss(self) -> torch.Tensor:
        """Prevent collapse of actual dynamic routes across batch and depth."""
        values = [
            layer.effective_routing_weights_for_loss()
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention)
        ]
        routing = torch.stack(values).mean(dim=0) if values else next(self.parameters()).new_ones(1)
        if routing.numel() <= 1:
            return routing.new_zeros(())
        bank_count = routing.numel()
        return (routing * (routing.clamp_min(1e-8).log() + math.log(bank_count))).sum()

    def memory_routing_entropy(self) -> torch.Tensor:
        """Normalized entropy of mean routing: 1 is balanced, 0 is collapsed."""
        routing = self.memory_routing_mean()
        if routing.numel() <= 1:
            return routing.new_ones(())
        return -(routing * routing.clamp_min(1e-8).log()).sum() / math.log(routing.numel())

    def adaptive_routing_delta_mean(self) -> torch.Tensor:
        values = [
            layer.last_adaptive_routing_delta
            for layer in self.backbone.layers
            if isinstance(layer, QwenDecoderLayerWithCrossAttention) and layer.last_adaptive_routing_delta is not None
        ]
        return torch.stack(values).mean() if values else next(self.parameters()).new_zeros(())
