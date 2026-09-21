"""Qwen causal decoder with one copied cross-attention in every layer."""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .grounded_copy import CopyState, GroundedCopyHead

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    GradientCheckpointingLayer = nn.Module


def _load_decoder(
    name: str, dtype: torch.dtype, trust_remote_code: bool = True, attention_implementation: str = "sdpa"
) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    if str(name) == "__tiny__":
        from transformers import Qwen3Config, Qwen3ForCausalLM

        config = Qwen3Config(
            vocab_size=128,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            max_position_embeddings=256,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
        )
        model = Qwen3ForCausalLM(config).to(dtype=dtype)
    else:
        raw = AutoConfig.from_pretrained(name, trust_remote_code=trust_remote_code)
        config = raw.get_text_config() if hasattr(raw, "get_text_config") else raw
        model = AutoModelForCausalLM.from_pretrained(
            name,
            config=config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation=attention_implementation,
        )
    return model, config


class CopiedCrossAttention(nn.Module):
    def __init__(
        self,
        self_attention: nn.Module,
        input_norm: nn.Module,
        config: Any,
        dropout: float,
        router_config: Optional[dict] = None,
    ):
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(config.num_attention_heads)
        self.num_kv_heads = int(config.num_key_value_heads)
        self.head_dim = int(getattr(config, "head_dim", self.hidden_size // self.num_heads))
        self.q_proj = copy.deepcopy(self_attention.q_proj)
        self.k_proj = copy.deepcopy(self_attention.k_proj)
        self.v_proj = copy.deepcopy(self_attention.v_proj)
        self.o_proj = copy.deepcopy(self_attention.o_proj)
        self.q_norm = copy.deepcopy(getattr(self_attention, "q_norm", nn.Identity()))
        self.k_norm = copy.deepcopy(getattr(self_attention, "k_norm", nn.Identity()))
        self.memory_norm = copy.deepcopy(input_norm)
        self.dropout = float(dropout)
        router_config = router_config or {}
        self.route_enabled = bool(router_config.get("enabled", False))
        self.last_route_scores: Optional[torch.Tensor] = None
        self._cache: Optional[tuple[torch.Tensor, ...]] = None
        if self.route_enabled:
            route_init = float(router_config.get("cross_strength_init", 0.10))
            self.route_strength_max = float(router_config.get("cross_strength_max", 0.50))
            if not 0 < route_init < self.route_strength_max:
                raise ValueError("cross route strength must satisfy 0 < init < max")
            self.route_strength_raw = nn.Parameter(
                torch.tensor(math.log(route_init / (self.route_strength_max - route_init)))
            )
            self.region_key_proj = nn.Linear(
                self.hidden_size,
                self.num_kv_heads * self.head_dim,
                bias=False,
                device=self.k_proj.weight.device,
                dtype=self.k_proj.weight.dtype,
            )
            with torch.no_grad():
                self.region_key_proj.weight.copy_(self.k_proj.weight)

    def _memory_kv(
        self, memory: torch.Tensor, value_memory: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = memory.shape
        if value_memory is not None and value_memory.shape != memory.shape:
            raise ValueError("Key and value memories must have matching token positions and dimensions")
        hidden = self.memory_norm(memory)
        key = self.k_norm(self.k_proj(hidden).view(batch, length, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        value_hidden = hidden if value_memory is None else self.memory_norm(value_memory)
        value = self.v_proj(value_hidden).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return key, value

    def _routed_memory_kv(
        self,
        memory: torch.Tensor,
        value_memory: Optional[torch.Tensor],
        region_states: Optional[torch.Tensor],
        source_region_ids: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        key, value = self._memory_kv(memory, value_memory)
        if not self.route_enabled:
            return key, value
        if region_states is None or source_region_ids is None:
            raise ValueError("enabled evidence router requires source regions")
        batch, source_length, _ = memory.shape
        if source_region_ids.shape != (batch, source_length) or region_states.shape[:1] != (batch,):
            raise ValueError("source regions must match memory batch and source length")
        region_keys = self.k_norm(
            self.region_key_proj(self.memory_norm(region_states)).view(
                batch, region_states.shape[1], self.num_kv_heads, self.head_dim
            )
        ).transpose(1, 2)
        gathered = region_keys.gather(
            2,
            source_region_ids.clamp_min(0)[:, None, :, None].expand(-1, self.num_kv_heads, -1, self.head_dim),
        )
        strength = self.route_strength_max * torch.sigmoid(self.route_strength_raw.float())
        key = key + strength.to(key.dtype) * gathered * source_region_ids.ge(0)[:, None, :, None]
        return key, value, region_keys

    def forward(
        self,
        query_states: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        source_bias: Optional[torch.Tensor],
        value_memory: Optional[torch.Tensor] = None,
        region_states: Optional[torch.Tensor] = None,
        source_region_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, query_length, _ = query_states.shape
        query = self.q_norm(
            self.q_proj(query_states).view(batch, query_length, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        cached = (
            self._cache
            if self._cache is not None and not self.training
            else self._routed_memory_kv(memory, value_memory, region_states, source_region_ids)
        )
        key, value = cached[:2]
        self.last_route_scores = None
        if self.route_enabled:
            region_keys = cached[2]
            repeats = self.num_heads // self.num_kv_heads
            grouped_query = (
                query.float().reshape(batch, self.num_kv_heads, repeats, query_length, self.head_dim).mean(2)
            )
            self.last_route_scores = torch.einsum("bhtd,bhkd->btk", grouped_query, region_keys.float()) / (
                self.num_kv_heads * math.sqrt(self.head_dim)
            )
        mask: Optional[torch.Tensor]
        if source_bias is not None:
            if source_bias.ndim == 2:
                mask = source_bias[:, None, None, :].to(query.dtype)
            elif source_bias.ndim == 4:
                mask = source_bias.to(query.dtype)
            else:
                raise ValueError("source_bias must be [B,S] or [B,1,1,S]")
        elif memory_mask is not None:
            mask = None
        else:
            mask = None
        if memory_mask is not None:
            valid = memory_mask.bool()[:, None, None, :]
            mask = valid if mask is None else mask.masked_fill(~valid, float("-inf"))
        repeats = self.num_heads // self.num_kv_heads
        if repeats > 1:
            try:
                attended = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                    enable_gqa=True,
                )
            except TypeError:
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
                attended = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False
                )
            except RuntimeError as error:
                if not any(term in str(error).lower() for term in ("gqa", "grouped query", "no available kernel")):
                    raise
                key = key.repeat_interleave(repeats, dim=1)
                value = value.repeat_interleave(repeats, dim=1)
                attended = F.scaled_dot_product_attention(
                    query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False
                )
        else:
            attended = F.scaled_dot_product_attention(
                query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False
            )
        return self.o_proj(attended.transpose(1, 2).reshape(batch, query_length, self.num_heads * self.head_dim))

    @torch.no_grad()
    def prepare_cache(
        self,
        memory: torch.Tensor,
        value_memory: Optional[torch.Tensor] = None,
        region_states: Optional[torch.Tensor] = None,
        source_region_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self._cache = tuple(
            value.contiguous()
            for value in self._routed_memory_kv(memory, value_memory, region_states, source_region_ids)
        )

    def clear_cache(self) -> None:
        self._cache = None


class DecoderLayerWithCross(GradientCheckpointingLayer):
    def __init__(
        self,
        base: nn.Module,
        config: Any,
        dropout: float,
        gate_init: float,
        gate_max: float,
        index: int,
        router_config: Optional[dict] = None,
    ):
        super().__init__()
        if not 0.0 < gate_init < gate_max <= 1.0:
            raise ValueError("cross gate must satisfy 0 < init < max <= 1")
        self.base = base
        self.cross_norm = copy.deepcopy(base.input_layernorm)
        self.cross = CopiedCrossAttention(base.self_attn, base.input_layernorm, config, dropout, router_config)
        self.cross_gate = nn.Parameter(torch.tensor(math.log(gate_init / (gate_max - gate_init)), dtype=torch.float32))
        self.cross_gate_max = float(gate_max)
        self.index = int(index)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_bias: Optional[torch.Tensor] = None,
        encoder_value_states: Optional[torch.Tensor] = None,
        encoder_region_states: Optional[torch.Tensor] = None,
        encoder_source_region_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        normalized = self.base.input_layernorm(hidden_states)
        self_states, _ = self.base.self_attn(
            hidden_states=normalized,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + self_states
        if encoder_hidden_states is not None:
            cross = self.cross(
                self.cross_norm(hidden_states),
                encoder_hidden_states,
                encoder_attention_mask,
                encoder_attention_bias,
                value_memory=encoder_value_states,
                region_states=encoder_region_states,
                source_region_ids=encoder_source_region_ids,
            )
            hidden_states = (
                hidden_states + self.cross_gate_max * torch.sigmoid(self.cross_gate).to(hidden_states.dtype) * cross
            )
        residual = hidden_states
        hidden_states = self.base.post_attention_layernorm(hidden_states)
        return residual + self.base.mlp(hidden_states)

    @torch.no_grad()
    def prepare_cache(
        self,
        memory: torch.Tensor,
        value_memory: Optional[torch.Tensor] = None,
        region_states: Optional[torch.Tensor] = None,
        source_region_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.cross.prepare_cache(memory, value_memory, region_states, source_region_ids)

    def clear_cache(self) -> None:
        self.cross.clear_cache()


class QwenCrossDecoder(nn.Module):
    def __init__(
        self,
        name: str,
        config: dict,
        dtype: torch.dtype,
        gradient_checkpointing: bool = True,
        trust_remote_code: bool = True,
        attention_implementation: str = "sdpa",
    ):
        super().__init__()
        causal_lm, model_config = _load_decoder(name, dtype, trust_remote_code, attention_implementation)
        self.model_name = str(name)
        self.config = model_config
        self.ce_chunk_size = int(config.get("ce_chunk_size", 1024))
        if self.ce_chunk_size <= 0:
            raise ValueError("ce_chunk_size must be positive")
        self.backbone = causal_lm.model
        self.lm_head = causal_lm.lm_head
        router_config = config.get("evidence_router", {})
        self.route_enabled = bool(router_config.get("enabled", False))
        copy_config = config.get("grounded_copy", {})
        self.grounded_copy = None
        if copy_config.get("enabled", False):
            with torch.random.fork_rng(devices=[]):
                self.grounded_copy = GroundedCopyHead(
                    int(model_config.hidden_size),
                    int(copy_config.get("key_dim", 128)),
                    float(copy_config.get("gate_init", 0.05)),
                    float(router_config.get("copy_strength_init", 0.10)) if self.route_enabled else 0.0,
                    float(router_config.get("copy_strength_max", 0.50)),
                )
        if int(config.get("cross_attention_every", 1)) != 1:
            raise ValueError("AFMR requires cross-attention in every decoder layer")
        gate_init = float(config.get("cross_gate_init", 0.10))
        gate_max = float(config.get("cross_gate_max", 1.0))
        wrapped = []
        original_layers = list(self.backbone.layers)
        routed_layers = int(router_config.get("last_n_layers", 2)) if self.route_enabled else 0
        if self.route_enabled and not 1 <= routed_layers <= len(original_layers):
            raise ValueError("evidence_router.last_n_layers must be within decoder depth")
        # Keep AFMR's random initialization identical in router-on/off ablations.
        with torch.random.fork_rng(devices=[]):
            for index, layer in enumerate(original_layers):
                if not hasattr(layer, "self_attn"):
                    raise ValueError("AFMR requires decoder layers with self_attn")
                if hasattr(layer.self_attn, "layer_idx"):
                    layer.self_attn.layer_idx = index
                wrapped.append(
                    DecoderLayerWithCross(
                        layer,
                        model_config,
                        float(config.get("attention_dropout", 0.0)),
                        gate_init,
                        gate_max,
                        index,
                        router_config if index >= len(original_layers) - routed_layers else None,
                    )
                )
        self.backbone.layers = nn.ModuleList(wrapped)
        self.cross_attention_indices = tuple(range(len(wrapped)))
        self.backbone.config.use_cache = False
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @property
    def embed_tokens(self) -> nn.Module:
        return self.backbone.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        source_bias: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        return_logits: bool = True,
        value_memory: Optional[torch.Tensor] = None,
        copy_state: Optional[CopyState] = None,
        region_states: Optional[torch.Tensor] = None,
        source_region_ids: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[Any], Optional[torch.Tensor]]:
        if (self.grounded_copy is None) != (copy_state is None):
            raise ValueError("Decoder grounded-copy configuration and source state disagree")
        if self.route_enabled and (region_states is None or source_region_ids is None):
            raise ValueError("evidence router requires source regions")
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(~attention_mask.bool(), 0)[:, -input_ids.shape[1] :]
        self.backbone.layers[-1].cross.last_route_scores = None
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            return_dict=True,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            encoder_attention_bias=source_bias,
            encoder_value_states=value_memory,
            encoder_region_states=region_states,
            encoder_source_region_ids=source_region_ids,
        )
        hidden = outputs.last_hidden_state
        output_hidden = hidden[:, -1:] if use_cache else hidden
        route_scores = self.backbone.layers[-1].cross.last_route_scores if self.route_enabled else None
        if self.route_enabled and route_scores is None:
            raise RuntimeError("final decoder cross-attention did not produce evidence scores")
        if use_cache and route_scores is not None:
            route_scores = route_scores[:, -1:]
        logits = self.lm_head(output_hidden) if return_logits else None
        if logits is not None and self.grounded_copy is not None:
            logits = self.grounded_copy.mix_logits(output_hidden, logits, copy_state, route_scores)
        loss = None
        if labels is not None:
            shift_labels = labels[:, 1:].contiguous()
            if self.grounded_copy is not None and logits is None:
                loss = self.grounded_copy.loss(
                    hidden[:, :-1],
                    shift_labels,
                    copy_state,
                    self.lm_head,
                    self.ce_chunk_size,
                    None if route_scores is None else route_scores[:, :-1],
                )
                return logits, getattr(outputs, "past_key_values", None) if use_cache else None, loss
            if logits is not None:
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            else:
                from torch.utils.checkpoint import checkpoint

                flat_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
                flat_labels = shift_labels.reshape(-1)
                valid = flat_labels.ne(-100)
                flat_hidden = flat_hidden[valid]
                flat_labels = flat_labels[valid]

                def chunk_ce(states, targets):
                    return F.cross_entropy(self.lm_head(states).float(), targets, ignore_index=-100, reduction="sum")

                losses = []
                for start in range(0, flat_hidden.shape[0], self.ce_chunk_size):
                    states = flat_hidden[start : start + self.ce_chunk_size]
                    targets = flat_labels[start : start + self.ce_chunk_size]
                    losses.append(
                        checkpoint(chunk_ce, states, targets, use_reentrant=False)
                        if torch.is_grad_enabled()
                        else chunk_ce(states, targets)
                    )
                loss = torch.stack(losses).sum() if losses else hidden.sum() * 0.0
            loss = loss / shift_labels.ne(-100).sum().clamp_min(1)
        return logits, getattr(outputs, "past_key_values", None) if use_cache else None, loss

    @torch.no_grad()
    def prepare_cross_cache(
        self,
        memory: torch.Tensor,
        value_memory: Optional[torch.Tensor] = None,
        region_states: Optional[torch.Tensor] = None,
        source_region_ids: Optional[torch.Tensor] = None,
    ) -> None:
        for layer in self.backbone.layers:
            if isinstance(layer, DecoderLayerWithCross):
                layer.prepare_cache(memory, value_memory, region_states, source_region_ids)

    def clear_cross_cache(self) -> None:
        for layer in self.backbone.layers:
            if isinstance(layer, DecoderLayerWithCross):
                layer.clear_cache()

    def select_cross_cache(self, indices: torch.Tensor) -> None:
        for layer in self.backbone.layers:
            if isinstance(layer, DecoderLayerWithCross) and layer.cross._cache is not None:
                layer.cross._cache = tuple(value.index_select(0, indices) for value in layer.cross._cache)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)
        if not trainable:
            if self.grounded_copy is not None:
                self.grounded_copy.requires_grad_(True)
            for layer in self.backbone.layers:
                if isinstance(layer, DecoderLayerWithCross):
                    for parameter in layer.cross.parameters():
                        parameter.requires_grad = True
                    layer.cross_norm.requires_grad_(True)
                    layer.cross_gate.requires_grad = True
