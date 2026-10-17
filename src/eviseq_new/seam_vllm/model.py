"""Qwen3 vLLM adapter that preserves the trained AFMR/SEAM computation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from eviseq_afmr.config import load_config, resolve_path
from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge
from eviseq_afmr.modeling.encoder import build_encoder
from eviseq_afmr.modeling.grounded_copy import CopyState, GroundedCopyHead
from eviseq_afmr.training.checkpoint import architecture_spec
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer, Qwen3ForCausalLM, Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from ._compat import (
    validate_compute_dtype,
    validate_decoder_config,
    validate_tokenizer_config,
    validate_vllm_config,
)
from ._copy_batch import CopyBatchCache, stack_copy_states
from ._copy_logits import CopyLogitsMixer
from ._cross_bias import pad_cross_key, pad_cross_query
from ._encoder_batch import encoder_batches
from ._prompt import configured_decoder_prompt_ids
from ._shape import padded_attention_head_dim
from ._tensors import _as_2d, bridge_prompt_batch, pack_bridge_memory, unpack_bridge_memory
from ._weights import (
    expected_qwen3_decoder_checkpoint_keys,
    missing_parameter_names,
    normalize_decoder_lm_head_key,
)


def _config_path(hf_config: Any, key: str) -> Path:
    value = getattr(hf_config, key, None)
    if not value:
        raise ValueError(f"SEAM requires hf_overrides.{key} with a local file path")
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SEAM local file does not exist: {path}")
    return path


class _SEAMCrossAttention(nn.Module):
    """vLLM paged cross-attention with the trained source-bias term."""

    def __init__(
        self,
        base: Qwen3DecoderLayer,
        config: Any,
        cache_config,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        attention = base.self_attn
        self.hidden_size = int(config.hidden_size)
        self.num_heads = int(attention.num_heads)
        self.num_kv_heads = int(attention.num_kv_heads)
        self.head_dim = int(attention.head_dim)
        self.padded_head_dim = padded_attention_head_dim(self.head_dim)
        if self.num_heads % self.num_kv_heads:
            raise ValueError("SEAM Qwen3 cross-attention requires query heads divisible by KV heads")

        # Names and shapes mirror the copied cross-attention tensors in the
        # AFMR checkpoint. Two padded coordinates carry compensated source_bias.
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.k_norm = RMSNorm(self.head_dim, eps=float(config.rms_norm_eps))
        self.memory_norm = RMSNorm(self.hidden_size, eps=float(config.rms_norm_eps))
        self.query_scale = float(getattr(attention, "scaling", self.head_dim**-0.5)) * math.sqrt(self.head_dim)
        self.attn = CrossAttention(
            num_heads=self.num_heads,
            head_size=self.padded_head_dim,
            scale=self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def forward(
        self,
        query_states: torch.Tensor,
        memory: torch.Tensor | None,
        value_memory: torch.Tensor | None,
        source_bias: torch.Tensor | None,
    ) -> torch.Tensor:
        query = self.q_proj(query_states).view(-1, self.num_heads, self.head_dim)
        query = self.q_norm(query) * self.query_scale
        query_pad = pad_cross_query(query, self.padded_head_dim)
        query_pad = query_pad.reshape(-1, self.num_heads * self.padded_head_dim)

        key = value = None
        if memory is not None:
            if source_bias is None:
                raise ValueError("SEAM cross-attention requires the source-bias side channel")
            value_memory = memory if value_memory is None else value_memory
            if memory.shape != value_memory.shape or source_bias.shape != memory.shape[:1]:
                raise ValueError("SEAM packed cross-attention memory has inconsistent shapes")
            normalized = self.memory_norm(memory)
            normalized_values = normalized if value_memory is memory else self.memory_norm(value_memory)
            key = self.k_proj(normalized).view(-1, self.num_kv_heads, self.head_dim)
            key = self.k_norm(key)
            key_pad = pad_cross_key(key, source_bias, self.padded_head_dim)
            key = key_pad.reshape(-1, self.num_kv_heads * self.padded_head_dim)
            value = self.v_proj(normalized_values).view(-1, self.num_kv_heads, self.head_dim)
            value_pad = value.new_zeros(value.shape[0], self.num_kv_heads, self.padded_head_dim)
            value_pad[..., : self.head_dim] = value
            value = value_pad.reshape(-1, self.num_kv_heads * self.padded_head_dim)

        attended = self.attn(query_pad, key, value)
        attended = attended.view(-1, self.num_heads, self.padded_head_dim)[..., : self.head_dim]
        return self.o_proj(attended.reshape(-1, self.num_heads * self.head_dim))


class SEAMQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(
        self,
        config,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, cache_config=cache_config, quant_config=quant_config, prefix=prefix)
        self.cross_norm = RMSNorm(config.hidden_size, eps=float(config.rms_norm_eps))
        self.cross = _SEAMCrossAttention(self, config, cache_config, quant_config, prefix=f"{prefix}.cross")
        self.cross_gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.cross_gate_max = float(getattr(config, "seam_cross_gate_max", 1.0))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        encoder_memory: torch.Tensor | None = None,
        encoder_value_memory: torch.Tensor | None = None,
        source_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # The AFMR cross route is inserted after self-attention residual
        # addition and before post-attention normalization/MLP.
        combined = residual + hidden_states
        cross = self.cross(self.cross_norm(combined), encoder_memory, encoder_value_memory, source_bias)
        hidden_states = hidden_states + self.cross_gate_max * torch.sigmoid(self.cross_gate).to(cross.dtype) * cross
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class SEAMQwen3Model(Qwen2Model):
    """Qwen3 decoder stack with AFMR cross-attention in each layer."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=SEAMQwen3DecoderLayer,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_memory: torch.Tensor | None = None,
        encoder_value_memory: torch.Tensor | None = None,
        source_bias: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if not get_pp_group().is_first_rank or not get_pp_group().is_last_rank:
            raise RuntimeError("SEAM's initial vLLM adapter supports a single pipeline stage")
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        elif input_ids is not None:
            hidden_states = self.embed_input_ids(input_ids)
        else:
            raise ValueError("SEAM requires input_ids or inputs_embeds")
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                encoder_memory,
                encoder_value_memory,
                source_bias,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class SEAMForQwen3(Qwen3ForCausalLM, SupportsMultiModal):
    """vLLM model for Qwen3 decoders and AFMR-trained checkpoints.

    This first adapter deliberately fails outside TP=1/PP=1. The exact copy
    mixture is applied before vLLM sampling; generation-only fallback is not
    allowed when copy is configured.
    """

    hf_to_vllm_mapper = Qwen3Model.hf_to_vllm_mapper
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        validate_vllm_config(vllm_config)
        if get_tensor_model_parallel_world_size() != 1 or get_pp_group().world_size != 1:
            raise RuntimeError(
                "SEAMForQwen3 currently supports tensor_parallel_size=1 and pipeline_parallel_size=1 only"
            )

        hf_config = vllm_config.model_config.hf_config
        config = hf_config.get_text_config()
        validate_decoder_config(config)
        self.config = config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model = SEAMQwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        config_path = _config_path(hf_config, "seam_config_path")
        checkpoint_path = _config_path(hf_config, "seam_checkpoint_path")
        self.seam_config_path = config_path
        self.seam_checkpoint_path = checkpoint_path
        self.seam_config = load_config(config_path, train_only=True)
        if int(self.seam_config["decoder"].get("cross_attention_every", 1)) != 1:
            raise NotImplementedError("SEAM vLLM currently requires cross_attention_every=1")
        self._validate_local_backbones(config_path)
        validate_tokenizer_config(vllm_config, self.seam_config["model"]["decoder_name"])
        expected_dtype = str(
            self.seam_config["model"].get("compute_dtype", self.seam_config["model"].get("dtype", "float32"))
        )
        validate_compute_dtype(vllm_config, expected_dtype)
        cross_gate_max = float(self.seam_config["decoder"].get("cross_gate_max", 1.0))
        for layer in self.model.layers:
            layer.cross_gate_max = cross_gate_max

        model_config = dict(self.seam_config["model"])
        model_config["gradient_checkpointing"] = False
        # Match native CUDA evaluation: load the encoder at inference precision
        # instead of first allocating FP32 weights and then converting them.
        model_config["dtype"] = expected_dtype
        encoder_config = dict(self.seam_config)
        encoder_config["model"] = model_config
        self.encoder = build_encoder(encoder_config)
        self.bridge = AdaptiveFullMemoryResidualBridge(
            self.encoder.hidden_size, int(config.hidden_size), self.seam_config["architecture"]
        )
        copy_config = self.seam_config["decoder"].get("grounded_copy", {})
        self.copy_head = (
            GroundedCopyHead(
                int(config.hidden_size), int(copy_config.get("key_dim", 128)), float(copy_config.get("gate_init", 0.05))
            )
            if copy_config.get("enabled", False)
            else None
        )
        dtype = vllm_config.model_config.dtype
        device = vllm_config.device_config.device
        self.encoder.to(device=device, dtype=dtype).eval()
        # AFMR evaluates its bridge and copy route in FP32 even when the
        # pretrained encoder/decoder use BF16. Keep that numerical contract;
        # only the memory consumed by decoder cross-attention is cast below.
        self.bridge.to(device=device, dtype=torch.float32).eval()
        if self.copy_head is not None:
            self.copy_head.to(device=device, dtype=torch.float32).eval()

        self.register_buffer("decoder_prompt_ids", self._build_decoder_prompt(), persistent=False)
        self._completed_copy_states: list[CopyState | None] = []
        self._pending_copy_state: CopyState | None = None
        self._has_copy_requests = False
        self._copy_batch_cache = CopyBatchCache()
        self._copy_logits_mixer = CopyLogitsMixer()
        self._actor_update = None
        self._actor_version = -1
        self._initial_weights_loaded = False

    def _validate_local_backbones(self, config_path: Path) -> None:
        for name in ("encoder_name", "decoder_name"):
            value = resolve_path(str(self.seam_config["model"][name]), self.seam_config)
            if not value.exists():
                raise FileNotFoundError(f"SEAM requires local {name}: {value.resolve()}")
            self.seam_config["model"][name] = str(value)

    def _build_decoder_prompt(self) -> torch.Tensor:
        from transformers import AutoTokenizer

        model = self.seam_config["model"]
        data = self.seam_config["data"]
        tokenizer = AutoTokenizer.from_pretrained(
            str(model["decoder_name"]),
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            local_files_only=True,
        )
        prompt_ids = configured_decoder_prompt_ids(tokenizer, data)
        return torch.tensor(prompt_ids, dtype=torch.long)

    def _restore_afmr_checkpoint(self, checkpoint_path: Path) -> set[str]:
        # Training checkpoints also contain optimizer states. Memory-map their
        # storages so inference only touches the model tensors it restores.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
        expected_spec = architecture_spec(self.seam_config)
        actual_spec = checkpoint.get("architecture_spec")
        if actual_spec != expected_spec:
            raise ValueError(
                "Checkpoint architecture_spec does not match the active SEAM config; "
                "load the checkpoint with its matching resolved_config.yaml"
            )
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError(f"Not an AFMR training checkpoint: {checkpoint_path}")
        expected_source_keys = expected_qwen3_decoder_checkpoint_keys(
            int(self.config.num_hidden_layers), bool(self.config.tie_word_embeddings)
        )
        missing_source_keys = sorted(expected_source_keys - set(state))
        if missing_source_keys:
            raise ValueError(
                f"AFMR checkpoint is missing required Qwen3 decoder tensors; missing={missing_source_keys[:8]}"
            )

        encoder_state = {
            key.removeprefix("encoder."): value for key, value in state.items() if key.startswith("encoder.")
        }
        bridge_state = {key.removeprefix("bridge."): value for key, value in state.items() if key.startswith("bridge.")}
        if not encoder_state or not bridge_state:
            raise ValueError("AFMR checkpoint is missing encoder or bridge weights")
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.bridge.load_state_dict(bridge_state, strict=True)

        copy_state = {
            key.removeprefix("decoder.grounded_copy."): value
            for key, value in state.items()
            if key.startswith("decoder.grounded_copy.")
        }
        if self.copy_head is None:
            if copy_state:
                raise ValueError("Checkpoint has grounded-copy weights but the resolved config disables copy")
        else:
            if not copy_state:
                raise ValueError("Grounded copy is enabled but its checkpoint weights are missing")
            self.copy_head.load_state_dict(copy_state, strict=True)

        decoder_base: list[tuple[str, torch.Tensor]] = []
        cross_values: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key.startswith("decoder.backbone."):
                suffix = key.removeprefix("decoder.backbone.").replace(".base.", ".")
                if suffix.startswith("layers.") and any(
                    marker in suffix for marker in (".cross.", ".cross_norm.", ".cross_gate")
                ):
                    cross_values[f"model.{suffix}"] = value
                else:
                    decoder_base.append((f"model.{suffix}", value))
            elif key.startswith("decoder.lm_head."):
                decoder_base.append((normalize_decoder_lm_head_key(key), value))

        if not decoder_base:
            raise ValueError("AFMR checkpoint is missing the trained decoder backbone")
        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."] if self.config.tie_word_embeddings else None)
        loaded = loader.load_weights(decoder_base)
        if not loaded:
            raise RuntimeError("vLLM did not load any trained decoder backbone weights from AFMR checkpoint")

        params = dict(self.named_parameters())
        expected_decoder_parameters = {
            name
            for name in params
            if (name.startswith("model.") or name.startswith("lm_head."))
            and not any(marker in name for marker in (".cross.", ".cross_norm.", ".cross_gate"))
        }
        if self.config.tie_word_embeddings:
            expected_decoder_parameters = {
                name for name in expected_decoder_parameters if not name.startswith("lm_head.")
            }
        missing_decoder_parameters = missing_parameter_names(expected_decoder_parameters, loaded)
        if missing_decoder_parameters:
            raise RuntimeError(
                "vLLM did not restore every trained Qwen3 decoder parameter from the AFMR checkpoint; "
                f"missing={missing_decoder_parameters[:8]}"
            )
        expected_cross = {
            name
            for name in params
            if name.startswith("model.layers.")
            and any(f".cross{suffix}" in name for suffix in (".", "_norm.", "_gate"))
        }
        if set(cross_values) != expected_cross:
            missing = sorted(expected_cross - set(cross_values))
            unexpected = sorted(set(cross_values) - expected_cross)
            raise ValueError(
                "AFMR cross-attention checkpoint mapping is incomplete; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        with torch.no_grad():
            for name, value in cross_values.items():
                parameter = params[name]
                if parameter.shape != value.shape:
                    raise ValueError(
                        f"SEAM cross-attention shape mismatch for {name}: checkpoint={tuple(value.shape)}, "
                        f"model={tuple(parameter.shape)}"
                    )
                parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

        # vLLM's DefaultModelLoader checks every named parameter, including
        # encoder/bridge/copy/cross modules loaded outside AutoWeightsLoader.
        restored = set(loaded) | set(cross_values)
        for prefix, module in (("encoder", self.encoder), ("bridge", self.bridge), ("copy_head", self.copy_head)):
            if module is not None:
                restored.update(f"{prefix}.{name}" for name, _ in module.named_parameters())
        missing = missing_parameter_names(params, restored)
        if missing:
            raise RuntimeError(f"SEAM checkpoint did not restore all model parameters: {missing[:8]}")
        return restored & set(params)

    def get_language_model(self) -> nn.Module:
        return self.model

    def get_model_state_cls(self):
        from .state import SEAMModelState

        return SEAMModelState

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def drain_copy_states(self) -> list[CopyState | None]:
        result = self._completed_copy_states
        self._completed_copy_states = []
        return result

    @torch.inference_mode()
    def embed_multimodal(self, encoder_input_ids, **kwargs) -> list[torch.Tensor]:
        if getattr(self, "_actor_update", None) is not None:
            raise RuntimeError("SEAM actor update is incomplete; generation is blocked")
        outputs = []
        device = self.decoder_prompt_ids.device
        budget = getattr(self.vllm_config.model_config.hf_config, "seam_encoder_max_batch_tokens", 8192)
        ids = _as_2d(encoder_input_ids, "encoder_input_ids", device, torch.long)
        fields = dict(kwargs)
        if fields.get("encoder_attention_mask") is None:
            pad = getattr(getattr(self.encoder.model, "config", None), "pad_token_id", None)
            fields["encoder_attention_mask"] = (
                ids.ne(int(pad)) if pad is not None else torch.ones_like(ids, dtype=torch.bool)
            )
        attention_mask = _as_2d(fields["encoder_attention_mask"], "encoder_attention_mask", device, torch.bool)
        # vLLM invokes this once for a whole encoder batch. Transfer lengths to
        # Python once, before microbatching, instead of synchronizing CUDA once
        # per chunk when slicing the returned source memories.
        source_lengths = attention_mask.sum(dim=-1).tolist()
        offset = 0
        for fields in encoder_batches(ids, fields, device, budget):
            count = fields["encoder_input_ids"].shape[0]
            fields["_source_lengths"] = source_lengths[offset : offset + count]
            offset += count
            outputs.extend(self._embed_multimodal_batch(**fields))
        return outputs

    def _embed_multimodal_batch(
        self,
        encoder_input_ids,
        encoder_attention_mask=None,
        source_content_mask=None,
        copy_token_ids=None,
        copy_token_mask=None,
        copy_encoder_indices=None,
        copy_token_indices=None,
        copy_alignment_weights=None,
        requested_output_tokens=None,
        bridge_prompt_ids=None,
        bridge_prompt_mask=None,
        _source_lengths=None,
        **kwargs,
    ) -> list[torch.Tensor]:
        device = self.decoder_prompt_ids.device
        ids = _as_2d(encoder_input_ids, "encoder_input_ids", device, torch.long)
        if encoder_attention_mask is None:
            pad = getattr(getattr(self.encoder.model, "config", None), "pad_token_id", None)
            attention_mask = ids.ne(int(pad)) if pad is not None else torch.ones_like(ids, dtype=torch.bool)
        else:
            attention_mask = _as_2d(encoder_attention_mask, "encoder_attention_mask", device, torch.bool)
        if source_content_mask is None:
            raise ValueError("SEAM source processor must provide source_content_mask")
        content_mask = _as_2d(source_content_mask, "source_content_mask", device, torch.bool)
        if ids.shape != attention_mask.shape or ids.shape != content_mask.shape:
            raise ValueError("SEAM encoder ids and masks must have matching shapes")

        encoder_state = self.encoder(ids, attention_mask, content_mask)
        batch = ids.shape[0]
        prompt_ids, prompt_mask = bridge_prompt_batch(
            self.decoder_prompt_ids, bridge_prompt_ids, bridge_prompt_mask, batch, device
        )
        prompt_embeddings = self.model.embed_tokens(prompt_ids)
        if requested_output_tokens is None:
            output_budget = torch.full(
                (batch,),
                int(self.seam_config["generation"].get("max_new_tokens", 256)),
                device=device,
                dtype=torch.float32,
            )
        else:
            output_budget = _as_2d(requested_output_tokens, "requested_output_tokens", device, torch.float32).reshape(
                -1
            )
            if output_budget.numel() != batch:
                raise ValueError("SEAM requires one requested output budget per source")
        bridge = self.bridge(encoder_state, prompt_embeddings, prompt_mask, output_budget)
        # The bridge result contains the memory, masks and prior needed below.
        # Release encoder layer taps and prompt intermediates before copy-state
        # construction and packing, which otherwise overlap with those tensors.
        del encoder_state, prompt_embeddings, prompt_ids, prompt_mask, output_budget
        decoder_dtype = self.model.embed_tokens.weight.dtype
        memory = bridge.memory.to(decoder_dtype)
        value_memory = None if bridge.value_memory is None else bridge.value_memory.to(decoder_dtype)

        copy_batch: list[CopyState | None] = [None] * batch
        if self.copy_head is not None:
            required = {
                "copy_token_ids": copy_token_ids,
                "copy_token_mask": copy_token_mask,
                "copy_encoder_indices": copy_encoder_indices,
                "copy_token_indices": copy_token_indices,
                "copy_alignment_weights": copy_alignment_weights,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"SEAM grounded copy is enabled but source alignment is missing: {missing}")
            prepared = self.copy_head.prepare(
                value_memory if value_memory is not None else memory,
                bridge.source_bias,
                bridge.content_mask,
                self.model.embed_tokens,
                copy_token_ids=_as_2d(copy_token_ids, "copy_token_ids", device, torch.long),
                copy_token_mask=_as_2d(copy_token_mask, "copy_token_mask", device, torch.bool),
                copy_encoder_indices=_as_2d(copy_encoder_indices, "copy_encoder_indices", device, torch.long),
                copy_token_indices=_as_2d(copy_token_indices, "copy_token_indices", device, torch.long),
                copy_alignment_weights=_as_2d(copy_alignment_weights, "copy_alignment_weights", device, torch.float32),
            )
            row_index = torch.arange(batch, device=device)
            copy_batch = [prepared.index_select(row_index[i : i + 1]) for i in range(batch)]

        packed = pack_bridge_memory(memory, value_memory, bridge.source_bias)
        lengths = _source_lengths if _source_lengths is not None else attention_mask.sum(dim=-1).tolist()
        if len(lengths) != batch:
            raise ValueError("SEAM source length count differs from the encoder batch size")
        self._completed_copy_states.extend(copy_batch)
        return [packed[index, : int(length)].contiguous() for index, length in enumerate(lengths)]

    _stack_copy_states = staticmethod(stack_copy_states)

    def clear_copy_batch_cache(self) -> None:
        self._copy_batch_cache.clear()
        self._copy_logits_mixer.clear()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        copy_states: list[CopyState | None] | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if getattr(self, "_actor_update", None) is not None:
            raise RuntimeError("SEAM actor update is incomplete; generation is blocked")
        memory = value_memory = source_bias = None
        if encoder_outputs:
            packed = torch.cat(encoder_outputs, dim=0)
            hidden = int(self.config.hidden_size)
            memory, value_memory, source_bias = unpack_bridge_memory(packed, hidden)
        self._pending_copy_state = self._copy_batch_cache.get(copy_states or [])
        self._has_copy_requests = bool(copy_states) and self.copy_head is not None
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            encoder_memory=memory,
            encoder_value_memory=value_memory,
            source_bias=source_bias,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        if getattr(self, "_actor_update", None) is not None:
            raise RuntimeError("SEAM actor update is incomplete; generation is blocked")
        logits = self.logits_processor(self.lm_head, hidden_states)
        copy_state = self._pending_copy_state
        self._pending_copy_state = None
        if self._has_copy_requests and copy_state is None and hidden_states.shape[0]:
            raise RuntimeError("SEAM copy state was already consumed; prompt_logprobs is not supported")
        if logits is None or self.copy_head is None or copy_state is None:
            return logits
        if logits.shape[0] != copy_state.token_ids.shape[0]:
            raise RuntimeError(
                "vLLM selected a different number/order of logits than SEAM request copy states; "
                "exact copy-mixed decoding cannot proceed"
            )
        return self._copy_logits_mixer.mix(self.copy_head, hidden_states, logits, copy_state)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._initial_weights_loaded:
            raise RuntimeError(
                "Use SEAMLLM.sync_weights for full-actor runtime updates; HF decoder-only reload is unsafe"
            )
        loader = AutoWeightsLoader(self, skip_prefixes=["lm_head."] if self.config.tie_word_embeddings else None)
        loader.load_weights(weights)
        # Report custom-module parameters too: vLLM checks the entire model,
        # not just the original HF decoder loaded by AutoWeightsLoader.
        restored = self._restore_afmr_checkpoint(self.seam_checkpoint_path)
        self._initial_weights_loaded = True
        return restored


def _as_token_list(values: Any) -> list[int]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (tuple, list)) and len(values) == 1 and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(value) for value in values]
