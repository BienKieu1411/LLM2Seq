from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .memory import LayerSourceKV, PreparedDraftSource, ordered_masked_pool, sinusoidal_positions


@dataclass(frozen=True)
class DrafterShape:
    encoder_width: int
    decoder_width: int
    width: int
    layers: int
    heads: int
    ffn_width: int
    max_block_size: int
    target_feature_window: int
    max_source_tokens: int


class DraftBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_width: int):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.self_norm = nn.LayerNorm(width)
        self.self_qkv = nn.Linear(width, width * 3)
        self.self_output = nn.Linear(width, width)
        self.cross_query_norm = nn.LayerNorm(width)
        self.cross_query = nn.Linear(width, width)
        self.cross_key = nn.Linear(width, width)
        self.cross_value = nn.Linear(width, width)
        self.cross_output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_width),
            nn.GELU(),
            nn.Linear(ffn_width, width),
        )

    def _heads(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = states.shape
        return states.view(batch_size, sequence_length, self.heads, self.head_width).transpose(1, 2)

    def project_source(self, source: torch.Tensor) -> LayerSourceKV:
        key = self._heads(self.cross_key(source))
        value = self._heads(self.cross_value(source))
        return LayerSourceKV(key=key, value=value)

    def project_history(self, history: torch.Tensor) -> LayerSourceKV:
        key = self._heads(self.cross_key(history))
        value = self._heads(self.cross_value(history))
        return LayerSourceKV(key=key, value=value)

    def forward(
        self,
        slots: torch.Tensor,
        source_kv: LayerSourceKV,
        source_mask: torch.Tensor,
        history_kv: LayerSourceKV,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, slot_count, _ = slots.shape
        normalized = self.self_norm(slots)
        query, key, value = self.self_qkv(normalized).chunk(3, dim=-1)
        query = self._heads(query)
        key = self._heads(key)
        value = self._heads(value)
        self_output = F.scaled_dot_product_attention(query, key, value, is_causal=False)
        self_output = self_output.transpose(1, 2).reshape(batch_size, slot_count, self.width)
        slots = slots + self.self_output(self_output)

        query = self._heads(self.cross_query(self.cross_query_norm(slots)))
        key = torch.cat([source_kv.key, history_kv.key], dim=-2)
        value = torch.cat([source_kv.value, history_kv.value], dim=-2)
        memory_mask = torch.cat([source_mask.bool(), history_mask.bool()], dim=-1)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=memory_mask[:, None, None, :],
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, slot_count, self.width)
        slots = slots + self.cross_output(attended)
        return slots + self.ffn(self.ffn_norm(slots))


class ParallelBlockDrafter(nn.Module):
    """One-pass proposal network over learned future-token slots."""

    def __init__(self, shape: DrafterShape):
        super().__init__()
        if shape.width % shape.heads:
            raise ValueError("draft width must be divisible by the number of heads")
        self.shape = shape
        self.source_projection = nn.Linear(shape.encoder_width, shape.width)
        self.history_projection = nn.Linear(shape.decoder_width, shape.width)
        self.pending_projection = nn.Linear(shape.decoder_width, shape.width)
        self.slot_embedding = nn.Embedding(shape.max_block_size, shape.width)
        self.source_type = nn.Parameter(torch.zeros(shape.width))
        self.history_type = nn.Parameter(torch.zeros(shape.width))
        self.blocks = nn.ModuleList(
            [DraftBlock(shape.width, shape.heads, shape.ffn_width) for _ in range(shape.layers)]
        )
        self.output_norm = nn.LayerNorm(shape.width)
        self.output_projection = nn.Linear(shape.width, shape.decoder_width)
        self.logit_softcap: Optional[float] = None

    @classmethod
    def from_config(cls, adapter, config: Dict[str, Any]) -> "ParallelBlockDrafter":
        draft = config["draft"]
        drafter = cls(
            DrafterShape(
                encoder_width=adapter.encoder_width,
                decoder_width=adapter.decoder_width,
                width=int(draft["width"]),
                layers=int(draft["layers"]),
                heads=int(draft["heads"]),
                ffn_width=int(draft["ffn_width"]),
                max_block_size=int(draft["max_block_size"]),
                target_feature_window=int(draft["target_feature_window"]),
                max_source_tokens=int(draft["max_source_tokens"]),
            )
        )
        drafter.logit_softcap = getattr(getattr(adapter.model.config, "decoder", None), "final_logit_softcapping", None)
        return drafter

    @property
    def max_block_size(self) -> int:
        return self.shape.max_block_size

    def prepare_source(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        fingerprint: str,
        mode: str = "full",
        budget: int = 256,
    ) -> PreparedDraftSource:
        if mode == "ordered_pool":
            memory, memory_mask, positions = ordered_masked_pool(hidden_states, attention_mask, budget)
            memory = memory.to(dtype=self.source_projection.weight.dtype)
            memory = self.source_projection(memory) + self.source_type
            memory = memory + sinusoidal_positions(positions, self.shape.width, memory.dtype)
        elif mode == "full":
            memory_mask = attention_mask.bool()
            memory = hidden_states.to(dtype=self.source_projection.weight.dtype)
            memory = self.source_projection(memory) + self.source_type
        else:
            raise ValueError(f"Unknown draft source-memory mode: {mode}")
        layer_kv = [block.project_source(memory) for block in self.blocks]
        return PreparedDraftSource(
            layer_kv=layer_kv,
            mask=memory_mask,
            fingerprint=fingerprint,
            source_tokens=int(attention_mask[0].sum().item()),
            retained_tokens=int(memory_mask[0].sum().item()),
        )

    def forward(
        self,
        pending_ids: torch.Tensor,
        accepted_features: torch.Tensor,
        source: PreparedDraftSource,
        input_embedding: nn.Module,
        output_head: nn.Module,
        block_size: Optional[int] = None,
    ) -> torch.Tensor:
        if pending_ids.ndim == 1:
            pending_ids = pending_ids.unsqueeze(1)
        if pending_ids.ndim != 2 or pending_ids.shape[1] != 1:
            raise ValueError("pending_ids must have shape [batch] or [batch, 1]")
        slot_count = self.shape.max_block_size if block_size is None else int(block_size)
        if not 1 <= slot_count <= self.shape.max_block_size:
            raise ValueError("block_size must be in [1, draft.max_block_size]")
        if accepted_features.ndim != 3 or accepted_features.shape[0] != pending_ids.shape[0]:
            raise ValueError("accepted_features must have shape [batch, history, decoder_width]")

        device = self.pending_projection.weight.device
        pending_ids = pending_ids.to(device=device, dtype=torch.long)
        accepted_features = accepted_features.to(
            device=device,
            dtype=self.history_projection.weight.dtype,
        )
        token_vectors = input_embedding(pending_ids).squeeze(1)
        token_vectors = token_vectors.to(dtype=self.pending_projection.weight.dtype)
        pending = self.pending_projection(token_vectors).unsqueeze(1)
        slot_positions = self.slot_embedding(torch.arange(slot_count, device=device)).unsqueeze(0)
        slots = pending + slot_positions

        history = self.history_projection(accepted_features) + self.history_type
        history_kv = [block.project_history(history) for block in self.blocks]
        history_mask = torch.ones(history.shape[:2], dtype=torch.bool, device=device)
        for index, block in enumerate(self.blocks):
            slots = block(
                slots,
                source.layer_kv[index],
                source.mask,
                history_kv[index],
                history_mask,
            )
        slots = self.output_projection(self.output_norm(slots))
        head_dtype = getattr(output_head.weight, "dtype", slots.dtype)
        logits = output_head(slots.to(dtype=head_dtype))
        softcap = self.logit_softcap
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def checkpoint_payload(self, target_manifest: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "t5gemma2_parallel_block_drafter",
            "format_version": 1,
            "shape": asdict(self.shape),
            "target_manifest": target_manifest,
            "state_dict": self.state_dict(),
        }


def load_drafter(
    checkpoint_path: str | Path,
    adapter,
    config: Dict[str, Any],
) -> ParallelBlockDrafter:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Drafter checkpoint does not exist: {checkpoint_path}")
    torch_version_type = getattr(getattr(torch, "torch_version", None), "TorchVersion", None)
    allowed_globals = [torch_version_type] if torch_version_type is not None else []
    with torch.serialization.safe_globals(allowed_globals):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "t5gemma2_parallel_block_drafter":
        raise ValueError("Unsupported drafter checkpoint format")
    expected_manifest = adapter.target_manifest()
    actual_manifest = payload.get("target_manifest", {})
    identity_fields = ("target_fingerprint", "config_sha256", "tokenizer_sha256")
    if any(actual_manifest.get(field) != expected_manifest.get(field) for field in identity_fields):
        raise ValueError(
            "Drafter checkpoint was trained against a different T5Gemma2 target. "
            "Use the same local checkpoint or set a stable target.identity for an identical copied checkpoint."
        )
    drafter = ParallelBlockDrafter.from_config(adapter, config)
    saved_shape = payload.get("shape", {})
    if saved_shape != asdict(drafter.shape):
        raise ValueError(f"Drafter architecture mismatch: checkpoint={saved_shape}, config={asdict(drafter.shape)}")
    drafter.load_state_dict(payload["state_dict"], strict=True)
    drafter.to(device=adapter.device, dtype=adapter.dtype)
    drafter.eval()
    return drafter
