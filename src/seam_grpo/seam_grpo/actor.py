"""Transformers-compatible actor wrapper for the complete SEAM policy."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

from .policy import Policy
from .rollout import select_bridge_state, select_rows


def find_unique_source_indices(batch: dict[str, Any]) -> tuple[list[int], torch.Tensor]:
    input_ids = batch["input_ids"]
    prompt_ids = batch.get("decoder_prompt_ids")
    batch_size = int(input_ids.shape[0])
    if batch_size <= 1:
        return [0], torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

    unique_rows: list[int] = []
    inverse = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)
    for i in range(batch_size):
        matched = -1
        for u_idx, u in enumerate(unique_rows):
            if torch.equal(input_ids[i], input_ids[u]):
                if prompt_ids is None or torch.equal(prompt_ids[i], prompt_ids[u]):
                    if "source_content_mask" not in batch or torch.equal(
                        batch["source_content_mask"][i], batch["source_content_mask"][u]
                    ):
                        matched = u_idx
                        break
        if matched >= 0:
            inverse[i] = matched
        else:
            inverse[i] = len(unique_rows)
            unique_rows.append(i)
    return unique_rows, inverse


class SEAMActor(PreTrainedModel):
    """Expose a SEAM model as one trainable actor to TRL and DDP.

    The complete AFMR model remains registered under ``policy.seam``.  Action
    scoring is split across candidate rows inside one actor forward so DDP sees
    a single forward/backward pair for each training microbatch.
    """

    config_class = PretrainedConfig
    main_input_name = "batch"
    supports_gradient_checkpointing = True

    def __init__(self, seam: torch.nn.Module, chunk_size: int = 32, score_batch_size: int = 1):
        if score_batch_size <= 0:
            raise ValueError("score_batch_size must be positive")
        decoder_config = seam.decoder.config
        config = PretrainedConfig(
            vocab_size=getattr(decoder_config, "vocab_size", None),
            hidden_size=getattr(decoder_config, "hidden_size", None),
            bos_token_id=getattr(decoder_config, "bos_token_id", None),
            eos_token_id=getattr(decoder_config, "eos_token_id", None),
            pad_token_id=getattr(decoder_config, "pad_token_id", None),
            is_encoder_decoder=True,
            tie_word_embeddings=bool(getattr(decoder_config, "tie_word_embeddings", False)),
        )
        config.model_type = "custom"
        config._name_or_path = "seam-local"
        config.architectures = [type(self).__name__]

        super().__init__(config)
        self.policy = Policy(seam, chunk_size=chunk_size)
        self.score_batch_size = int(score_batch_size)

        try:
            self.generation_config = GenerationConfig.from_model_config(decoder_config)
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            self.generation_config = GenerationConfig()
        self.generation_config.is_encoder_decoder = True
        # Keep the decoder's special IDs initially.  The trainer aligns EOS
        # with the active tokenizer after constructing the processing class.
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"):
            value = getattr(decoder_config, name, None)
            if value is not None:
                setattr(self.generation_config, name, value)

    @property
    def seam(self) -> torch.nn.Module:
        return self.policy.seam

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.seam.decoder.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.seam.decoder.lm_head

    def train(self, mode: bool = True) -> "SEAMActor":
        # Keep the wrapper's train/eval state visible to Trainer while ensuring
        # both rollout and policy scoring use the same dropout-free model.
        super().train(mode)
        self.seam.eval()
        return self

    def _encode_unique_sources(self, batch: dict[str, Any]) -> tuple[list[Any], torch.Tensor]:
        unique_indices, inverse_indices = find_unique_source_indices(batch)
        unique_bridges: list[Any] = []
        for u in unique_indices:
            single_batch = select_rows(batch, [u])
            if torch.is_grad_enabled():
                bridge_u = checkpoint(self.policy.encode_source, single_batch, use_reentrant=False)
            else:
                bridge_u = self.policy.encode_source(single_batch)
            unique_bridges.append(bridge_u)
        return unique_bridges, inverse_indices

    def _score_candidates(
        self,
        batch: dict[str, Any],
        temperature: float,
        unique_bridges: list[Any],
        inverse_indices: torch.Tensor,
        compute_entropy_with_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch["input_ids"].shape[0])
        action_lengths = batch["action_mask"].bool().sum(dim=-1).tolist()
        inverse_list = inverse_indices.detach().cpu().tolist()
        logps: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        start = 0
        while start < batch_size:
            u_idx = int(inverse_list[start])
            act_len = int(action_lengths[start])
            stop = start + 1
            while (
                stop < min(batch_size, start + self.score_batch_size)
                and int(inverse_list[stop]) == u_idx
                and int(action_lengths[stop]) == act_len
            ):
                stop += 1
            candidate_batch = select_rows(batch, range(start, stop))
            zeros = torch.zeros(stop - start, dtype=torch.long, device=batch["input_ids"].device)
            slice_bridge = select_bridge_state(unique_bridges[u_idx], zeros)

            def score(batch_arg: dict[str, Any], bridge_arg: Any) -> tuple[torch.Tensor, torch.Tensor]:
                return self.policy(
                    batch_arg,
                    temperature=temperature,
                    return_entropy=True,
                    bridge=bridge_arg,
                    compute_entropy_with_grad=compute_entropy_with_grad,
                )

            if torch.is_grad_enabled():
                logp_chunk, entropy_chunk = checkpoint(score, candidate_batch, slice_bridge, use_reentrant=False)
            else:
                logp_chunk, entropy_chunk = score(candidate_batch, slice_bridge)
            logps.append(logp_chunk)
            entropies.append(entropy_chunk if compute_entropy_with_grad else entropy_chunk.detach())
            start = stop
        return torch.cat(logps, dim=0), torch.cat(entropies, dim=0)

    def _score_gold(
        self,
        batch: dict[str, Any],
        gold_selector: torch.Tensor | None,
        logps: torch.Tensor,
        unique_bridges: list[Any],
        inverse_indices: torch.Tensor,
    ) -> torch.Tensor:
        zero = logps.sum() * 0.0
        if gold_selector is None:
            return zero
        if not isinstance(gold_selector, torch.Tensor) or gold_selector.ndim != 1 or gold_selector.dtype != torch.bool:
            raise ValueError("gold_selector must be a boolean tensor with one entry per batch row")
        if gold_selector.shape[0] != batch["input_ids"].shape[0]:
            raise ValueError("gold_selector must have one entry per batch row")
        selector = gold_selector.to(device=batch["input_ids"].device, dtype=torch.bool)
        selected = selector.nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            return zero
        if "gold_decoder_input_ids" not in batch or "gold_decoder_attention_mask" not in batch:
            raise ValueError("Gold CE requires preserved gold_decoder_input_ids and gold_decoder_attention_mask")

        ce_sum = zero
        for offset in range(selected.numel()):
            indices = selected[offset : offset + 1]
            gold_batch = select_rows(batch, indices)
            labels = gold_batch.get("labels")
            if labels is None:
                raise ValueError("Gold CE requires collator labels")
            token_count = labels[:, 1:].ne(-100).sum()
            u_idx = int(inverse_indices[indices[0]].item())
            zeros = torch.zeros(1, dtype=torch.long, device=batch["input_ids"].device)
            slice_bridge = select_bridge_state(unique_bridges[u_idx], zeros)

            def score_ce(batch_arg: dict[str, Any], bridge_arg: Any) -> torch.Tensor:
                return self.policy(batch_arg, temperature=1.0, ce=True, bridge=bridge_arg)

            ce_mean = (
                checkpoint(score_ce, gold_batch, slice_bridge, use_reentrant=False)
                if torch.is_grad_enabled()
                else score_ce(gold_batch, slice_bridge)
            )
            ce_sum = ce_sum + ce_mean * token_count.to(dtype=ce_mean.dtype)
        return ce_sum

    def forward(
        self,
        batch: dict[str, Any],
        *,
        temperature: float,
        gold_selector: torch.Tensor | None = None,
        compute_entropy_with_grad: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unique_bridges, inverse_indices = self._encode_unique_sources(batch)
        logps, entropies = self._score_candidates(
            batch,
            temperature,
            unique_bridges,
            inverse_indices,
            compute_entropy_with_grad=compute_entropy_with_grad,
        )
        ce_sum = self._score_gold(batch, gold_selector, logps, unique_bridges, inverse_indices)
        return logps, entropies if compute_entropy_with_grad else entropies.detach(), ce_sum
