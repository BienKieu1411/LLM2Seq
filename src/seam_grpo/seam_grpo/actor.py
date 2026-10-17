"""Transformers-compatible actor wrapper for the complete SEAM policy."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.checkpoint import checkpoint
from transformers import GenerationConfig, PretrainedConfig, PreTrainedModel

from .policy import Policy
from .rollout import select_rows


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

    def _score_candidates(self, batch: dict[str, Any], temperature: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(batch["input_ids"].shape[0])
        logps: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for start in range(0, batch_size, self.score_batch_size):
            stop = min(batch_size, start + self.score_batch_size)
            candidate_batch = select_rows(batch, range(start, stop))

            def score(batch_arg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
                return self.policy(batch_arg, temperature=temperature, return_entropy=True)

            if torch.is_grad_enabled():
                logp_chunk, entropy_chunk = checkpoint(score, candidate_batch, use_reentrant=False)
            else:
                logp_chunk, entropy_chunk = score(candidate_batch)
            logps.append(logp_chunk)
            entropies.append(entropy_chunk.detach())
        return torch.cat(logps, dim=0), torch.cat(entropies, dim=0)

    def _score_gold(
        self, batch: dict[str, Any], gold_selector: torch.Tensor | None, logps: torch.Tensor
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
        for offset in range(0, selected.numel(), self.score_batch_size):
            indices = selected[offset : offset + self.score_batch_size]
            gold_batch = select_rows(batch, indices)
            labels = gold_batch.get("labels")
            if labels is None:
                raise ValueError("Gold CE requires collator labels")
            token_count = labels[:, 1:].ne(-100).sum()

            def score_ce(batch_arg: dict[str, Any]) -> torch.Tensor:
                return self.policy(batch_arg, temperature=1.0, ce=True)

            ce_mean = (
                checkpoint(score_ce, gold_batch, use_reentrant=False)
                if torch.is_grad_enabled()
                else score_ce(gold_batch)
            )
            ce_sum = ce_sum + ce_mean * token_count.to(dtype=ce_mean.dtype)
        return ce_sum

    def forward(
        self,
        batch: dict[str, Any],
        *,
        temperature: float,
        gold_selector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logps, entropies = self._score_candidates(batch, temperature)
        ce_sum = self._score_gold(batch, gold_selector, logps)
        return logps, entropies.detach(), ce_sum
