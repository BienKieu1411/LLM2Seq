"""Differentiable action scoring for the AFMR SEAM policy."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
from torch.utils.checkpoint import checkpoint

from .context import validate_context
from .rollout import trim_source_batch


class Policy(nn.Module):
    """Wrap an ``EviSeqAFMR`` model with CE and sampled-action objectives.

    Action scores are returned as per-token log-probabilities with shape
    ``[batch, action_tokens]``. The caller applies ``action_mask`` when forming
    sequence-level rewards or losses.
    """

    def __init__(self, seam: nn.Module, chunk_size: int = 32):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.seam = seam
        self.chunk_size = int(chunk_size)

    @staticmethod
    def _output_budget(batch: dict[str, Any], seam: nn.Module, batch_size: int, device: torch.device):
        budget = batch.get("output_budget")
        if budget is None:
            default = int(getattr(seam, "config", {}).get("generation", {}).get("max_new_tokens", 256))
            return torch.full((batch_size,), default, dtype=torch.float32, device=device)
        if isinstance(budget, torch.Tensor):
            value = budget.to(device=device, dtype=torch.float32)
            if value.ndim == 0:
                value = value.expand(batch_size)
        else:
            value = torch.full((batch_size,), float(budget), dtype=torch.float32, device=device)
        if value.shape != (batch_size,):
            raise ValueError("output_budget must be a scalar or one value per batch row")
        return value

    def _copy_inputs(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {key: batch[key] for key in COPY_INPUT_KEYS if key in batch}

    def encode_source(self, batch: dict[str, Any]) -> Any:
        self.seam.eval()
        batch = trim_source_batch(batch)
        input_ids = batch["input_ids"]
        batch_size = int(input_ids.shape[0])
        budget = self._output_budget(batch, self.seam, batch_size, input_ids.device)
        copy_inputs = self._copy_inputs(batch)
        prompt_ids = batch["decoder_prompt_ids"].to(device=input_ids.device, dtype=torch.long)
        prompt_mask = batch["decoder_prompt_mask"].to(device=input_ids.device, dtype=torch.bool)
        return self.seam.encode_source(
            input_ids,
            batch["attention_mask"],
            batch["source_content_mask"],
            prompt_ids,
            prompt_mask,
            budget,
            **copy_inputs,
        )

    def forward(
        self,
        batch: dict[str, Any],
        *,
        temperature: float,
        ce: bool = False,
        return_entropy: bool = False,
        bridge: Any | None = None,
        compute_entropy_with_grad: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if bridge is None:
            batch = trim_source_batch(batch)
        input_ids = batch["input_ids"]
        batch_size = int(input_ids.shape[0])
        budget = self._output_budget(batch, self.seam, batch_size, input_ids.device)
        copy_inputs = self._copy_inputs(batch)

        if ce:
            if return_entropy:
                raise ValueError("Entropy is only available when scoring sampled actions")
            labels = batch.get("labels")
            if labels is None:
                raise ValueError("CE scoring requires collator labels")
            decoder_input_ids = batch.get("gold_decoder_input_ids", batch.get("decoder_input_ids"))
            decoder_attention_mask = batch.get("gold_decoder_attention_mask", batch.get("decoder_attention_mask"))
            if decoder_input_ids is None or decoder_input_ids.shape != labels.shape:
                raise ValueError("CE requires gold decoder inputs aligned with labels")
            if (
                decoder_attention_mask is not None
                and decoder_attention_mask.ndim == 2
                and decoder_attention_mask.shape[1] > 1
            ):
                active_gold = decoder_attention_mask.bool().any(dim=0).nonzero(as_tuple=True)[0]
                gold_width = int(active_gold[-1].item()) + 1 if active_gold.numel() else 1
                if gold_width < decoder_input_ids.shape[1]:
                    decoder_input_ids = decoder_input_ids[:, :gold_width]
                    decoder_attention_mask = decoder_attention_mask[:, :gold_width]
                    labels = labels[:, :gold_width]
            if bridge is None:
                output = self.seam(
                    input_ids,
                    batch["attention_mask"],
                    batch["source_content_mask"],
                    batch["decoder_prompt_ids"],
                    batch["decoder_prompt_mask"],
                    decoder_input_ids,
                    decoder_attention_mask,
                    labels,
                    output_budget=budget,
                    return_logits=False,
                    **copy_inputs,
                )
                if output.loss_ce is None:
                    raise RuntimeError("AFMR did not return a supervised CE loss")
                return output.loss_ce
            else:
                _, _, loss_ce = self.seam.decoder(
                    decoder_input_ids,
                    bridge.memory,
                    bridge.memory_mask,
                    bridge.source_bias,
                    decoder_attention_mask,
                    labels=labels,
                    return_logits=False,
                    value_memory=bridge.value_memory,
                    copy_state=bridge.copy_state,
                )
                if loss_ce is None:
                    raise RuntimeError("AFMR decoder did not return a supervised CE loss")
                return loss_ce

        if not math.isfinite(float(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        action_ids = batch.get("action_ids")
        action_mask = batch.get("action_mask")
        decoder_input_ids = batch.get("decoder_input_ids")
        decoder_attention_mask = batch.get("decoder_attention_mask")
        if action_ids is None or action_mask is None:
            raise ValueError("Action scoring requires action_ids and action_mask")
        if decoder_input_ids is None or decoder_attention_mask is None:
            raise ValueError("Action scoring requires full prompt-plus-action decoder inputs and mask")
        if action_ids.ndim != 2 or action_mask.shape != action_ids.shape:
            raise ValueError("action_ids and action_mask must have matching [batch, action_tokens] shapes")
        action_ids = action_ids.to(device=input_ids.device, dtype=torch.long)
        action_mask = action_mask.to(device=input_ids.device, dtype=torch.bool)
        decoder_input_ids = decoder_input_ids.to(device=input_ids.device, dtype=torch.long)
        decoder_attention_mask = decoder_attention_mask.to(device=input_ids.device, dtype=torch.bool)
        prompt_ids = batch["decoder_prompt_ids"].to(device=input_ids.device, dtype=torch.long)
        prompt_mask = batch["decoder_prompt_mask"].to(device=input_ids.device, dtype=torch.bool)
        if prompt_mask.ndim != 2 or prompt_mask.shape[0] != batch_size:
            raise ValueError("decoder_prompt_mask must be [batch, prompt_tokens]")
        prompt_width = int(prompt_ids.shape[1])
        if prompt_width == 0 or not bool(prompt_mask.all()):
            raise ValueError("Action scoring requires fixed-width, non-empty, unpadded prompts")
        action_width = int(action_ids.shape[1])
        if action_ids.shape[0] != batch_size or action_width <= 0:
            raise ValueError("action_ids must contain at least one action token for every batch row")
        expected_width = prompt_width + action_width
        if decoder_input_ids.shape != (batch_size, expected_width):
            raise ValueError("decoder_input_ids must contain exactly the prompt followed by action_ids")
        if decoder_attention_mask.shape != decoder_input_ids.shape:
            raise ValueError("decoder_attention_mask must match decoder_input_ids")
        if not torch.equal(decoder_input_ids[:, :prompt_width], prompt_ids):
            raise ValueError("decoder_input_ids prompt prefix must match decoder_prompt_ids")
        if not torch.equal(decoder_input_ids[:, prompt_width:], action_ids):
            raise ValueError("decoder_input_ids action suffix must match action_ids")
        if not torch.equal(decoder_attention_mask[:, prompt_width:], action_mask):
            raise ValueError("decoder attention mask after the prompt must equal action_mask")

        active_action_cols = action_mask.any(dim=0).nonzero(as_tuple=True)[0]
        active_action_width = int(active_action_cols[-1].item()) + 1 if active_action_cols.numel() else 1
        if active_action_width < action_width:
            action_ids = action_ids[:, :active_action_width]
            action_mask = action_mask[:, :active_action_width]
            decoder_input_ids = decoder_input_ids[:, : prompt_width + active_action_width]
            decoder_attention_mask = decoder_attention_mask[:, : prompt_width + active_action_width]

        # Generation runs in eval mode. Keep dropout disabled while retaining
        # autograd so policy scores match the rollout distribution.
        self.seam.eval()
        decoder = self.seam.decoder
        validate_context(decoder, prompt_width, budget)
        decoder.clear_cross_cache()
        if bridge is None:
            bridge = self.seam.encode_source(
                input_ids,
                batch["attention_mask"],
                batch["source_content_mask"],
                prompt_ids,
                prompt_mask,
                budget,
                **copy_inputs,
            )

        position_ids = decoder_attention_mask.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(~decoder_attention_mask, 0)

        def run_backbone(
            ids: torch.Tensor,
            token_mask: torch.Tensor,
            positions: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            source_bias: torch.Tensor,
            value_memory: torch.Tensor | None,
        ) -> torch.Tensor:
            output = decoder.backbone(
                input_ids=ids,
                attention_mask=token_mask,
                past_key_values=None,
                use_cache=False,
                position_ids=positions[:, -ids.shape[1] :],
                return_dict=True,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,
                encoder_attention_bias=source_bias,
                encoder_value_states=value_memory,
            )
            return output.last_hidden_state

        backbone_args = (
            decoder_input_ids,
            decoder_attention_mask,
            position_ids,
            bridge.memory,
            bridge.memory_mask,
            bridge.source_bias,
            bridge.value_memory,
        )
        if torch.is_grad_enabled():
            hidden = checkpoint(run_backbone, *backbone_args, use_reentrant=False)
        else:
            hidden = run_backbone(*backbone_args)

        first_action_position = prompt_width - 1
        chunks = []
        entropy_chunks = []
        for start in range(0, active_action_width, self.chunk_size):
            stop = min(active_action_width, start + self.chunk_size)
            hidden_start = first_action_position + start
            hidden_stop = first_action_position + stop
            hidden_chunk = hidden[:, hidden_start:hidden_stop]
            action_chunk = action_ids[:, start:stop]
            mask_chunk = action_mask[:, start:stop]

            def score_chunk(
                states: torch.Tensor, actions: torch.Tensor, active: torch.Tensor
            ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                logits = decoder.lm_head(states)
                if decoder.grounded_copy is not None:
                    if bridge.copy_state is None:
                        raise ValueError("Grounded-copy scoring requires the encoded copy state")
                    logits = decoder.grounded_copy.mix_logits(states, logits, bridge.copy_state)
                scaled = logits.float() / float(temperature)
                log_probs = F.log_softmax(scaled, dim=-1)
                selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
                selected = torch.where(active, selected, torch.zeros_like(selected))
                if not return_entropy:
                    return selected
                if compute_entropy_with_grad:
                    probabilities = log_probs.exp()
                    safe_log_probs = log_probs.masked_fill(~torch.isfinite(log_probs), 0.0)
                    entropy = -(probabilities * safe_log_probs).sum(dim=-1)
                    entropy = torch.where(active, entropy, torch.zeros_like(entropy))
                else:
                    with torch.no_grad():
                        distribution_log_probs = log_probs.detach()
                        probabilities = distribution_log_probs.exp()
                        safe_log_probs = distribution_log_probs.masked_fill(
                            ~torch.isfinite(distribution_log_probs), 0.0
                        )
                        entropy = -(probabilities * safe_log_probs).sum(dim=-1)
                        entropy = torch.where(active, entropy, torch.zeros_like(entropy))
                return selected, entropy

            if torch.is_grad_enabled():
                scored = checkpoint(score_chunk, hidden_chunk, action_chunk, mask_chunk, use_reentrant=False)
            else:
                scored = score_chunk(hidden_chunk, action_chunk, mask_chunk)
            if return_entropy:
                selected_log_probs, entropy = scored
            else:
                selected_log_probs = scored
            if bool(mask_chunk.any()) and not bool(torch.isfinite(selected_log_probs.detach()[mask_chunk]).all()):
                raise FloatingPointError("Non-finite policy action log-probability")
            chunks.append(selected_log_probs)
            if return_entropy:
                entropy_chunks.append(entropy if compute_entropy_with_grad else entropy.detach())

        log_probs = torch.cat(chunks, dim=1)
        if active_action_width < action_width:
            log_probs = F.pad(log_probs, (0, action_width - active_action_width), value=0.0)
        if return_entropy:
            entropies = torch.cat(entropy_chunks, dim=1)
            if active_action_width < action_width:
                entropies = F.pad(entropies, (0, action_width - active_action_width), value=0.0)
            return log_probs, entropies
        return log_probs
