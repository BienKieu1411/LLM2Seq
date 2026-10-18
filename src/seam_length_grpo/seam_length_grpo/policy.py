"""Ragged decoder prompts with exact unpadded native likelihoods."""

from __future__ import annotations

import math

import torch
from seam_grpo.context import validate_context
from seam_grpo.policy import Policy
from seam_grpo.rollout import select_bridge_state, select_rows


def validate_budgets(seam, collator, token_cap):
    validate_context(seam.decoder, collator.length["decoder_prompt_reserve"], token_cap)
    config = getattr(seam.encoder.model, "config", None)
    config = getattr(config, "text_config", None) or config
    value = getattr(config, "max_position_embeddings", None)
    try:
        limit = float(value)
    except (TypeError, ValueError, OverflowError):
        return
    if math.isfinite(limit) and limit > 0 and collator.encoder_limit > limit:
        raise ValueError(
            f"Encoder body + prompt reserve ({collator.encoder_limit}) exceeds encoder positional capacity {int(limit)}"
        )


def unpad_rows(batch, rows):
    result = select_rows(batch, rows)
    mask = result["decoder_prompt_mask"][0].bool()
    if not torch.equal(result["decoder_prompt_mask"].bool(), mask.expand(len(rows), -1)):
        raise ValueError("A native microbatch must have the same decoder prompt width")
    positions = mask.nonzero(as_tuple=True)[0]
    if not positions.numel() or not bool(mask[positions[0] :].all()):
        raise ValueError("Decoder prompts must be nonempty and left padded")
    offset = int(positions[0])
    for key in (
        "decoder_prompt_ids",
        "decoder_prompt_mask",
        "decoder_input_ids",
        "decoder_attention_mask",
        "labels",
        "gold_decoder_input_ids",
        "gold_decoder_attention_mask",
    ):
        if key in result:
            result[key] = result[key][:, offset:]
    return result


def unpad_row(batch, row):
    return unpad_rows(batch, [row])


def prompt_groups(batch, maximum):
    """Contiguous equal-width rows preserve the sampler's candidate order."""
    widths = batch["decoder_prompt_mask"].sum(-1).tolist()
    start = 0
    while start < len(widths):
        end = start + 1
        while end < min(len(widths), start + maximum) and widths[end] == widths[start]:
            end += 1
        yield list(range(start, end))
        start = end


class LengthPolicy(Policy):
    def forward(
        self,
        batch,
        *,
        temperature,
        ce=False,
        return_entropy=False,
        bridge=None,
        compute_entropy_with_grad=False,
    ):
        scores, entropies, losses, counts = [], [], [], []
        for indices in prompt_groups(batch, batch["input_ids"].shape[0]):
            row = unpad_rows(batch, indices)
            row_bridge = (
                select_bridge_state(bridge, torch.tensor(indices, device=bridge.memory.device))
                if bridge is not None
                else None
            )
            output = super().forward(
                row,
                temperature=temperature,
                ce=ce,
                return_entropy=return_entropy,
                bridge=row_bridge,
                compute_entropy_with_grad=compute_entropy_with_grad,
            )
            if ce:
                count = row["labels"][:, 1:].ne(-100).sum()
                losses.append(output * count)
                counts.append(count)
            elif return_entropy:
                score, entropy = output
                scores.append(score)
                entropies.append(entropy)
            else:
                scores.append(output)
        if ce:
            return torch.stack(losses).sum() / torch.stack(counts).sum().clamp_min(1)
        return (torch.cat(scores), torch.cat(entropies)) if return_entropy else torch.cat(scores)
