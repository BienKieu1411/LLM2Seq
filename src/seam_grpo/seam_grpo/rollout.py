"""Native, full-support sampling for grouped SEAM rollouts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from eviseq_afmr.data.copy_alignment import COPY_INPUT_KEYS
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .context import validate_context


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    while isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module
    return model


def _batch_size(batch: Mapping[str, Any]) -> int:
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim < 1:
        raise ValueError("batch.input_ids must be a batched tensor")
    return int(input_ids.shape[0])


def select_rows(batch: Mapping[str, Any], indices: Sequence[int] | torch.Tensor) -> dict[str, Any]:
    """Select batch rows from tensor and per-example metadata fields."""

    size = _batch_size(batch)
    if isinstance(indices, torch.Tensor):
        rows = [int(value) for value in indices.detach().cpu().reshape(-1).tolist()]
    else:
        rows = [int(value) for value in indices]
    if any(value < 0 or value >= size for value in rows):
        raise IndexError("row index is outside the batch")
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == size:
            index = torch.tensor(rows, dtype=torch.long, device=value.device)
            result[key] = value.index_select(0, index)
        elif isinstance(value, list) and len(value) == size:
            result[key] = [value[index] for index in rows]
        elif isinstance(value, tuple) and len(value) == size:
            result[key] = tuple(value[index] for index in rows)
        else:
            result[key] = value
    return result


def repeat_batch(batch: Mapping[str, Any], group_size: int) -> dict[str, Any]:
    """Repeat each source row contiguously ``group_size`` times."""

    if group_size <= 0:
        raise ValueError("group_size must be positive")
    indices = torch.arange(_batch_size(batch), dtype=torch.long).repeat_interleave(int(group_size))
    return select_rows(batch, indices)


def rollout(
    seam: torch.nn.Module,
    batch: Mapping[str, Any],
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    """Sample one completion per row and return its exact temperature-only log-probs.

    The distribution has full vocabulary support: no top-k/top-p truncation,
    repetition penalty, or forced EOS. EOS is a valid action; subsequent pads
    have ``action_mask=False``. If no EOS is sampled, the sequence ends at the
    token budget without inserting an EOS.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    size = _batch_size(batch)
    prompt_ids = batch.get("decoder_prompt_ids")
    prompt_mask = batch.get("decoder_prompt_mask")
    if not isinstance(prompt_ids, torch.Tensor) or not isinstance(prompt_mask, torch.Tensor):
        raise ValueError("rollout requires decoder_prompt_ids and decoder_prompt_mask tensors")
    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape or prompt_ids.shape[0] != size:
        raise ValueError("decoder prompts and masks must have matching [batch, prompt_tokens] shapes")
    prompt_mask = prompt_mask.bool()
    if prompt_ids.shape[1] == 0 or not bool(prompt_mask.all()):
        raise ValueError("rollout requires fixed-width, non-empty, unpadded prompts")

    model = _unwrap(seam)
    seam.eval()
    device = prompt_ids.device
    batch_size = size
    decoder = model.decoder
    validate_context(decoder, int(prompt_ids.shape[1]), max_new_tokens)
    output_budget = torch.full((batch_size,), float(max_new_tokens), dtype=torch.float32, device=device)
    copy_inputs = {key: batch[key] for key in COPY_INPUT_KEYS if key in batch}
    eos_value = getattr(tokenizer, "eos_token_id", None)
    eos_token_id = None if eos_value is None else int(eos_value)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)

    action_steps: list[torch.Tensor] = []
    mask_steps: list[torch.Tensor] = []
    logp_steps: list[torch.Tensor] = []
    # Explicitly disable an enclosing inference_mode so returned tokens are
    # ordinary tensors that a later gradient-enabled embedding lookup can use.
    try:
        decoder.clear_cross_cache()
        with torch.inference_mode(False), torch.no_grad():
            bridge = model.encode_source(
                batch["input_ids"],
                batch["attention_mask"],
                batch["source_content_mask"],
                prompt_ids,
                prompt_mask,
                output_budget,
                **copy_inputs,
            )
            token_ids = prompt_ids
            decode_mask = prompt_mask
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            past = None
            decoder.prepare_cross_cache(
                bridge.memory,
                **({"value_memory": bridge.value_memory} if bridge.value_memory is not None else {}),
            )
            for _ in range(int(max_new_tokens)):
                current = token_ids if past is None else token_ids[:, -1:]
                logits, past, _ = decoder(
                    current,
                    bridge.memory,
                    bridge.memory_mask,
                    bridge.source_bias,
                    decode_mask,
                    past_key_values=past,
                    use_cache=True,
                    value_memory=bridge.value_memory,
                    copy_state=bridge.copy_state,
                )
                scores = logits[:, -1].float() / float(temperature)
                active = ~finished
                active_rows = active.nonzero(as_tuple=True)[0]
                if active_rows.numel() == 0:
                    break
                active_scores = scores.index_select(0, active_rows)
                if not bool(torch.isfinite(active_scores).all()):
                    raise FloatingPointError("Non-finite logits encountered during rollout")
                active_log_probs = F.log_softmax(active_scores, dim=-1)
                if not bool(torch.isfinite(active_log_probs).all()):
                    raise FloatingPointError("Non-finite sampling distribution encountered during rollout")
                sampled = torch.multinomial(active_log_probs.exp(), 1, generator=generator).squeeze(-1)
                step_ids = torch.full((batch_size,), pad_token_id, dtype=prompt_ids.dtype, device=device)
                step_ids[active_rows] = sampled
                step_log_probs = torch.zeros(batch_size, dtype=torch.float32, device=device)
                step_log_probs[active_rows] = active_log_probs.gather(-1, sampled[:, None]).squeeze(-1)
                action_steps.append(step_ids)
                mask_steps.append(active)
                logp_steps.append(step_log_probs)

                decode_mask = torch.cat((decode_mask, active[:, None]), dim=1)
                token_ids = torch.cat((token_ids, step_ids[:, None]), dim=1)
                if eos_token_id is not None:
                    finished = finished | (active & step_ids.eq(eos_token_id))
                    if bool(finished.all()):
                        break

            action_ids = torch.stack(action_steps, dim=1).detach().clone()
            action_mask = torch.stack(mask_steps, dim=1).detach().clone()
            old_logps = torch.stack(logp_steps, dim=1).detach().clone()
    finally:
        decoder.clear_cross_cache()

    rows = [action_ids[row][action_mask[row]].tolist() for row in range(batch_size)]
    texts = list(tokenizer.batch_decode(rows, skip_special_tokens=True))
    return {
        "action_ids": action_ids,
        "action_mask": action_mask,
        "old_logps": old_logps,
        "texts": texts,
    }
