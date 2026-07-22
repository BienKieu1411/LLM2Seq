"""Standalone autoregressive generation for GenBridge models."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F


def _blocked_ngram_tokens(tokens: Sequence[int], ngram_size: int) -> list[int]:
    if ngram_size <= 0 or len(tokens) < ngram_size:
        return []
    prefix_size = ngram_size - 1
    current_prefix = tuple(tokens[-prefix_size:]) if prefix_size else ()
    blocked = []
    for index in range(len(tokens) - ngram_size + 1):
        ngram = tuple(tokens[index : index + ngram_size])
        if ngram[:-1] == current_prefix:
            blocked.append(ngram[-1])
    return blocked


class OutputOnlyRepetitionPenaltyLogitsProcessor:
    """Apply repetition penalty after, but never inside, a fixed prompt."""

    def __init__(self, penalty: float, prompt_length: int):
        if penalty <= 0.0:
            raise ValueError("repetition penalty must be positive")
        self.penalty = float(penalty)
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        history = input_ids[:, self.prompt_length :]
        if history.numel() == 0 or self.penalty == 1.0:
            return scores
        previous_scores = torch.gather(scores, 1, history)
        adjusted = torch.where(
            previous_scores < 0,
            previous_scores * self.penalty,
            previous_scores / self.penalty,
        )
        return scores.scatter(1, history, adjusted)


class OutputOnlyNoRepeatNGramLogitsProcessor:
    """Block n-grams formed by generated output, excluding source tokens."""

    def __init__(self, ngram_size: int, prompt_length: int):
        if ngram_size <= 0:
            raise ValueError("ngram_size must be positive")
        self.ngram_size = int(ngram_size)
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for batch_index in range(input_ids.shape[0]):
            tokens = input_ids[batch_index, self.prompt_length :].tolist()
            blocked = _blocked_ngram_tokens(tokens, self.ngram_size)
            if blocked:
                scores[batch_index, blocked] = float("-inf")
        return scores


@torch.inference_mode()
def autoregressive_generate(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    unit_ids: Optional[torch.Tensor] = None,
    bridge_output=None,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    decoder_prefix_ids: Optional[Sequence[int]] = None,
    use_cache: bool = True,
    return_diagnostics: bool = False,
) -> Any:
    model.eval()
    batch_size = input_ids.size(0)
    if bridge_output is None:
        bridge_output = model.encode(
            input_ids,
            attention_mask,
            unit_ids=unit_ids,
            return_bridge_output=True,
        )
    memory_kwargs = model.decoder_memory_kwargs(bridge_output)
    if hasattr(model.decoder, "prepare_cross_attention_cache"):
        model.decoder.prepare_cross_attention_cache(
            memory_kwargs["encoder_hidden_states"],
            token_encoder_hidden_states=memory_kwargs.get("token_encoder_hidden_states"),
            plan_encoder_hidden_states=memory_kwargs.get("plan_encoder_hidden_states"),
        )
    if decoder_prefix_ids is None:
        if bos_token_id is None:
            bos_token_id = eos_token_id or 0
        decoder_prefix_ids = [bos_token_id]
    if not decoder_prefix_ids:
        raise ValueError("decoder_prefix_ids cannot be empty")
    decoder_seed = torch.tensor(
        list(decoder_prefix_ids),
        dtype=torch.long,
        device=input_ids.device,
    )
    generated = decoder_seed.unsqueeze(0).expand(batch_size, -1).clone()
    prefix_length = generated.shape[1]
    finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)
    past_key_values = None
    plan_layer_sums: Optional[torch.Tensor] = None
    plan_layer_counts: Optional[torch.Tensor] = None
    plan_step_sums: list[torch.Tensor] = []
    plan_step_counts: list[torch.Tensor] = []

    for step in range(max_new_tokens):
        decoder_input = generated[:, -1:] if use_cache and past_key_values is not None else generated
        decoder_states, next_cache = model.decoder(
            input_ids=decoder_input,
            past_key_values=past_key_values if use_cache else None,
            use_cache=use_cache,
            **memory_kwargs,
        )
        if use_cache:
            past_key_values = next_cache
        logits = model.lm_head(decoder_states[:, -1, :])

        if return_diagnostics and hasattr(model.decoder, "plan_gate_values"):
            gate_values = model.decoder.plan_gate_values()
            if gate_values:
                # At the first decoding step the decoder consumes the entire
                # fixed prefix; only its final query predicts the next output.
                latest = torch.stack([values[:, -1] for values in gate_values], dim=0)
                active = (~finished).float().unsqueeze(0)
                if plan_layer_sums is None:
                    plan_layer_sums = torch.zeros(latest.shape[0], device=latest.device, dtype=torch.float32)
                    plan_layer_counts = torch.zeros_like(plan_layer_sums)
                plan_layer_sums += (latest * active).sum(dim=1)
                plan_layer_counts += active.sum(dim=1).expand_as(plan_layer_counts)
                plan_step_sums.append((latest * active).sum())
                plan_step_counts.append(active.sum() * float(latest.shape[0]))

        # Match Transformers' MinNewTokensLengthLogitsProcessor: EOS remains
        # blocked while the number of already generated tokens is below the
        # requested minimum. The previous step+1 check allowed EOS one token
        # too early.
        if eos_token_id is not None and step < min_new_tokens:
            logits[:, eos_token_id] = float("-inf")
        if repetition_penalty != 1.0:
            for batch_index in range(batch_size):
                # The fixed decoder instruction is conditioning context, not
                # generated summary text.  Penalising its tokens distorts the
                # first prediction and unfairly blocks ordinary task words.
                previous = torch.unique(generated[batch_index, prefix_length:])
                scores = logits[batch_index, previous]
                logits[batch_index, previous] = torch.where(
                    scores < 0,
                    scores * repetition_penalty,
                    scores / repetition_penalty,
                )
        generated_length = generated.size(1) - prefix_length
        if no_repeat_ngram_size > 0 and generated_length >= no_repeat_ngram_size:
            for batch_index in range(batch_size):
                tokens = generated[batch_index, prefix_length:].tolist()
                blocked = _blocked_ngram_tokens(tokens, no_repeat_ngram_size)
                if blocked:
                    logits[batch_index, blocked] = float("-inf")

        if temperature <= 0:
            do_sample = False
        elif temperature != 1.0:
            logits = logits / temperature
        if do_sample and top_k > 0:
            threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        if do_sample and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            remove = remove.scatter(1, sorted_indices, remove)
            logits = logits.masked_fill(remove, float("-inf"))

        if do_sample:
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)
        if pad_token_id is not None:
            next_token = next_token.masked_fill(finished.unsqueeze(1), pad_token_id)
        generated = torch.cat([generated, next_token], dim=1)
        if eos_token_id is not None:
            finished |= next_token.squeeze(1).eq(eos_token_id)
            if finished.all():
                break

    if hasattr(model.decoder, "clear_cross_attention_cache"):
        model.decoder.clear_cross_attention_cache()
    output = generated[:, prefix_length:]
    if not return_diagnostics:
        return output
    diagnostics = {
        "plan_gate_layer_sums": (plan_layer_sums.detach().cpu().tolist() if plan_layer_sums is not None else []),
        "plan_gate_layer_counts": (plan_layer_counts.detach().cpu().tolist() if plan_layer_counts is not None else []),
        "plan_gate_step_sums": [float(value.detach().cpu()) for value in plan_step_sums],
        "plan_gate_step_counts": [float(value.detach().cpu()) for value in plan_step_counts],
    }
    return output, diagnostics
