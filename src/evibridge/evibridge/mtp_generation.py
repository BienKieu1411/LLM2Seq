"""Cache-safe, main-model-verified multi-token decoding for EviBridge."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch


@dataclass
class MTPGenerationOutput:
    generated_ids: torch.Tensor
    metrics: Dict[str, float | bool]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _apply_constraints(
    logits: torch.Tensor,
    prefix_tokens: Sequence[int],
    generated_count: int,
    min_new_tokens: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_token_id: Optional[int],
) -> torch.Tensor:
    constrained = logits.clone()
    if eos_token_id is not None and generated_count + 1 < min_new_tokens:
        constrained[:, eos_token_id] = float("-inf")
    if repetition_penalty != 1.0 and prefix_tokens:
        previous = sorted(set(int(token) for token in prefix_tokens))
        scores = constrained[0, previous]
        constrained[0, previous] = torch.where(
            scores < 0,
            scores * repetition_penalty,
            scores / repetition_penalty,
        )
    if no_repeat_ngram_size > 0 and len(prefix_tokens) >= no_repeat_ngram_size:
        prefix_width = no_repeat_ngram_size - 1
        current = tuple(prefix_tokens[-prefix_width:]) if prefix_width else ()
        blocked = []
        for index in range(len(prefix_tokens) - no_repeat_ngram_size + 1):
            ngram = tuple(prefix_tokens[index : index + no_repeat_ngram_size])
            if ngram[:-1] == current:
                blocked.append(int(ngram[-1]))
        if blocked:
            constrained[0, blocked] = float("-inf")
    return constrained


def _snapshot_recurrent_states(cache: Any) -> List[Dict[int, torch.Tensor]]:
    snapshots: List[Dict[int, torch.Tensor]] = []
    for layer in getattr(cache, "layers", []):
        layer_states: Dict[int, torch.Tensor] = {}
        recurrent = getattr(layer, "recurrent_states", {})
        initialized = getattr(layer, "is_recurrent_states_initialized", {})
        for state_index, tensor in recurrent.items():
            if tensor is not None and bool(initialized.get(state_index, True)):
                layer_states[int(state_index)] = tensor.clone()
        snapshots.append(layer_states)
    return snapshots


def _restore_recurrent_states(cache: Any, snapshots: List[Dict[int, torch.Tensor]]) -> None:
    for layer, layer_states in zip(getattr(cache, "layers", []), snapshots):
        recurrent = getattr(layer, "recurrent_states", {})
        for state_index, snapshot in layer_states.items():
            if recurrent.get(state_index) is None:
                recurrent[state_index] = snapshot.clone()
            else:
                recurrent[state_index].copy_(snapshot)


def _activate_rollback(cache: Any) -> None:
    if not hasattr(cache, "activate_past_recording") or not hasattr(cache, "crop"):
        raise TypeError(
            "Verified MTP requires a Transformers Cache with activate_past_recording() and crop()"
        )
    cache.activate_past_recording()


def _crop_cache(cache: Any, tokens_to_remove: int) -> None:
    if tokens_to_remove < 0:
        raise ValueError("tokens_to_remove must be non-negative")
    cache.crop(-tokens_to_remove if tokens_to_remove else 0)


@torch.inference_mode()
def verified_mtp_generate(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    unit_ids: Optional[torch.Tensor] = None,
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    bos_token_id: Optional[int] = None,
    fallback_probe_steps: int = 8,
    fallback_min_accepted_drafts: float = 1.0,
) -> MTPGenerationOutput:
    """Return output exactly matching greedy main-decoder generation.

    Draft verification mutates the hybrid Qwen3.5 cache.  Full-attention and
    convolution states can be cropped, while recurrent linear-attention states
    cannot; those small fixed-size states are snapshotted and restored before a
    partial candidate is replayed.  This avoids the invalid-cache bug in the old
    LLM2Seq MTP path.
    """

    if input_ids.shape[0] != 1:
        raise ValueError("Verified MTP currently requires batch_size=1 for exact per-example acceptance")
    predictor = model.mtp_predictor
    if predictor is None:
        raise ValueError("Call model.enable_mtp() and load a phase-3 checkpoint first")
    model.eval()
    device = input_ids.device
    if bos_token_id is None:
        bos_token_id = eos_token_id if eos_token_id is not None else 0
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else 0

    _synchronize(device)
    encode_start = time.perf_counter()
    memory, memory_mask = model.encode(
        input_ids,
        attention_mask,
        unit_ids=unit_ids,
        return_attention_mask=True,
    )
    if hasattr(model.decoder, "prepare_cross_attention_cache"):
        model.decoder.prepare_cross_attention_cache(memory)
    _synchronize(device)
    encode_seconds = time.perf_counter() - encode_start

    prefix_tokens = [int(bos_token_id)]
    emitted: List[int] = []
    current_input = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    past_key_values = None
    decoder_calls = 0
    verify_calls = 0
    replay_calls = 0
    drafted_tokens = 0
    accepted_drafts = 0
    cycles = 0
    fallback = False

    _synchronize(device)
    decode_start = time.perf_counter()
    while len(emitted) < max_new_tokens:
        states, past_key_values = model.decoder(
            input_ids=current_input,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        decoder_calls += 1
        logits = _apply_constraints(
            model.lm_head(states[:, -1, :]),
            prefix_tokens,
            len(emitted),
            min_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
            eos_token_id,
        )
        main_token = logits.argmax(dim=-1, keepdim=True)

        # Greedy decoding is already complete when the main model emits EOS;
        # drafting and verifying beyond it only adds latency.
        if eos_token_id is not None and int(main_token.item()) == eos_token_id:
            emitted.append(int(eos_token_id))
            prefix_tokens.append(int(eos_token_id))
            break

        remaining = max_new_tokens - len(emitted)
        if fallback or remaining == 1:
            token = int(main_token.item())
            emitted.append(token)
            prefix_tokens.append(token)
            current_input = main_token
            if eos_token_id is not None and token == eos_token_id:
                break
            continue

        maximum_drafts = min(predictor.num_draft_tokens, remaining - 1)

        def constrain_draft(draft_logits: torch.Tensor, draft_prefix: Sequence[int]) -> torch.Tensor:
            return _apply_constraints(
                draft_logits,
                draft_prefix,
                len(draft_prefix) - 1,
                min_new_tokens,
                repetition_penalty,
                no_repeat_ngram_size,
                eos_token_id,
            )

        drafts = predictor.draft(
            states[:, -1:, :],
            main_token,
            model.decoder.embed_tokens,
            model.lm_head,
            constrain_draft,
            prefix_tokens,
            maximum=maximum_drafts,
        )
        if not drafts:
            token = int(main_token.item())
            emitted.append(token)
            prefix_tokens.append(token)
            current_input = main_token
            if eos_token_id is not None and token == eos_token_id:
                break
            continue

        candidate = torch.cat([main_token, *drafts], dim=1)
        drafted_tokens += len(drafts)
        recurrent_snapshot = _snapshot_recurrent_states(past_key_values)
        _activate_rollback(past_key_values)
        verify_states, verified_cache = model.decoder(
            input_ids=candidate,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        decoder_calls += 1
        verify_calls += 1
        past_key_values = verified_cache
        # One batched vocabulary GEMM is materially faster than K+1 tiny
        # projections on Qwen3.5's large vocabulary.
        verification_logits = model.lm_head(verify_states)

        accepted = 0
        correction = None
        candidate_values = [int(token) for token in candidate[0].tolist()]
        for position in range(candidate.shape[1]):
            verifier_prefix = prefix_tokens + candidate_values[: position + 1]
            verifier_logits = _apply_constraints(
                verification_logits[:, position, :],
                verifier_prefix,
                len(emitted) + position + 1,
                min_new_tokens,
                repetition_penalty,
                no_repeat_ngram_size,
                eos_token_id,
            )
            verifier_token = verifier_logits.argmax(dim=-1, keepdim=True)
            if position == candidate.shape[1] - 1:
                correction = verifier_token
                break
            if int(candidate[0, position + 1]) == int(verifier_token.item()):
                accepted += 1
                continue
            correction = verifier_token
            break
        if correction is None:  # pragma: no cover - defensive
            raise RuntimeError("Verifier failed to produce a correction/bonus token")

        accepted_drafts += accepted
        cycles += 1
        accepted_candidate = candidate[:, : 1 + accepted]
        all_drafts_accepted = accepted == len(drafts)

        if all_drafts_accepted:
            # The verified cache already contains the whole candidate. Compact
            # recorded convolution history; the recurrent state is correct.
            _crop_cache(past_key_values, 0)
        else:
            # Remove every speculative token, restore the pre-verification
            # recurrent states, then commit only the accepted candidate prefix.
            _crop_cache(past_key_values, candidate.shape[1])
            _restore_recurrent_states(past_key_values, recurrent_snapshot)
            _, past_key_values = model.decoder(
                input_ids=accepted_candidate,
                encoder_hidden_states=memory,
                encoder_attention_mask=memory_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            decoder_calls += 1
            replay_calls += 1
            _crop_cache(past_key_values, 0)

        continuation = [int(token) for token in accepted_candidate[0].tolist()]
        continuation.append(int(correction.item()))
        continuation = continuation[:remaining]
        eos_found = False
        if eos_token_id is not None and eos_token_id in continuation:
            continuation = continuation[: continuation.index(eos_token_id) + 1]
            eos_found = True
        emitted.extend(continuation)
        prefix_tokens.extend(continuation)
        current_input = correction

        if eos_found or len(emitted) >= max_new_tokens:
            break
        if (
            not fallback
            and fallback_probe_steps > 0
            and cycles >= fallback_probe_steps
            and accepted_drafts / max(1, cycles) < fallback_min_accepted_drafts
        ):
            fallback = True

    _synchronize(device)
    decode_seconds = time.perf_counter() - decode_start
    if hasattr(model.decoder, "clear_cross_attention_cache"):
        model.decoder.clear_cross_attention_cache()
    generated = torch.tensor([emitted], dtype=torch.long, device=device)
    metrics: Dict[str, float | bool] = {
        "encode_seconds": encode_seconds,
        "decode_seconds": decode_seconds,
        "decoder_calls": float(decoder_calls),
        "verify_calls": float(verify_calls),
        "replay_calls": float(replay_calls),
        "emitted_tokens": float(len(emitted)),
        "drafted_tokens": float(drafted_tokens),
        "accepted_drafts": float(accepted_drafts),
        "tokens_per_decoder_call": len(emitted) / max(1, decoder_calls),
        "draft_acceptance_rate": accepted_drafts / max(1, drafted_tokens),
        "average_accepted_drafts": accepted_drafts / max(1, cycles),
        "fallback_triggered": fallback,
        "verified_with_main": True,
    }
    return MTPGenerationOutput(generated, metrics)
