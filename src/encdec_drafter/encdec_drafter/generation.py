from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from .cache import CacheTransaction
from .drafter import ParallelBlockDrafter
from .memory import PreparedDraftSource
from .policy import GreedyPolicy
from .state import DecodeState, SourceState


@dataclass
class GenerationResult:
    decoder_ids: List[int]
    stats: Dict[str, Any]


class PhaseTimers:
    def __init__(self, device: torch.device):
        self.device = device
        self.events: Dict[str, List[Tuple[Any, Any]]] = {}
        self.cpu_times: Dict[str, float] = {}

    def start(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def stop(self, name: str, started) -> None:
        if self.device.type == "cuda":
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self.events.setdefault(name, []).append((started, ended))
        else:
            self.cpu_times[name] = self.cpu_times.get(name, 0.0) + time.perf_counter() - started

    def resolve(self) -> Dict[str, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            result = dict(self.cpu_times)
            for name, event_pairs in self.events.items():
                result[name] = sum(start.elapsed_time(end) for start, end in event_pairs) / 1000.0
            return result
        return dict(self.cpu_times)


class GreedySpeculativeGenerator:
    def __init__(self, adapter, config: Dict[str, Any], drafter: Optional[ParallelBlockDrafter] = None):
        self.adapter = adapter
        self.config = config
        self.drafter = drafter
        generation = config["generation"]
        self.max_new_tokens = int(generation["max_new_tokens"])
        self.block_size = int(config["speculation"].get("block_size", 0)) if drafter is not None else 0
        if drafter is not None and not 0 <= self.block_size <= drafter.max_block_size:
            raise ValueError("speculation.block_size must be in [0, max_block_size]")
        self.policy = GreedyPolicy(
            vocab_size=adapter.vocab_size,
            eos_token_ids=adapter.eos_token_ids,
            min_new_tokens=int(generation.get("min_new_tokens", 0)),
            repetition_penalty=float(generation.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(generation.get("no_repeat_ngram_size", 0)),
            pad_token_id=int(adapter.pad_token_id),
        )

    @torch.inference_mode()
    def generate_source(self, source: SourceState, source_id: str = "") -> GenerationResult:
        started_at = time.perf_counter()
        device = self.adapter.device
        target_cache = self.adapter.new_cache()
        feature_width = int(self.adapter.model.config.decoder.hidden_size)
        feature_capacity = max(1, self.max_new_tokens + 1)
        feature_buffer = torch.empty(
            (1, feature_capacity, feature_width),
            dtype=next(self.adapter.model.parameters()).dtype,
            device=device,
        )
        state = DecodeState(
            source=source,
            decoder_ids=[self.adapter.start_token_id],
            target_cache=target_cache,
            target_cache_length=0,
            feature_buffer=feature_buffer,
            stats={"source_id": source_id or source.source_id},
        )
        timers = PhaseTimers(device)
        stats: Dict[str, Any] = {
            "cycles": 0,
            "target_forwards": 0,
            "proposed_tokens": 0,
            "accepted_tokens": 0,
            "source_tokens": source.valid_length,
            "draft_source_tokens": None,
            "generated_tokens": 0,
        }
        prepared_source: Optional[PreparedDraftSource] = None
        if self.drafter is not None:
            memory_cfg = self.config["source_memory"]
            prepared_source = self.drafter.prepare_source(
                source.hidden_states,
                source.attention_mask,
                source.fingerprint,
                mode=str(memory_cfg.get("mode", "full")),
                budget=int(memory_cfg.get("budget", 256)),
            )
            source.draft_memory = prepared_source
            stats["draft_source_tokens"] = prepared_source.retained_tokens

        try:
            while state.generated_count < self.max_new_tokens:
                state.assert_nonterminal_invariant()
                remaining = self.max_new_tokens - state.generated_count
                block_size = 0
                if self.drafter is not None and state.feature_length > 0:
                    block_size = min(self.block_size, remaining)
                if block_size == 0:
                    self._target_only_step(state, timers, stats)
                    if state.finished:
                        break
                    continue
                if prepared_source is None:
                    raise RuntimeError("Draft source was not prepared")
                self._speculative_step(state, prepared_source, block_size, timers, stats)
                if state.finished:
                    break
            state.finished = True
            stats.update(timers.resolve())
            stats["generated_tokens"] = state.generated_count
            stats["elapsed_seconds"] = time.perf_counter() - started_at
            return GenerationResult(decoder_ids=list(state.decoder_ids), stats=stats)
        finally:
            source.draft_memory = None

    def _target_only_step(self, state: DecodeState, timers: PhaseTimers, stats: Dict[str, Any]) -> None:
        cache = state.target_cache
        transaction = cache.begin()
        timer = timers.start()
        try:
            logits, hidden = self.adapter.forward_decoder(state.source, [state.pending_token], cache)
        except BaseException as exc:
            cache.abort_on_error(transaction, exc)
        timers.stop("target_decode", timer)
        processed = self.policy.process_block(
            logits[0],
            state.decoder_ids,
            None,
            state.generated_count,
        )
        token = int(processed[0].argmax().item())
        transaction.commit_to(cache.length)
        state.target_cache_length = cache.length
        state.append_processed_features(hidden)
        state.decoder_ids.append(token)
        state.generated_count += 1
        stats["target_forwards"] += 1
        stats["cycles"] += 1
        if token in self.adapter.eos_token_ids or state.generated_count >= self.max_new_tokens:
            state.finished = True
        else:
            state.assert_nonterminal_invariant()

    def _speculative_step(
        self,
        state: DecodeState,
        prepared_source: PreparedDraftSource,
        block_size: int,
        timers: PhaseTimers,
        stats: Dict[str, Any],
    ) -> None:
        if self.drafter is None:
            raise RuntimeError("Speculative step requested without a drafter")
        accepted_length = state.feature_length
        history_start = max(0, accepted_length - self.drafter.shape.target_feature_window)
        history = state.feature_buffer[:, history_start:accepted_length]

        timer = timers.start()
        draft_logits = self.drafter(
            pending_ids=torch.tensor([state.pending_token], dtype=torch.long, device=self.adapter.device),
            accepted_features=history,
            source=prepared_source,
            input_embedding=self.adapter.input_embedding,
            output_head=self.adapter.output_head,
            block_size=block_size,
        )
        proposals = draft_logits[0].argmax(dim=-1)
        timers.stop("draft", timer)

        pending = torch.tensor([state.pending_token], dtype=torch.long, device=self.adapter.device)
        verifier_input = torch.cat([pending, proposals], dim=0)
        cache = state.target_cache
        transaction: CacheTransaction = cache.begin()
        starting_cache_length = transaction.start_length
        timer = timers.start()
        try:
            logits, hidden = self.adapter.forward_decoder(state.source, verifier_input, cache)
        except BaseException as exc:
            cache.abort_on_error(transaction, exc)
        timers.stop("target_verify", timer)
        processed_logits = self.policy.process_block(
            logits[0],
            state.decoder_ids,
            proposals,
            state.generated_count,
        )
        expected = processed_logits.argmax(dim=-1)
        matches = expected[:-1] == proposals
        compact_summary = torch.cat([matches.to(torch.long), expected.to(torch.long)]).cpu().tolist()
        match_values = [bool(value) for value in compact_summary[:block_size]]
        expected_values = [int(value) for value in compact_summary[block_size:]]

        accepted = 0
        emitted: List[int] = []
        mismatch = False
        stopped_on_eos = False
        for index in range(block_size):
            if match_values[index]:
                accepted += 1
                token = expected_values[index]
                emitted.append(token)
                if token in self.adapter.eos_token_ids:
                    stopped_on_eos = True
                    break
            else:
                emitted.append(expected_values[index])
                mismatch = True
                break

        if mismatch:
            keep_length = starting_cache_length + accepted + 1
            transaction.commit_to(keep_length)
            state.target_cache_length = cache.length
            state.append_processed_features(hidden[:, : accepted + 1])
            state.decoder_ids.extend(emitted)
            state.generated_count += len(emitted)
            state.finished = emitted[-1] in self.adapter.eos_token_ids or state.generated_count >= self.max_new_tokens
        elif stopped_on_eos:
            transaction.commit_to(starting_cache_length + accepted)
            state.target_cache_length = cache.length
            state.append_processed_features(hidden[:, :accepted])
            state.decoder_ids.extend(emitted)
            state.generated_count += len(emitted)
            state.finished = True
        else:
            all_proposals_fit = state.generated_count + block_size >= self.max_new_tokens
            if all_proposals_fit:
                transaction.commit_to(starting_cache_length + block_size)
                state.append_processed_features(hidden[:, :block_size])
                state.decoder_ids.extend(emitted)
                state.generated_count += len(emitted)
                state.target_cache_length = cache.length
                state.finished = True
            else:
                bonus = expected_values[block_size]
                transaction.commit_to(cache.length)
                state.append_processed_features(hidden)
                state.decoder_ids.extend(emitted)
                state.decoder_ids.append(bonus)
                state.generated_count += len(emitted) + 1
                state.target_cache_length = cache.length
                state.finished = bonus in self.adapter.eos_token_ids or state.generated_count >= self.max_new_tokens

        stats["target_forwards"] += 1
        stats["cycles"] += 1
        stats["proposed_tokens"] += block_size
        stats["accepted_tokens"] += accepted
        if not state.finished:
            state.assert_nonterminal_invariant()


def generate_native(adapter, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> GenerationResult:
    started = time.perf_counter()
    generated = adapter.native_generate(input_ids, attention_mask)
    if adapter.device.type == "cuda":
        torch.cuda.synchronize(adapter.device)
    return GenerationResult(
        decoder_ids=[int(value) for value in generated.detach().cpu().tolist()],
        stats={
            "elapsed_seconds": time.perf_counter() - started,
            "generated_tokens": max(0, int(generated.shape[0]) - 1),
        },
    )
