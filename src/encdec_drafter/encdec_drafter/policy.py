from __future__ import annotations

from typing import List, Optional, Sequence

import torch


class GreedyPolicy:
    def __init__(
        self,
        vocab_size: int,
        eos_token_ids: Sequence[int],
        min_new_tokens: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        pad_token_id: int = 0,
    ):
        self.vocab_size = int(vocab_size)
        self.eos_token_ids = tuple(int(token_id) for token_id in eos_token_ids)
        self.min_new_tokens = int(min_new_tokens)
        self.repetition_penalty = float(repetition_penalty)
        self.no_repeat_ngram_size = int(no_repeat_ngram_size)
        self.pad_token_id = int(pad_token_id)
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.min_new_tokens < 0 or self.no_repeat_ngram_size < 0:
            raise ValueError("min_new_tokens and no_repeat_ngram_size must be nonnegative")

    def process_block(
        self,
        logits: torch.Tensor,
        committed_ids: Sequence[int],
        proposed_ids: Optional[torch.Tensor],
        generated_count: int,
    ) -> torch.Tensor:
        if logits.ndim != 2:
            raise ValueError("Expected target logits with shape [block_rows, vocabulary]")
        row_count, vocab_size = logits.shape
        if vocab_size != self.vocab_size:
            raise ValueError(f"Target vocabulary changed: policy={self.vocab_size}, logits={vocab_size}")
        if proposed_ids is None:
            proposed_ids = torch.empty((0,), dtype=torch.long, device=logits.device)
        else:
            proposed_ids = proposed_ids.to(device=logits.device, dtype=torch.long).reshape(-1)
        if row_count != proposed_ids.numel() + 1:
            raise ValueError("A block with K proposals must provide K+1 target-logit rows")

        prefix, prefix_mask, prefix_lengths = self._build_prefixes(
            committed_ids,
            proposed_ids,
            row_count,
            logits.device,
        )
        scores = logits.clone()

        if self.repetition_penalty != 1.0:
            token_counts = torch.zeros((row_count, vocab_size), dtype=torch.int32, device=logits.device)
            token_indices = prefix.clamp(min=0, max=vocab_size - 1)
            token_counts.scatter_add_(1, token_indices, prefix_mask.to(torch.int32))
            present = token_counts.bool()
            adjusted = torch.where(scores < 0, scores * self.repetition_penalty, scores / self.repetition_penalty)
            scores = torch.where(present, adjusted, scores)

        if self.no_repeat_ngram_size > 0:
            banned = self._ngram_banned_tokens(prefix, prefix_lengths, self.no_repeat_ngram_size, vocab_size)
            scores.masked_fill_(banned, float("-inf"))

        if self.eos_token_ids:
            for row in range(row_count):
                if generated_count + row < self.min_new_tokens:
                    scores[row, list(self.eos_token_ids)] = float("-inf")
        return scores

    def _build_prefixes(
        self,
        committed_ids: Sequence[int],
        proposed_ids: torch.Tensor,
        row_count: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.as_tensor(list(committed_ids), dtype=torch.long, device=device)
        base_length = int(base.numel())
        proposal_count = int(proposed_ids.numel())
        prefix = torch.full(
            (row_count, base_length + proposal_count),
            self.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        mask = torch.zeros_like(prefix, dtype=torch.bool)
        if base_length:
            prefix[:, :base_length] = base.unsqueeze(0)
            mask[:, :base_length] = True
        if proposal_count:
            rows = torch.arange(row_count, device=device).unsqueeze(1)
            proposal_positions = torch.arange(proposal_count, device=device).unsqueeze(0)
            proposal_mask = proposal_positions < rows
            repeated = proposed_ids.unsqueeze(0).expand(row_count, -1)
            prefix[:, base_length:] = torch.where(proposal_mask, repeated, self.pad_token_id)
            mask[:, base_length:] = proposal_mask
        lengths = base_length + torch.arange(row_count, dtype=torch.long, device=device)
        return prefix, mask, lengths

    @staticmethod
    def _ngram_banned_tokens(
        prefix: torch.Tensor,
        prefix_lengths: torch.Tensor,
        ngram_size: int,
        vocab_size: int,
    ) -> torch.Tensor:
        row_count, max_length = prefix.shape
        banned_counts = torch.zeros((row_count, vocab_size), dtype=torch.int32, device=prefix.device)
        if ngram_size == 1:
            token_indices = prefix.clamp(min=0, max=vocab_size - 1)
            valid = torch.arange(max_length, device=prefix.device).unsqueeze(0) < prefix_lengths.unsqueeze(1)
            banned_counts.scatter_add_(1, token_indices, valid.to(torch.int32))
            return banned_counts.bool()

        context_width = ngram_size - 1
        if max_length < ngram_size:
            return banned_counts.bool()
        suffix_positions = (
            prefix_lengths.unsqueeze(1) - context_width + torch.arange(context_width, device=prefix.device).unsqueeze(0)
        )
        suffix_positions = suffix_positions.clamp(min=0, max=max_length - 1)
        suffix = prefix.gather(1, suffix_positions)
        window_count = max_length - ngram_size + 1
        windows = prefix.unfold(dimension=1, size=ngram_size, step=1)
        matches = (windows[:, :, :context_width] == suffix.unsqueeze(1)).all(dim=-1)
        starts = torch.arange(window_count, device=prefix.device).unsqueeze(0)
        complete = starts + ngram_size <= prefix_lengths.unsqueeze(1)
        valid = matches & complete
        next_ids = windows[:, :, -1].clamp(min=0, max=vocab_size - 1)
        banned_counts.scatter_add_(1, next_ids, valid.to(torch.int32))
        return banned_counts.bool()

    def greedy_ids(
        self,
        logits: torch.Tensor,
        committed_ids: Sequence[int],
        proposed_ids: Optional[torch.Tensor],
        generated_count: int,
    ) -> torch.Tensor:
        return self.process_block(logits, committed_ids, proposed_ids, generated_count).argmax(dim=-1)
