"""Frozen full-SEAM source-discrimination scoring, never an actor gradient path."""

from __future__ import annotations

import torch
from seam_grpo.actor import find_unique_source_indices
from seam_grpo.data import move, trajectory
from seam_grpo.rollout import select_bridge_state, select_rows


class SourceTeacher:
    def __init__(self, actor, collator, config, *, max_new_tokens, micro_batch_size):
        if actor is None or any(p.requires_grad for p in actor.parameters()):
            raise ValueError("Source teacher must be a frozen full SEAM actor")
        self.actor, self.collator, self.config = actor, collator, config
        self.budget, self.micro_batch_size = max_new_tokens, micro_batch_size
        self.actor.eval()

    @torch.no_grad()
    def __call__(self, true_source, rows, completion_ids):
        if len(rows) != len(completion_ids):
            raise ValueError("Teacher source/action row mismatch")
        device = true_source["input_ids"].device
        width = max(map(len, completion_ids))
        if width < 1 or width > self.budget:
            raise ValueError("Teacher requires sampled actions within the token cap")
        actions = torch.zeros(len(rows), width, device=device, dtype=torch.long)
        mask = torch.zeros_like(actions, dtype=torch.bool)
        for i, tokens in enumerate(completion_ids):
            if not tokens:
                raise ValueError("Teacher action rows cannot be empty")
            actions[i, : len(tokens)] = torch.tensor(tokens, device=device)
            mask[i, : len(tokens)] = True
        negatives = []
        for row, visible in zip(rows, true_source["sources"], strict=True):
            if not row.get("negative_source"):
                raise ValueError("Source reward requires a matched TRAIN negative per example")
            negative = {**row, "id": row["negative_id"], "prompt": row["negative_source"]}
            if self.collator.visible_source(negative["prompt"]) == visible:
                raise ValueError("Negative source equals the actual visible source")
            negatives.append(negative)
        # This rebuilds encoder offsets, memory, bias AND copy state for the negative.
        wrong_source = move(self.collator(negatives), device)
        means = []
        for source in (true_source, wrong_source):
            unique_indices, inverse_indices = find_unique_source_indices(source)
            unique_batch = select_rows(source, unique_indices)
            unique_bridge = self.actor.policy.encode_source(unique_batch)
            values = []
            for start in range(0, len(rows), self.micro_batch_size):
                end = min(len(rows), start + self.micro_batch_size)
                batch = trajectory(
                    select_rows(source, range(start, end)), actions[start:end], mask[start:end], self.budget
                )
                slice_bridge = select_bridge_state(unique_bridge, inverse_indices[start:end])
                logps = self.actor.policy(batch, temperature=1.0, bridge=slice_bridge)
                values.append(logps.masked_fill(~mask[start:end], 0).sum(-1) / mask[start:end].sum(-1))
            means.append(torch.cat(values))
        delta = means[0] - means[1]
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Non-finite source-teacher margin")
        tau = float(self.config["tau"])
        max_delta = float(self.config.get("max_delta", 10.0))
        clamped_delta = delta.clamp(min=-max_delta, max=max_delta)
        return torch.tanh(clamped_delta / tau).tolist(), delta.tolist()
