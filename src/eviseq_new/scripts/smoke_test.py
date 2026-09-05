#!/usr/bin/env python3
"""Offline AFMR smoke test; never contacts Hugging Face or downloads weights."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eviseq_afmr.modeling.afmr import AdaptiveFullMemoryResidualBridge  # noqa: E402
from eviseq_afmr.modeling.outputs import EncoderState  # noqa: E402
from eviseq_afmr.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402


class TinyEncoder(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 24, taps: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(taps))

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, content_mask: torch.Tensor
    ) -> EncoderState:
        state = self.embedding(input_ids)
        hidden = []
        for layer in self.layers:
            state = torch.tanh(layer(state))
            hidden.append(state)
        return EncoderState(state, tuple(hidden), attention_mask, content_mask)


class TinyDecoder(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 24):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.output = nn.Linear(hidden, vocab)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        source_bias: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        query = self.embed_tokens(input_ids)
        scores = source_bias.masked_fill(~memory_mask, torch.finfo(source_bias.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("bt,btd->bd", weights, memory).unsqueeze(1)
        logits = self.output(query + context)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
        return logits, loss


class TinyAFMR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()
        self.bridge = AdaptiveFullMemoryResidualBridge(
            24,
            24,
            {
                "controller_dim": 12,
                "depth_taps": 2,
                "depth_rank": 8,
                "feature_rank": 8,
                "focus_hidden": 12,
                "focus_windows": [4, 8, 16],
                "focus_overlap": 0.5,
                "depth_gate_init": 0.02,
                "depth_gate_max": 0.15,
                "feature_gate_init": 0.02,
                "feature_gate_max": 0.20,
                "focus_strength_init": 0.10,
                "focus_strength_max": 1.0,
                "temperature_init": 1.0,
                "temperature_min": 0.5,
                "temperature_max": 2.0,
            },
        )
        self.decoder = TinyDecoder()

    def forward(self, batch: dict[str, torch.Tensor]):
        state = self.encoder(batch["input_ids"], batch["attention_mask"], batch["source_content_mask"])
        prompt = self.decoder.embed_tokens(batch["decoder_prompt_ids"])
        bridge = self.bridge(state, prompt, batch["decoder_prompt_mask"], batch["output_budget"])
        logits, ce = self.decoder(
            batch["decoder_input_ids"], bridge.memory, bridge.memory_mask, bridge.source_bias, batch["labels"]
        )
        return ce, bridge


def main() -> None:
    torch.manual_seed(7)
    model = TinyAFMR()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    batch_size, source_len, target_len = 2, 19, 7
    attention = torch.ones(batch_size, source_len, dtype=torch.bool)
    content = attention.clone()
    content[:, :2] = False
    content[:, -1] = False
    labels = torch.randint(0, 64, (batch_size, target_len))
    labels[:, 0] = -100
    batch = {
        "input_ids": torch.randint(0, 64, (batch_size, source_len)),
        "attention_mask": attention,
        "source_content_mask": content,
        "decoder_prompt_ids": torch.randint(0, 64, (batch_size, 3)),
        "decoder_prompt_mask": torch.ones(batch_size, 3, dtype=torch.bool),
        "decoder_input_ids": torch.randint(0, 64, (batch_size, target_len)),
        "labels": labels,
        "output_budget": torch.full((batch_size,), target_len, dtype=torch.float32),
    }
    initial = None
    final = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, bridge = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Smoke loss is not finite")
        if initial is None:
            initial = float(loss.detach())
        loss.backward()
        if bridge.source_bias.grad_fn is None:
            raise RuntimeError("Source prior is detached from the graph")
        optimizer.step()
        final = float(loss.detach())
    config = {
        "model": {"encoder_name": "tiny", "decoder_name": "tiny"},
        "architecture": {
            "name": "afmr_v1",
            "depth_taps": 2,
            "depth_rank": 8,
            "feature_rank": 8,
            "focus_windows": [4, 8, 16],
        },
        "decoder": {"cross_attention_every": 1},
    }
    with tempfile.TemporaryDirectory(prefix="afmr_smoke_") as directory:
        checkpoint = Path(directory) / "last.pt"
        save_checkpoint(checkpoint, model, optimizer, config, epoch=1, step=2)
        restored = TinyAFMR()
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2.0e-3)
        metadata = load_checkpoint(checkpoint, restored, restored_optimizer, config)
    output = {
        "status": "ok",
        "initial_loss": initial,
        "final_loss": final,
        "checkpoint_epoch": metadata["epoch"],
        "checkpoint_step": metadata["step"],
        "source_bias_shape": list(bridge.source_bias.shape),
    }
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
