#!/usr/bin/env python3
"""Offline XOV smoke test using the repository's tiny encoder/decoder fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from eviseq_xov.config import load_config  # noqa: E402
from eviseq_xov.evaluation.generate import generate_greedy  # noqa: E402
from eviseq_xov.modeling.model import EviSeqXOV  # noqa: E402
from eviseq_xov.runtime import build_loaders  # noqa: E402


def main() -> None:
    config_path = PACKAGE_ROOT / "configs" / "xov_smoke.yaml"
    config = load_config(config_path)
    loaders = build_loaders(config, max_train_examples=2, max_validation_examples=2)
    batch = next(iter(loaders["train"]))
    model = EviSeqXOV(config)
    alignment = {
        key: batch[key]
        for key in (
            "copy_token_ids",
            "copy_token_mask",
            "copy_encoder_indices",
            "copy_token_indices",
            "copy_alignment_weights",
        )
        if key in batch
    }
    output = model(
        batch["input_ids"],
        batch["attention_mask"],
        batch["source_content_mask"],
        batch["decoder_prompt_ids"],
        batch["decoder_prompt_mask"],
        batch["decoder_input_ids"],
        batch["decoder_attention_mask"],
        batch["labels"],
        return_logits=False,
        **alignment,
    )
    if output.loss_ce is None or not torch.isfinite(output.loss_ce):
        raise RuntimeError("XOV smoke forward did not produce a finite CE loss")
    output.loss_ce.backward()
    test_loader = build_loaders(config, split="test", batch_size_override=2)["test"]
    test_loader.collate_fn.include_targets = False
    test_batch = next(iter(test_loader))
    generate_greedy(
        model,
        test_batch,
        test_loader.collate_fn.decoder_tokenizer,
        max_new_tokens=2,
        min_new_tokens=0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        compact_finished=True,
    )
    print("XOV smoke test passed: config, forward, backward, and cached generation")


if __name__ == "__main__":
    main()
