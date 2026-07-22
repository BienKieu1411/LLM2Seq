#!/usr/bin/env python3
"""Count GenBridge parameters on the meta device without downloading weights."""

from __future__ import annotations

import argparse
import json

import torch
from genbridge.bridge import SummaryBridge
from genbridge.config import MODEL_PROFILES
from genbridge.pretrained_decoder import QwenCrossAttention
from transformers import AutoConfig, AutoModelForCausalLM


def count_profile(size: str) -> dict[str, int | str]:
    profile = MODEL_PROFILES[size]
    model_name = str(profile["name"])
    raw_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config = raw_config.get_text_config() if hasattr(raw_config, "get_text_config") else raw_config
    with torch.device("meta"):
        causal_lm = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        bridge = SummaryBridge(
            int(config.hidden_size),
            int(config.hidden_size),
            {
                "mode": "genbridge",
                "hidden_size": 512,
                "token_num_layers": 4,
                "unit_num_layers": 1,
                "num_heads": 8,
                "ffn_size": 2048,
                "dropout": 0.1,
            },
        )
        cross_attention = QwenCrossAttention(config)

    one_backbone = sum(parameter.numel() for parameter in causal_lm.parameters())
    bridge_parameters = sum(parameter.numel() for parameter in bridge.parameters())
    plan_tokens = 16 * int(config.hidden_size)
    cross_layers = (int(config.num_hidden_layers) + 3) // 4
    # Added module = shared GQA projections/head norms + copied decoder query
    # RMSNorm + copied encoder-memory RMSNorm + scalar residual gate +
    # query-dependent plan gate (H -> 1).
    cross_per_layer = (
        sum(parameter.numel() for parameter in cross_attention.parameters())
        + int(config.hidden_size)
        + int(config.hidden_size)
        + 1
        + int(config.hidden_size)
        + 1
    )
    cross_parameters = cross_layers * cross_per_layer
    total = 2 * one_backbone + bridge_parameters + plan_tokens + cross_parameters
    return {
        "size": size,
        "model_name": model_name,
        "encoder_backbone": one_backbone,
        "decoder_backbone": one_backbone,
        "summary_plan_tokens": plan_tokens,
        "summary_bridge": bridge_parameters,
        "cross_attention_layers": cross_layers,
        "cross_attention": cross_parameters,
        "total": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-size",
        choices=[*sorted(MODEL_PROFILES), "all"],
        default="all",
    )
    args = parser.parse_args()
    sizes = sorted(MODEL_PROFILES) if args.model_size == "all" else [args.model_size]
    print(json.dumps([count_profile(size) for size in sizes], indent=2))


if __name__ == "__main__":
    main()
