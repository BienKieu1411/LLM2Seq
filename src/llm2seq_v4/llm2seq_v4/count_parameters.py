"""Exact parameter counts from configs without downloading model weights (v4)."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import torch

from .adapter import ProspectiveSummaryBridge
from .config import load_config
from .contrastive import SourceAlignmentHead


def _model_from_config(
    name: str,
    causal_lm: bool,
    *,
    trust_remote_code: bool = True,
    revision: str | None = None,
):
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    pretrained_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
    }
    if revision is not None:
        pretrained_kwargs["revision"] = str(revision)
    config = AutoConfig.from_pretrained(name, **pretrained_kwargs)
    cls = AutoModelForCausalLM if causal_lm else AutoModel
    with torch.device("meta"):
        model = cls.from_config(config, trust_remote_code=bool(trust_remote_code))
    return model, config


def count(config_path: str) -> Dict[str, Any]:
    config = load_config(config_path)
    model_config = config["model"]
    encoder, encoder_config = _model_from_config(
        model_config["encoder_name"],
        False,
        trust_remote_code=bool(model_config.get("encoder_trust_remote_code", True)),
        revision=(str(model_config["encoder_revision"]) if model_config.get("encoder_revision") is not None else None),
    )
    decoder, decoder_config = _model_from_config(model_config["decoder_name"], True)
    encoder_parameters = sum(parameter.numel() for parameter in encoder.parameters())
    decoder_parameters = sum(parameter.numel() for parameter in decoder.parameters())

    decoder_core = getattr(decoder, "model", decoder)
    layers = list(decoder_core.layers)
    every = int(config["decoder"].get("cross_attention_every", 1))
    cross_indices = [index for index in range(len(layers)) if (index + 1) % every == 0 or index == len(layers) - 1]
    cross_parameters = 0
    memory_bank_count = int(config["decoder"].get("memory_bank_count", 1))
    for index in cross_indices:
        layer = layers[index]
        cross_parameters += sum(parameter.numel() for parameter in layer.self_attn.parameters())
        # QwenCopiedCrossAttention has its own memory norm; the injected layer
        # also has its own query-side cross-attention norm.
        cross_parameters += 2 * sum(parameter.numel() for parameter in layer.input_layernorm.parameters())
        cross_parameters += 1  # learned residual gate
        if memory_bank_count > 1:
            cross_parameters += memory_bank_count  # learned depth-router logits
            if bool(config["decoder"].get("query_adaptive_routing", False)):
                # One copied RMSNorm plus a zero-initialized D -> K router.
                cross_parameters += sum(parameter.numel() for parameter in layer.input_layernorm.parameters())
                cross_parameters += int(decoder_config.hidden_size) * memory_bank_count
                cross_parameters += int(decoder_config.hidden_size) * memory_bank_count

    fallback = int(model_config.get("hidden_size", 0))
    encoder_hidden = int(model_config.get("encoder_hidden_size", fallback or encoder_config.hidden_size))
    decoder_hidden = int(model_config.get("decoder_hidden_size", fallback or decoder_config.hidden_size))
    with torch.device("meta"):
        adapter = ProspectiveSummaryBridge(
            encoder_hidden,
            decoder_hidden,
            config["adapter"],
        )
    adapter_parameters = sum(parameter.numel() for parameter in adapter.parameters())
    objectives = config.get("objectives", {})
    with torch.device("meta"):
        alignment_head = (
            SourceAlignmentHead(
                decoder_hidden,
                int(objectives.get("contrastive_projection_size", 256)),
                pooling=str(objectives.get("contrastive_pooling", "mean_last")),
            )
            if bool(objectives.get("use_contrastive", True)) and bool(objectives.get("use_prompt_alignment", True))
            else None
        )
    alignment_parameters = (
        sum(parameter.numel() for parameter in alignment_head.parameters()) if alignment_head is not None else 0
    )
    deployable_parameters = encoder_parameters + adapter_parameters + decoder_parameters + cross_parameters
    result = {
        "config": config_path,
        "encoder_name": model_config["encoder_name"],
        "encoder_revision": model_config.get("encoder_revision"),
        "decoder_name": model_config["decoder_name"],
        "encoder_parameters": encoder_parameters,
        "adapter_parameters": adapter_parameters,
        "decoder_pretrained_parameters": decoder_parameters,
        "cross_attention_parameters": cross_parameters,
        "prompt_alignment_head_parameters": alignment_parameters,
        "cross_attention_layers": len(cross_indices),
        "total_deployable_parameters": deployable_parameters,
        "training_only_parameters": alignment_parameters,
        "total_training_parameters": deployable_parameters + alignment_parameters,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="May be repeated. Defaults to the Qwen and PPLX prospective-summary profiles.",
    )
    args = parser.parse_args()
    paths = args.config or [
        "configs/qwen3_embedding_0_6b_psb.yaml",
        "configs/pplx_embed_v1_0_6b_psb.yaml",
    ]
    for path in paths:
        count(path)


if __name__ == "__main__":
    main()
