"""Load the real 0.6B+0.6B graph and verify its architectural invariants."""

from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .data import decoder_seed_ids, encode_source
from .decoder import QwenDecoderLayerWithCrossAttention
from .model import LLM2SeqV2
from .training import _tokenizers


@torch.inference_mode()
def check(config_path: str) -> dict:
    config = load_config(config_path)
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    model = LLM2SeqV2(config).eval()
    cross_layers = [
        layer
        for layer in model.decoder.backbone.layers
        if isinstance(layer, QwenDecoderLayerWithCrossAttention)
    ]
    expected_cross = len(model.decoder.backbone.layers)
    if int(config["decoder"].get("cross_attention_every", 1)) == 1 and len(cross_layers) != expected_cross:
        raise RuntimeError(f"Expected cross-attention in {expected_cross} layers, found {len(cross_layers)}")
    if bool(config["decoder"].get("initialize_cross_from_self", True)):
        first = cross_layers[0]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            copied = getattr(first.cross_attn, name).weight
            native = getattr(first.base_layer.self_attn, name).weight
            if not torch.equal(copied, native):
                raise RuntimeError(f"{name} was not copied exactly from native self-attention")

    source_ids, unit_ids, _ = encode_source(
        encoder_tokenizer,
        "Mở cài đặt. Chọn mục hệ thống. Khởi động lại thiết bị.",
        {**config["data"], "max_source_length": 64},
    )
    seed = decoder_seed_ids(decoder_tokenizer, config["data"])
    target = decoder_tokenizer("Khởi động lại thiết bị.", add_special_tokens=False)["input_ids"]
    if decoder_tokenizer.eos_token_id is not None:
        target = [*target, decoder_tokenizer.eos_token_id]
    decoder_input = torch.tensor([seed + target[:-1]], dtype=torch.long)
    labels = torch.tensor([[-100] * (len(seed) - 1) + target], dtype=torch.long)
    output = model(
        input_ids=torch.tensor([source_ids], dtype=torch.long),
        attention_mask=torch.ones(1, len(source_ids), dtype=torch.long),
        unit_ids=torch.tensor([unit_ids], dtype=torch.long),
        decoder_input_ids=decoder_input,
        decoder_attention_mask=torch.ones_like(decoder_input),
        labels=labels,
    )
    if not torch.isfinite(output["loss"]):
        raise RuntimeError("Real-model architecture check produced a non-finite loss")
    summary = {
        **model.parameter_summary(),
        "encoder_vocab_config": int(model.encoder.config.vocab_size),
        "decoder_vocab_config": int(model.decoder.config.vocab_size),
        "encoder_tokenizer_size": len(encoder_tokenizer),
        "decoder_tokenizer_size": len(decoder_tokenizer),
        "cross_attention_layers": len(cross_layers),
        "bidirectional_adapter_layers": len(model.adapter.bidirectional_layers),
        "copied_cross_attention": bool(config["decoder"].get("initialize_cross_from_self", True)),
        "loss": float(output["loss"]),
        "final_graph": "one_encoder_one_adapter_one_decoder",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_0_6b.yaml")
    args = parser.parse_args()
    check(args.config)


if __name__ == "__main__":
    main()
