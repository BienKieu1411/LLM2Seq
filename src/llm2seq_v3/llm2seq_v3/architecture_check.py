"""Load the real 0.6B+0.6B graph and verify its architectural invariants (v3)."""

from __future__ import annotations

import argparse
import json

import torch

from .config import load_config
from .data import decoder_seed_ids, encode_source
from .decoder import QwenDecoderLayerWithCrossAttention
from .generation import generate
from .model import LLM2SeqV3
from .training import _tokenizers, verify_declared_parameter_budget


@torch.inference_mode()
def check(config_path: str) -> dict:
    config = load_config(config_path)
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    model = LLM2SeqV3(config).eval()
    parameter_summary = model.parameter_summary()
    parameter_budget = verify_declared_parameter_budget(config, parameter_summary)
    cross_layers = [
        layer for layer in model.decoder.backbone.layers if isinstance(layer, QwenDecoderLayerWithCrossAttention)
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

    # Verify the actual optimization contract, not only the module graph.
    # Phase 1 must leave pretrained backbones untouched while training every
    # newly introduced interface. Phase 2 is the requested full fine-tune.
    model.set_training_stage("interface_warmup")
    warmup_trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if any(name.startswith("encoder.") for name in warmup_trainable):
        raise RuntimeError("Interface warm-up unexpectedly trains the pretrained encoder")
    if not any(name.startswith("adapter.") for name in warmup_trainable):
        raise RuntimeError("Interface warm-up left the adapter frozen")
    if model.alignment_head is not None and not any(name.startswith("alignment_head.") for name in warmup_trainable):
        raise RuntimeError("Interface warm-up left the alignment head frozen")
    if not any(".cross_attn." in name for name in warmup_trainable):
        raise RuntimeError("Interface warm-up left copied cross-attention frozen")
    if int(config["decoder"].get("memory_bank_count", 1)) > 1 and not any(
        name.endswith(".memory_router_logits") for name in warmup_trainable
    ):
        raise RuntimeError("Interface warm-up left HiRoute routers frozen")
    if bool(config["decoder"].get("query_adaptive_routing", False)) and not any(
        ".memory_router_proj." in name for name in warmup_trainable
    ):
        raise RuntimeError("Interface warm-up left query-adaptive routers frozen")
    unexpected_warmup_decoder = [
        name
        for name in warmup_trainable
        if name.startswith("decoder.")
        and ".cross_attn" not in name
        and not name.endswith(".cross_gate")
        and ".memory_router" not in name
    ]
    if unexpected_warmup_decoder:
        raise RuntimeError(
            "Interface warm-up unexpectedly trains native decoder parameters: "
            + ", ".join(unexpected_warmup_decoder[:10])
        )
    warmup_trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.set_training_stage("full_finetune")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Full-finetune stage did not enable every model parameter")
    full_trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    source_ids, unit_ids, _ = encode_source(
        encoder_tokenizer,
        "Mở cài đặt. Chọn mục hệ thống. Khởi động lại thiết bị.",
        {**config["data"], "max_source_length": 64},
    )
    alternate_source_ids, alternate_unit_ids, _ = encode_source(
        encoder_tokenizer,
        "Đun sôi nước. Cho trà vào cốc. Rót nước và chờ ba phút.",
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
    encoded = model.encode(
        input_ids=torch.tensor([source_ids], dtype=torch.long),
        attention_mask=torch.ones(1, len(source_ids), dtype=torch.long),
        unit_ids=torch.tensor([unit_ids], dtype=torch.long),
    )
    expected_banks = int(config["decoder"].get("memory_bank_count", 1))
    actual_banks = int(encoded.memory.shape[1]) if encoded.memory.ndim == 4 else 1
    if actual_banks != expected_banks:
        raise RuntimeError(f"Adapter produced {actual_banks} memory banks, expected {expected_banks}")
    model.decoder.prepare_cross_attention_cache(encoded.memory)
    cached_banks = int(cross_layers[0].cross_attn._memory_cache[0].shape[1])
    expected_cached_banks = (
        expected_banks
        if str(config["decoder"].get("memory_routing_mode", "attention_output")) == "attention_output"
        else 1
    )
    model.decoder.clear_cross_attention_cache()
    if cached_banks != expected_cached_banks:
        raise RuntimeError(f"Cross-attention cached {cached_banks} banks, expected {expected_cached_banks}")
    generated = generate(
        model,
        input_ids=torch.tensor([source_ids], dtype=torch.long),
        attention_mask=torch.ones(1, len(source_ids), dtype=torch.long),
        unit_ids=torch.tensor([unit_ids], dtype=torch.long),
        decoder_seed=seed,
        max_new_tokens=2,
        min_new_tokens=0,
        eos_token_id=None,
        pad_token_id=decoder_tokenizer.pad_token_id,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
    )
    if generated.shape != (1, 2):
        raise RuntimeError(f"Cached generation produced unexpected shape {tuple(generated.shape)}")
    if any(layer.cross_attn._memory_cache is not None for layer in cross_layers):
        raise RuntimeError("Generation left a stale multi-bank cross-attention cache")

    # Exercise every training-only objective on the real combined graph. No
    # backward is needed here: unit tests separately prove gradient flow, while
    # this check catches shape/dtype/integration errors across full checkpoints.
    source_width = max(len(source_ids), len(alternate_source_ids))
    training_input = torch.full(
        (2, source_width),
        int(encoder_tokenizer.pad_token_id),
        dtype=torch.long,
    )
    training_mask = torch.zeros(2, source_width, dtype=torch.long)
    training_units = torch.zeros(2, source_width, dtype=torch.long)
    for row, (ids, units) in enumerate(((source_ids, unit_ids), (alternate_source_ids, alternate_unit_ids))):
        training_input[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        training_mask[row, : len(ids)] = 1
        training_units[row, : len(units)] = torch.tensor(units, dtype=torch.long)
    if hasattr(model.encoder.model, "gradient_checkpointing_disable"):
        model.encoder.model.gradient_checkpointing_disable()
    if hasattr(model.decoder.backbone, "gradient_checkpointing_disable"):
        model.decoder.backbone.gradient_checkpointing_disable()
    model.train()
    training_output = model(
        input_ids=training_input,
        attention_mask=training_mask,
        unit_ids=training_units,
        decoder_input_ids=decoder_input.expand(2, -1).clone(),
        decoder_attention_mask=torch.ones(2, decoder_input.shape[1], dtype=torch.long),
        labels=labels.expand(2, -1).clone(),
    )
    objective_names = (
        "loss",
        "loss_contrastive",
        "loss_source_swap",
        "loss_routing_balance",
        "source_swap_nll_gap",
    )
    if not all(torch.isfinite(training_output[name]) for name in objective_names):
        raise RuntimeError("Real-model training objective check produced a non-finite value")
    model.eval()
    diagnostic_output = model(
        input_ids=training_input,
        attention_mask=training_mask,
        unit_ids=training_units,
        decoder_input_ids=decoder_input.expand(2, -1).clone(),
        decoder_attention_mask=torch.ones(2, decoder_input.shape[1], dtype=torch.long),
        labels=labels.expand(2, -1).clone(),
        compute_source_diagnostics=True,
    )
    diagnostic_names = (
        "loss_contrastive",
        "prompt_retrieval_accuracy",
        "loss_source_swap",
        "source_swap_nll_gap",
        "source_swap_accuracy",
        "source_swap_negative_similarity",
        "memory_routing_entropy",
        "adaptive_routing_delta",
    )
    if not all(torch.isfinite(diagnostic_output[name]) for name in diagnostic_names):
        raise RuntimeError("Held-out source-utilization diagnostics produced a non-finite value")
    expected_eval_loss = (
        diagnostic_output["loss_ce"].float() + model.salience_weight * diagnostic_output["loss_salience"].float()
    )
    if not torch.allclose(diagnostic_output["loss"].float(), expected_eval_loss, atol=1e-6):
        raise RuntimeError("Validation diagnostics leaked auxiliary objectives into eval loss")
    summary = {
        **parameter_summary,
        "parameter_budget": parameter_budget,
        "encoder_vocab_config": int(model.encoder.config.vocab_size),
        "decoder_vocab_config": int(model.decoder.config.vocab_size),
        "encoder_tokenizer_size": len(encoder_tokenizer),
        "decoder_tokenizer_size": len(decoder_tokenizer),
        "cross_attention_layers": len(cross_layers),
        "bidirectional_adapter_layers": len(model.adapter.bidirectional_layers),
        "copied_cross_attention": bool(config["decoder"].get("initialize_cross_from_self", True)),
        "loss": float(output["loss"]),
        "cached_generation_tokens": int(generated.shape[1]),
        "adapter_memory_banks": actual_banks,
        "cross_attention_cached_banks": cached_banks,
        "memory_routing_mode": str(config["decoder"].get("memory_routing_mode", "attention_output")),
        "query_adaptive_routing": bool(config["decoder"].get("query_adaptive_routing", False)),
        "warmup_trainable_parameters": warmup_trainable_parameters,
        "full_trainable_parameters": full_trainable_parameters,
        "training_stage_contract_checked": True,
        "training_objectives_checked": list(objective_names),
        "training_loss": float(training_output["loss"]),
        "training_contrastive_loss": float(training_output["loss_contrastive"]),
        "training_source_swap_loss": float(training_output["loss_source_swap"]),
        "training_routing_balance_loss": float(training_output["loss_routing_balance"]),
        "heldout_diagnostics_checked": list(diagnostic_names),
        "heldout_prompt_retrieval_accuracy": float(diagnostic_output["prompt_retrieval_accuracy"]),
        "heldout_source_swap_accuracy": float(diagnostic_output["source_swap_accuracy"]),
        "heldout_source_swap_nll_gap": float(diagnostic_output["source_swap_nll_gap"]),
        "heldout_source_swap_negative_similarity": float(diagnostic_output["source_swap_negative_similarity"]),
        "final_graph": "one_encoder_one_adapter_one_decoder",
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_0_6b_hiroute.yaml")
    args = parser.parse_args()
    check(args.config)


if __name__ == "__main__":
    main()
