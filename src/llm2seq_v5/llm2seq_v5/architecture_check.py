"""Load a real LLM2Seq-v5 graph and verify its architectural invariants."""

from __future__ import annotations

import argparse
import json

import torch

from .checkpoint import load_last_checkpoint
from .config import load_config
from .data import decoder_seed_ids, encode_source
from .decoder import QwenDecoderLayerWithCrossAttention
from .generation import generate
from .model import LLM2SeqV5
from .training import _tokenizers, verify_declared_parameter_budget


@torch.inference_mode()
def probe_future_token_influence(
    encoder: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    vocab_size: int,
    pad_token_id: int | None = None,
) -> dict:
    """Measure whether changing a later token alters an earlier token state.

    Config flags are useful but not sufficient for custom Hub code. This
    behavioral probe catches an accidentally causal attention mask while
    preserving sequence length and every attention-mask position.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Future-context probe requires input_ids shaped [1, sequence]")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("Future-context probe mask must match input_ids")
    valid_positions = torch.nonzero(attention_mask[0].bool(), as_tuple=False).flatten()
    if valid_positions.numel() < 4:
        raise ValueError("Future-context probe requires at least four unmasked tokens")
    query_position = int(valid_positions[1])
    changed_position = int(valid_positions[-2])
    if changed_position <= query_position:
        query_position = int(valid_positions[0])
        changed_position = int(valid_positions[-1])
    if changed_position <= query_position:
        raise ValueError("Could not choose ordered query/change positions for future-context probe")
    if vocab_size <= 1:
        raise ValueError("Future-context probe requires a vocabulary larger than one token")

    changed_ids = input_ids.clone()
    original_id = int(changed_ids[0, changed_position])
    replacement_id = (original_id + 1) % int(vocab_size)
    forbidden = {original_id}
    if pad_token_id is not None:
        forbidden.add(int(pad_token_id))
    while replacement_id in forbidden:
        replacement_id = (replacement_id + 1) % int(vocab_size)
    changed_ids[0, changed_position] = replacement_id

    was_training = encoder.training
    encoder.eval()
    try:
        original_states = encoder(input_ids, attention_mask)[-1]
        changed_states = encoder(changed_ids, attention_mask)[-1]
    finally:
        encoder.train(was_training)
    original_query = original_states[0, query_position].float()
    changed_query = changed_states[0, query_position].float()
    absolute_l2 = torch.linalg.vector_norm(changed_query - original_query)
    relative_l2 = absolute_l2 / torch.linalg.vector_norm(original_query).clamp_min(1e-12)
    return {
        "query_position": query_position,
        "changed_position": changed_position,
        "original_token_id": original_id,
        "replacement_token_id": replacement_id,
        "absolute_l2_change": float(absolute_l2),
        "relative_l2_change": float(relative_l2),
    }


@torch.inference_mode()
def check(config_path: str, checkpoint_path: str | None = None) -> dict:
    config = load_config(config_path)
    encoder_tokenizer, decoder_tokenizer = _tokenizers(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLM2SeqV5(config).to(device).eval()
    cross_layers = [
        layer for layer in model.decoder.backbone.layers if isinstance(layer, QwenDecoderLayerWithCrossAttention)
    ]
    decoder_layer_count = len(model.decoder.backbone.layers)
    cross_every = int(config["decoder"].get("cross_attention_every", 1))
    expected_cross = sum(
        (index + 1) % cross_every == 0 or index == decoder_layer_count - 1 for index in range(decoder_layer_count)
    )
    if len(cross_layers) != expected_cross:
        raise RuntimeError(f"Expected cross-attention in {expected_cross} layers, found {len(cross_layers)}")
    if bool(config["decoder"].get("initialize_cross_from_self", True)):
        first = cross_layers[0]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            copied = getattr(first.cross_attn, name).weight
            native = getattr(first.base_layer.self_attn, name).weight
            if not torch.equal(copied, native):
                raise RuntimeError(f"{name} was not copied exactly from native self-attention")

    # The equality above is an *initialization* invariant.  Check it on the
    # freshly constructed graph, before loading a trained checkpoint.  After
    # fine-tuning, copied cross-attention and native self-attention intentionally
    # use different parameter groups/LRs and must be allowed to diverge.
    checkpoint_payload = None
    if checkpoint_path is not None:
        checkpoint_payload = load_last_checkpoint(model, checkpoint_path)
        model.to(device).eval()
    parameter_summary = model.parameter_summary()
    parameter_budget = verify_declared_parameter_budget(config, parameter_summary)

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
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)
    source_mask = torch.ones_like(source_tensor)
    future_context_probe = probe_future_token_influence(
        model.encoder,
        source_tensor,
        source_mask,
        vocab_size=int(model.encoder.config.vocab_size),
        pad_token_id=encoder_tokenizer.pad_token_id,
    )
    attention_mode = str(config["model"].get("encoder_attention_mode", "auto"))
    min_future_change = float(config["model"].get("encoder_future_context_min_relative_change", 1e-6))
    if attention_mode == "bidirectional" and (future_context_probe["relative_l2_change"] <= min_future_change):
        raise RuntimeError(
            "Encoder is declared bidirectional, but changing a future token did not "
            f"change an earlier final-layer state enough: {future_context_probe}"
        )
    if attention_mode == "causal" and (future_context_probe["relative_l2_change"] > min_future_change):
        raise RuntimeError(
            "Encoder is declared causal, but an earlier final-layer token changed "
            f"after replacing a future token: {future_context_probe}"
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
    decoder_input = torch.tensor([seed + target[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor([[-100] * (len(seed) - 1) + target], dtype=torch.long, device=device)
    output = model(
        input_ids=source_tensor,
        attention_mask=source_mask,
        unit_ids=torch.tensor([unit_ids], dtype=torch.long, device=device),
        decoder_input_ids=decoder_input,
        decoder_attention_mask=torch.ones_like(decoder_input),
        labels=labels,
    )
    if not torch.isfinite(output["loss"]):
        raise RuntimeError("Real-model architecture check produced a non-finite loss")
    encoded = model.encode(
        input_ids=source_tensor,
        attention_mask=source_mask,
        unit_ids=torch.tensor([unit_ids], dtype=torch.long, device=device),
    )
    changed_source_tensor = source_tensor.clone()
    changed_source_tensor[
        0,
        int(future_context_probe["changed_position"]),
    ] = int(future_context_probe["replacement_token_id"])
    changed_encoded = model.encode(
        input_ids=changed_source_tensor,
        attention_mask=source_mask,
        unit_ids=torch.tensor([unit_ids], dtype=torch.long, device=device),
    )
    original_bridge_memory = encoded.memory[:, -1] if encoded.memory.ndim == 4 else encoded.memory
    changed_bridge_memory = (
        changed_encoded.memory[:, -1] if changed_encoded.memory.ndim == 4 else changed_encoded.memory
    )
    bridge_query_position = int(future_context_probe["query_position"])
    original_bridge_query = original_bridge_memory[0, bridge_query_position].float()
    changed_bridge_query = changed_bridge_memory[0, bridge_query_position].float()
    bridge_absolute_l2 = torch.linalg.vector_norm(changed_bridge_query - original_bridge_query)
    bridge_relative_l2 = bridge_absolute_l2 / torch.linalg.vector_norm(original_bridge_query).clamp_min(1e-12)
    bridge_future_context_probe = {
        "query_position": bridge_query_position,
        "changed_position": int(future_context_probe["changed_position"]),
        "absolute_l2_change": float(bridge_absolute_l2),
        "relative_l2_change": float(bridge_relative_l2),
    }
    if len(model.adapter.bidirectional_layers) > 0 and float(bridge_relative_l2) <= min_future_change:
        raise RuntimeError(
            "The bridge has bidirectional refinement layers, but replacing a future "
            f"source token did not affect an earlier memory state: {bridge_future_context_probe}"
        )
    expected_slots = int(config["adapter"].get("num_summary_slots", 16))
    if bool(config["adapter"].get("use_summary_planner", True)):
        if encoded.summary_prefix.shape != (1, expected_slots, int(model.decoder.config.hidden_size)):
            raise RuntimeError(f"Prospective-summary prefix has the wrong shape: {tuple(encoded.summary_prefix.shape)}")
        if not bool(encoded.summary_prefix_mask.all()):
            raise RuntimeError("A generated summary slot was unexpectedly masked")
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
        input_ids=source_tensor,
        attention_mask=source_mask,
        unit_ids=torch.tensor([unit_ids], dtype=torch.long, device=device),
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
        device=device,
    )
    training_mask = torch.zeros(2, source_width, dtype=torch.long, device=device)
    training_units = torch.zeros(2, source_width, dtype=torch.long, device=device)
    for row, (ids, units) in enumerate(((source_ids, unit_ids), (alternate_source_ids, alternate_unit_ids))):
        training_input[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        training_mask[row, : len(ids)] = 1
        training_units[row, : len(units)] = torch.tensor(units, dtype=torch.long, device=device)
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
        decoder_attention_mask=torch.ones_like(decoder_input.expand(2, -1)),
        labels=labels.expand(2, -1).clone(),
    )
    objective_names = (
        "loss",
        "loss_response_alignment",
        "loss_phrase_mixture",
        "loss_phrase_copy",
        "loss_phrase_continue",
        "loss_phrase_labels",
        "loss_phrase_coverage",
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
        decoder_attention_mask=torch.ones_like(decoder_input.expand(2, -1)),
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
    phrase_mixture_weight = (
        float(getattr(model, "phrase_mixture_weight", 0.0))
        if getattr(model, "phrase_pointer", None) is not None
        else 0.0
    )
    expected_eval_loss = (
        (1.0 - phrase_mixture_weight) * diagnostic_output["loss_ce"].float()
        + phrase_mixture_weight * diagnostic_output["loss_phrase_mixture"].float()
        + model.salience_weight * diagnostic_output["loss_salience"].float()
    )
    if not torch.allclose(diagnostic_output["loss"].float(), expected_eval_loss, atol=1e-6):
        raise RuntimeError("Validation diagnostics leaked auxiliary objectives into eval loss")
    summary = {
        **parameter_summary,
        "parameter_budget": parameter_budget,
        "device": str(device),
        "encoder_vocab_config": int(model.encoder.config.vocab_size),
        "decoder_vocab_config": int(model.decoder.config.vocab_size),
        "encoder_tokenizer_size": len(encoder_tokenizer),
        "decoder_tokenizer_size": len(decoder_tokenizer),
        "encoder_source_add_special_tokens": bool(config["data"].get("source_add_special_tokens", False)),
        "encoder_source_special_wrapper": getattr(
            encoder_tokenizer,
            "_llm2seq_v5_default_special_wrapper",
            ([], []),
        ),
        "encoder_attention_mode_declared": attention_mode,
        "encoder_future_context_probe": future_context_probe,
        "bridge_future_context_probe": bridge_future_context_probe,
        "encoder_future_context_min_relative_change": min_future_change,
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
        "training_response_alignment_loss": float(training_output["loss_response_alignment"]),
        "training_phrase_mixture_loss": float(training_output["loss_phrase_mixture"]),
        "training_phrase_continue_loss": float(training_output["loss_phrase_continue"]),
        "training_routing_balance_loss": float(training_output["loss_routing_balance"]),
        "heldout_diagnostics_checked": list(diagnostic_names),
        "heldout_prompt_retrieval_accuracy": float(diagnostic_output["prompt_retrieval_accuracy"]),
        "heldout_source_swap_accuracy": float(diagnostic_output["source_swap_accuracy"]),
        "heldout_source_swap_nll_gap": float(diagnostic_output["source_swap_nll_gap"]),
        "heldout_source_swap_negative_similarity": float(diagnostic_output["source_swap_negative_similarity"]),
        "final_graph": "one_encoder_one_adapter_one_decoder",
        "checkpoint_checked": checkpoint_path,
        "checkpoint_epoch": (
            int(checkpoint_payload["epoch"])
            if checkpoint_payload is not None and checkpoint_payload.get("epoch") is not None
            else None
        ),
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_embedding_0_6b_phrase_continuation.yaml")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained last.pt to audit instead of only the freshly initialized graph.",
    )
    args = parser.parse_args()
    check(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
