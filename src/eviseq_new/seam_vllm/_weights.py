"""Dependency-light helpers for AFMR checkpoint key translation."""

from __future__ import annotations

from collections.abc import Iterable


def expected_qwen3_decoder_checkpoint_keys(num_hidden_layers: int, tie_word_embeddings: bool) -> set[str]:
    """Return the AFMR state-dict keys needed to fully restore a Qwen3 decoder."""

    keys = {
        "decoder.backbone.embed_tokens.weight",
        "decoder.backbone.norm.weight",
    }
    layer_weights = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for layer_index in range(num_hidden_layers):
        prefix = f"decoder.backbone.layers.{layer_index}.base."
        keys.update(prefix + name for name in layer_weights)
    if not tie_word_embeddings:
        keys.add("decoder.lm_head.weight")
    return keys


def missing_parameter_names(expected: Iterable[str], loaded: Iterable[str]) -> list[str]:
    """List expected vLLM decoder parameters not reported as loaded."""

    return sorted(set(expected) - set(loaded))


def normalize_decoder_lm_head_key(key: str) -> str:
    """Map the AFMR wrapper's output-head key to vLLM's root-level name."""

    prefix = "decoder.lm_head."
    if key.startswith(prefix):
        return key.removeprefix("decoder.")
    return key
