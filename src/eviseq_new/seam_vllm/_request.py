"""Shared request construction for the Python and HTTP interfaces."""

from __future__ import annotations

from typing import Any

from ._prompt import configured_decoder_prompt_ids


def request_prompt_ids(tokenizer, data, *, decoder_prompt=None, decoder_prefix=None, decoder_chat_template=None):
    options = dict(data)
    for name, value in (("decoder_prompt", decoder_prompt), ("decoder_prefix", decoder_prefix)):
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"{name} must be text")
            options[name] = value
    if decoder_chat_template is not None:
        if not isinstance(decoder_chat_template, bool):
            raise ValueError("decoder_chat_template must be a boolean")
        options["decoder_chat_template"] = decoder_chat_template
    return configured_decoder_prompt_ids(tokenizer, options)


def source_prompt(
    source: str,
    prompt_ids: list[int],
    *,
    source_limit: int,
    output_budget: int,
    encoder_prompt: str | None = None,
) -> dict[str, Any]:
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Every SEAM source must be non-empty text")
    for name, value in (("max_input_tokens", source_limit), ("max_tokens", output_budget)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not prompt_ids or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in prompt_ids):
        raise ValueError("SEAM requires non-empty decoder prompt token IDs")
    kwargs = {
        "seam_output_budget": output_budget,
        "seam_max_source_length": source_limit,
        # Also condition the bridge on this exact request's decoder prompt.
        "seam_decoder_prompt_ids": list(prompt_ids),
    }
    if encoder_prompt is not None:
        if not isinstance(encoder_prompt, str):
            raise ValueError("encoder_prompt must be text")
        kwargs["seam_encoder_prompt"] = encoder_prompt
    return {
        "prompt_token_ids": list(prompt_ids),
        "multi_modal_data": {"text": source},
        "mm_processor_kwargs": kwargs,
    }
