"""Shared decoder-prompt tokenization for the SEAM facade and vLLM processor."""

from __future__ import annotations

from typing import Any


def configured_decoder_prompt_ids(tokenizer: Any, data: dict[str, Any]) -> list[int]:
    instruction = str(data.get("decoder_prompt", "")).strip()
    if bool(data.get("decoder_chat_template", False)):
        if not instruction:
            raise ValueError("decoder_chat_template requires data.decoder_prompt")
        values = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        values = tokenizer(instruction, add_special_tokens=False)["input_ids"] if instruction else []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    result = [int(value) for value in values]
    prefix = str(data.get("decoder_prefix", ""))
    if prefix:
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        result.extend(int(value) for value in prefix_ids)
    if not result:
        start = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        if start is None:
            raise ValueError("SEAM needs a non-empty decoder prompt or a BOS/EOS token")
        result = [int(start)]
    return result


def validate_decoder_prompt_ids(actual: list[int], expected: list[int]) -> None:
    if actual != expected:
        raise ValueError(
            "SEAM's bridge is conditioned on the decoder prompt stored in resolved_config.yaml; "
            "the vLLM request prompt must use those exact token IDs. Use SEAMLLM or pass its configured "
            "prompt_token_ids to LLM.generate()."
        )
