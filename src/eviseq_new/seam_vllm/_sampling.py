"""Translate AFMR generation names to vLLM sampling options."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def sampling_kwargs(generation: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    options = dict(overrides)
    for alias, native in (("max_new_tokens", "max_tokens"), ("min_new_tokens", "min_tokens")):
        if alias in options:
            if native in options:
                raise ValueError(f"Pass either {alias} or {native}, not both")
            options[native] = options.pop(alias)
    for key, supported in (("num_beams", 1), ("no_repeat_ngram_size", 0)):
        if int(options.pop(key, supported)) != supported:
            raise NotImplementedError(f"SEAM vLLM requires {key}={supported}")

    params = {
        "max_tokens": int(generation.get("max_new_tokens", 256)),
        "min_tokens": int(generation.get("min_new_tokens", 0)),
        "temperature": float(generation.get("temperature", 0.0)),
        "top_p": float(generation.get("top_p", 1.0)),
        "top_k": int(generation.get("top_k", 0)),
        # AFMR excludes chat prompt tokens from this penalty; vLLM includes them.
        # Use the native neutral default unless explicitly requested by the caller.
        "repetition_penalty": 1.0,
    }
    do_sample = options.pop("do_sample", None)
    explicit_temperature = "temperature" in options
    params.update(options)
    if do_sample is True and float(params["temperature"]) <= 0:
        raise ValueError("do_sample=True requires temperature > 0")
    if do_sample is False or (
        do_sample is None and not explicit_temperature and not bool(generation.get("do_sample", False))
    ):
        params["temperature"] = 0.0
    return params


def validate_sampling_params(params: Any) -> None:
    # The bridge uses this same finite budget; it cannot infer an unbounded one.
    if params.max_tokens is None or params.max_tokens < 1:
        raise ValueError("SEAM requires a finite, positive max_tokens output budget")
    if params.prompt_logprobs is not None:
        raise NotImplementedError("SEAM copy state does not support prompt_logprobs; output logprobs are supported")
