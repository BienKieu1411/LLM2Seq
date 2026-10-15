"""Fail-fast checks for vLLM modes not validated by the SEAM adapter."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_decoder_config(config: Any) -> None:
    """Restrict this adapter to the causal Qwen3 decoder used in AFMR."""

    if getattr(config, "model_type", None) != "qwen3":
        raise ValueError(
            "SEAMForQwen3 supports Qwen3 causal decoders only; "
            f"received model_type={getattr(config, 'model_type', None)!r}"
        )
    if not bool(getattr(config, "is_causal", True)):
        raise ValueError("SEAMForQwen3 requires a causal Qwen3 decoder (is_causal=true)")
    if bool(getattr(config, "attention_bias", False)):
        raise ValueError("SEAMForQwen3 currently requires Qwen3 attention_bias=false")


def validate_engine_options(options: Mapping[str, Any]) -> None:
    """Reject facade overrides that bypass SEAM's model/runtime contract."""

    protected = {
        "model",
        "hf_overrides",
        "trust_remote_code",
        "generation_config",
        "tokenizer",
        "tokenizer_mode",
        "tokenizer_revision",
        "skip_tokenizer_init",
        "dtype",
        "override_attention_dtype",
    } & options.keys()
    if protected:
        names = ", ".join(sorted(protected))
        raise ValueError(f"SEAM owns vLLM option(s) {names}; they must match the trained decoder")

    kv_cache_dtype = options.get("kv_cache_dtype", "auto")
    if kv_cache_dtype != "auto":
        raise NotImplementedError("SEAM currently requires kv_cache_dtype='auto' for cross-attention parity")

    if int(options.get("tensor_parallel_size", 1)) != 1:
        raise NotImplementedError("SEAM vLLM currently supports tensor_parallel_size=1 only")
    if int(options.get("pipeline_parallel_size", 1)) != 1:
        raise NotImplementedError("SEAM vLLM currently supports pipeline_parallel_size=1 only")
    if options.get("enforce_eager", True) is not True:
        raise NotImplementedError("SEAM request-side copy state requires enforce_eager=True")
    if options.get("enable_prefix_caching", False) is not False:
        raise NotImplementedError("SEAM has not validated vLLM prefix caching; set enable_prefix_caching=False")
    if options.get("speculative_config") is not None:
        raise NotImplementedError("SEAM vLLM has not validated speculative decoding")
    if options.get("quantization") is not None or options.get("quantization_config") is not None:
        raise NotImplementedError("SEAM vLLM currently requires unquantized model weights")


def validate_vllm_config(vllm_config: Any) -> None:
    """Enforce the same runtime boundary for users constructing ``vllm.LLM`` directly."""

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    if bool(model_config.skip_tokenizer_init):
        raise NotImplementedError("SEAM requires vLLM tokenizer initialization for prompts and output text")
    if not bool(model_config.enforce_eager):
        raise NotImplementedError("SEAM request-side copy state requires enforce_eager=True")
    if bool(cache_config.enable_prefix_caching):
        raise NotImplementedError("SEAM has not validated vLLM prefix caching; disable enable_prefix_caching")
    if vllm_config.speculative_config is not None:
        raise NotImplementedError("SEAM vLLM has not validated speculative decoding")
    if vllm_config.quant_config is not None:
        raise NotImplementedError("SEAM vLLM currently requires unquantized model weights")
    if cache_config.cache_dtype != "auto":
        raise NotImplementedError("SEAM currently requires kv_cache_dtype='auto' for cross-attention parity")
    if getattr(model_config, "override_attention_dtype", None) is not None:
        raise NotImplementedError("SEAM has not validated override_attention_dtype")


def validate_tokenizer_config(vllm_config: Any, expected_decoder_path: str) -> None:
    """Ensure prompt tokenization, copy alignment, and detokenization use the trained decoder."""

    model_config = vllm_config.model_config
    expected_path = Path(expected_decoder_path).expanduser().resolve()
    for name in ("model", "tokenizer"):
        value = getattr(model_config, name, None)
        if not isinstance(value, (str, Path)) or not value:
            raise ValueError(f"SEAM requires vLLM {name} to point to the local trained decoder")
        actual_path = Path(value).expanduser().resolve()
        if actual_path != expected_path:
            raise ValueError(
                f"SEAM requires vLLM {name}={expected_path}; received {actual_path}. "
                "The decoder tokenizer must stay aligned with prompt tokenization, source-copy alignment, "
                "and output detokenization."
            )

    if getattr(model_config, "tokenizer_mode", None) != "auto":
        raise NotImplementedError("SEAM has only validated vLLM tokenizer_mode='auto'")
    if getattr(model_config, "tokenizer_revision", None) != getattr(model_config, "revision", None):
        raise ValueError("SEAM requires tokenizer_revision to match the decoder revision")


def validate_compute_dtype(vllm_config: Any, expected_dtype: str) -> None:
    """Require vLLM to use AFMR's CUDA evaluation precision."""

    actual_dtype = str(vllm_config.model_config.dtype).rsplit(".", maxsplit=1)[-1]
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"SEAM requires vLLM dtype={expected_dtype!r} to match AFMR evaluation; got {actual_dtype!r}. "
            "Set dtype from model.compute_dtype in resolved_config.yaml."
        )
