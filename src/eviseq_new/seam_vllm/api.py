"""Small Python facade over vLLM for source-to-summary SEAM requests."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from eviseq_afmr.config import load_config, resolve_path

from ._bootstrap import ensure_source_plugin
from ._compat import validate_engine_options
from ._prompt import configured_decoder_prompt_ids
from ._request import request_prompt_ids, source_prompt
from ._sampling import sampling_kwargs, validate_sampling_params


class SEAMLLM:
    """Load a local AFMR checkpoint into vLLM and summarize source strings.

    Example::

        from seam_vllm import SEAMLLM

        model = SEAMLLM("runs/exp/resolved_config.yaml", "runs/exp/last.pt")
        summaries = model.summarize(["A source document ..."])
    """

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        encoder_max_batch_tokens: int = 8192,
        **vllm_options: Any,
    ) -> None:
        if (
            isinstance(encoder_max_batch_tokens, bool)
            or not isinstance(encoder_max_batch_tokens, int)
            or encoder_max_batch_tokens < 1
        ):
            raise ValueError("encoder_max_batch_tokens must be a positive integer")
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"SEAM config not found: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SEAM checkpoint not found: {self.checkpoint_path}")
        self.config = load_config(self.config_path, train_only=True)
        model = self.config["model"]
        self.decoder_path = resolve_path(model["decoder_name"], self.config)
        self.encoder_path = resolve_path(model["encoder_name"], self.config)
        for name, path in (("decoder", self.decoder_path), ("encoder", self.encoder_path)):
            if not path.exists():
                raise FileNotFoundError(f"SEAM {name} must be present locally: {path}")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if os.environ.get("VLLM_USE_V2_MODEL_RUNNER", "1") != "1":
            raise ValueError("SEAM requires VLLM_USE_V2_MODEL_RUNNER=1 for request-specific copy state")
        os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
        ensure_source_plugin()
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM

        if vllm.__version__.split("+")[0] != "0.25.1":
            raise RuntimeError(f"SEAM targets vLLM 0.25.1; received {vllm.__version__}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.decoder_path),
            trust_remote_code=bool(model.get("trust_remote_code", True)),
            local_files_only=True,
        )
        if bool(self.config["decoder"].get("grounded_copy", {}).get("enabled", False)) and not getattr(
            self.tokenizer, "is_fast", False
        ):
            raise ValueError("SEAM grounded copy requires the trained decoder's fast tokenizer for source offsets")
        self.prompt_token_ids = configured_decoder_prompt_ids(self.tokenizer, self.config["data"])

        validate_engine_options(vllm_options)
        defaults: dict[str, Any] = {
            "model": str(self.decoder_path),
            "dtype": str(model.get("compute_dtype", model.get("dtype", "float32"))),
            "trust_remote_code": bool(model.get("trust_remote_code", True)),
            "hf_overrides": {
                "architectures": ["SEAMForQwen3"],
                "is_encoder_decoder": True,
                "seam_config_path": str(self.config_path),
                "seam_checkpoint_path": str(self.checkpoint_path),
                "max_source_positions": int(self.config["data"]["max_source_length"]),
                "eos_token_id": self.tokenizer.eos_token_id,
                # The vLLM renderer resolves this even when our processor
                # skips prepending it. Use an existing trained prompt token;
                # do not add BOS to Qwen3 or change the actual decoder IDs.
                "decoder_start_token_id": self.prompt_token_ids[0],
                "seam_encoder_max_batch_tokens": encoder_max_batch_tokens,
            },
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            # Request-specific copy states are intentionally Python-side in
            # vLLM 0.25.1; eager execution keeps their ordering explicit.
            "enforce_eager": True,
            # The adapter keeps copy state outside vLLM's KV cache. Until
            # request-state restoration on cache hits is verified, disable APC.
            "enable_prefix_caching": False,
            # Use only the generation settings translated from the resolved
            # AFMR config; do not inherit model-folder sampling defaults.
            "generation_config": "vllm",
        }
        defaults.update(vllm_options)
        self._actor_lock = RLock()
        self._actor_sync_failed = False
        self._actor_version = -1
        self._actor_sync_supported = (
            defaults.get("distributed_executor_backend") == "external_launcher"
            and os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
        )
        self.engine = LLM(**defaults)

    def sync_weights(self, model, *, policy_version: int, chunk_bytes: int = 128 * 1024 * 1024):
        """Synchronize a full native EviSeqAFMR actor in a colocated process.

        Direct GPU copies avoid checkpoint I/O and host staging. Only the
        external-launcher/in-process engine is supported here; arbitrary HTTP
        servers and decoder-only vLLM weight reloads do not implement this API.
        A failed transaction blocks generation until the engine is rebuilt.
        """
        from eviseq_afmr.training.checkpoint import architecture_spec

        from ._actor_rpc import begin_actor_update, finish_actor_update, load_actor_chunk

        if not self._actor_sync_supported:
            raise RuntimeError("Actor sync requires external_launcher and VLLM_ENABLE_V1_MULTIPROCESSING=0")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1:
            raise ValueError("chunk_bytes must be a positive integer")
        with self._actor_lock:
            if self._actor_sync_failed or policy_version <= self._actor_version:
                raise RuntimeError("Actor sync failed previously or policy version is stale")
            if self.engine.llm_engine.has_unfinished_requests():
                raise RuntimeError("Drain all inference requests before synchronizing the actor")
            if architecture_spec(model.config) != architecture_spec(self.config):
                raise ValueError("Actor architecture differs from the engine config")
            weights = list(model.named_parameters())
            manifest = [(name, list(value.shape)) for name, value in weights]
            self._actor_sync_failed = True
            if not self.engine.reset_prefix_cache():
                raise RuntimeError("vLLM refused to invalidate the previous actor KV cache")
            self.engine.reset_mm_cache()
            self.engine.collective_rpc(
                begin_actor_update, args=(manifest, architecture_spec(model.config), policy_version)
            )
            chunk, size = [], 0
            for name, value in weights:
                chunk.append((name, value.detach()))
                size += value.numel() * value.element_size()
                if size >= chunk_bytes:
                    self.engine.collective_rpc(load_actor_chunk, args=(chunk,))
                    chunk, size = [], 0
            if chunk:
                self.engine.collective_rpc(load_actor_chunk, args=(chunk,))
            result = self.engine.collective_rpc(finish_actor_update)
            if len(result) != 1 or result[0]["policy_version"] != policy_version:
                raise RuntimeError("SEAM worker did not confirm the requested policy version")
            self._actor_version = policy_version
            self._actor_sync_failed = False
            return result[0]

    def _get_sampling_params(self, sampling_params=None, **overrides):
        from vllm import SamplingParams

        if sampling_params is not None and overrides:
            raise ValueError("Pass SamplingParams or generation keyword options, not both")
        if sampling_params is None:
            sampling_params = SamplingParams(**sampling_kwargs(self.config["generation"], overrides))
        params = [sampling_params] if isinstance(sampling_params, SamplingParams) else list(sampling_params)
        if not params or not all(isinstance(param, SamplingParams) for param in params):
            raise TypeError("sampling_params must be SamplingParams or a sequence of SamplingParams")
        for param in params:
            validate_sampling_params(param)
        return sampling_params if isinstance(sampling_params, SamplingParams) else params

    @staticmethod
    def _normalize_sources(sources: str | Sequence[str]) -> tuple[list[str], bool]:
        single = isinstance(sources, str)
        normalized = [sources] if single else list(sources)
        if not normalized:
            raise ValueError("SEAMLLM expects at least one source string")
        for source in normalized:
            if not isinstance(source, str) or not source.strip():
                raise ValueError("Every SEAM source must be a non-empty string")
        return normalized, single

    def generate(
        self,
        sources: str | Sequence[str],
        sampling_params=None,
        *,
        use_tqdm: bool = False,
        max_input_tokens: int | None = None,
        encoder_prompt: str | None = None,
        decoder_prompt: str | None = None,
        decoder_prefix: str | None = None,
        decoder_chat_template: bool | None = None,
        **generation_options,
    ):
        """Return native vLLM RequestOutput objects; accept native or AFMR option names."""
        normalized, _ = self._normalize_sources(sources)
        sampling_params = self._get_sampling_params(sampling_params, **generation_options)
        from vllm import SamplingParams

        params = [sampling_params] * len(normalized) if isinstance(sampling_params, SamplingParams) else sampling_params
        if len(params) != len(normalized):
            raise ValueError("sampling_params must contain one entry per source")
        source_limit = int(self.config["data"]["max_source_length"])
        if max_input_tokens is not None and not 0 < max_input_tokens <= source_limit:
            raise ValueError(
                f"max_input_tokens must be in [1, {source_limit}]; increase the config limit before loading"
            )
        prompt_ids = self.prompt_token_ids
        if any(value is not None for value in (decoder_prompt, decoder_prefix, decoder_chat_template)):
            prompt_ids = request_prompt_ids(
                self.tokenizer,
                self.config["data"],
                decoder_prompt=decoder_prompt,
                decoder_prefix=decoder_prefix,
                decoder_chat_template=decoder_chat_template,
            )
        prompts = []
        for source, param in zip(normalized, params, strict=True):
            prompts.append(
                source_prompt(
                    source,
                    prompt_ids,
                    source_limit=max_input_tokens or source_limit,
                    output_budget=param.max_tokens,
                    encoder_prompt=encoder_prompt,
                )
            )
        with getattr(self, "_actor_lock", nullcontext()):
            if getattr(self, "_actor_sync_failed", False):
                raise RuntimeError("SEAM actor update failed; generation is blocked until engine rebuild")
            return self.engine.generate(prompts, sampling_params, use_tqdm=use_tqdm)

    def summarize(self, sources: str | Sequence[str], sampling_params=None, **options) -> str | list[str]:
        normalized, single = self._normalize_sources(sources)
        # Text convenience calls return exactly one completion per source.
        request_keys = {
            "use_tqdm",
            "max_input_tokens",
            "encoder_prompt",
            "decoder_prompt",
            "decoder_prefix",
            "decoder_chat_template",
        }
        generation_options = {key: value for key, value in options.items() if key not in request_keys}
        params = self._get_sampling_params(sampling_params, **generation_options)
        from vllm import SamplingParams

        all_params = [params] if isinstance(params, SamplingParams) else params
        if any(param.n != 1 or not param.detokenize for param in all_params):
            raise ValueError("summarize() requires n=1 and detokenize=True; use generate() for candidates/token IDs")
        outputs = self.generate(
            normalized,
            params,
            **{key: value for key, value in options.items() if key in request_keys},
        )
        summaries = [output.outputs[0].text for output in outputs]
        return summaries[0] if single and len(normalized) == 1 else summaries

    def __call__(self, sources: str | Sequence[str], sampling_params=None, **options) -> str | list[str]:
        """Summarize one source or a batch, like a text-generation pipeline."""
        return self.summarize(sources, sampling_params, **options)
