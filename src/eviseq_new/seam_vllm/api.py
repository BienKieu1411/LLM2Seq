"""Small Python facade over vLLM for source-to-summary SEAM requests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from eviseq_afmr.config import load_config, resolve_path

from ._compat import validate_engine_options
from ._prompt import configured_decoder_prompt_ids


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
        **vllm_options: Any,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.config_path.is_file():
            raise FileNotFoundError(f"SEAM config not found: {self.config_path}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SEAM checkpoint not found: {self.checkpoint_path}")
        self.config = load_config(self.config_path)
        model = self.config["model"]
        self.decoder_path = resolve_path(model["decoder_name"], self.config)
        self.encoder_path = resolve_path(model["encoder_name"], self.config)
        for name, path in (("decoder", self.decoder_path), ("encoder", self.encoder_path)):
            if not path.exists():
                raise FileNotFoundError(f"SEAM {name} must be present locally: {path}")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoTokenizer
        from vllm import LLM

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
        config = self.config
        generation = config["generation"]
        if int(generation.get("num_beams", 1)) != 1:
            raise ValueError("SEAM vLLM currently supports greedy/sampling decode, not beam search")
        if int(generation.get("no_repeat_ngram_size", 0)) > 0:
            raise NotImplementedError(
                "vLLM 0.25.1 does not implement AFMR no_repeat_ngram_size; "
                "set generation.no_repeat_ngram_size=0 explicitly or use the AFMR evaluator"
            )
        if (
            bool(self.config["data"].get("decoder_chat_template", False))
            and float(generation.get("repetition_penalty", 1.0)) != 1.0
        ):
            raise NotImplementedError(
                "SEAM vLLM cannot exclude decoder chat-template tokens from repetition_penalty; "
                "set generation.repetition_penalty=1.0 or disable decoder_chat_template"
            )

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
        self.engine = LLM(**defaults)
        self._sampling_params = None

    def _get_sampling_params(self):
        if self._sampling_params is not None:
            return self._sampling_params
        from vllm import SamplingParams

        generation = self.config["generation"]
        temperature = float(generation.get("temperature", 0.0))
        params = {
            "max_tokens": int(generation.get("max_new_tokens", 256)),
            "min_tokens": int(generation.get("min_new_tokens", 0)),
            "temperature": temperature,
            "top_p": float(generation.get("top_p", 1.0)),
            "top_k": max(0, int(generation.get("top_k", 0))),
            "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
        }
        if temperature <= 0.0 or not bool(generation.get("do_sample", False)):
            params["temperature"] = 0.0
            params["top_p"] = 1.0
            params["top_k"] = 0
        self._sampling_params = SamplingParams(**params)
        return self._sampling_params

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

    def generate(self, sources: str | Sequence[str], *, use_tqdm: bool = False):
        normalized, _ = self._normalize_sources(sources)
        prompts = []
        for source in normalized:
            prompts.append(
                {
                    "prompt_token_ids": self.prompt_token_ids,
                    "multi_modal_data": {"text": source},
                }
            )
        return self.engine.generate(prompts, self._get_sampling_params(), use_tqdm=use_tqdm)

    def summarize(self, sources: str | Sequence[str], *, use_tqdm: bool = False) -> str | list[str]:
        normalized, single = self._normalize_sources(sources)
        outputs = self.generate(sources, use_tqdm=use_tqdm)
        summaries = [output.outputs[0].text for output in outputs]
        return summaries[0] if single and len(normalized) == 1 else summaries

    def __call__(self, sources: str | Sequence[str], *, use_tqdm: bool = False) -> str | list[str]:
        """Summarize one source or a batch, like a text-generation pipeline."""
        return self.summarize(sources, use_tqdm=use_tqdm)
