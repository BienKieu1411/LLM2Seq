from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutput

from ..cache import FullHistoryCache
from ..config import resolve_path, torch_dtype
from ..state import SourceState


class T5Gemma2Adapter:
    def __init__(self, model, tokenizer, config: Dict[str, Any], checkpoint: Path):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.checkpoint = checkpoint
        self.target_identity = str(config["target"].get("identity") or "")
        self._manifest_cache: Optional[Dict[str, Any]] = None
        self.device = torch.device(config["target"].get("device", "cuda"))
        self.dtype = torch_dtype(config["target"].get("dtype", "bfloat16"))
        model_type = str(getattr(model.config, "model_type", ""))
        if model_type != "t5gemma2" and "t5gemma2" not in model.__class__.__name__.lower():
            raise TypeError(f"Expected a T5Gemma2 checkpoint, got model_type={model_type!r}")
        if not hasattr(model, "model") or not hasattr(model.model, "decoder"):
            raise TypeError("The loaded model does not expose the T5Gemma2 encoder-decoder core")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.input_embedding = self.model.get_input_embeddings()
        self.output_head = self.model.get_output_embeddings()
        if self.input_embedding is None or self.output_head is None:
            raise RuntimeError("T5Gemma2 must expose its shared token embedding and output head")
        self.start_token_id = self._resolve_start_token()
        eos_value = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_value is None:
            eos_value = getattr(self.model.config, "eos_token_id", None)
        self.eos_token_ids = self._as_id_set(eos_value)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.start_token_id
        self._validate_generation_config()

    @classmethod
    def load(cls, config: Dict[str, Any]) -> "T5Gemma2Adapter":
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        target_cfg = config["target"]
        checkpoint = resolve_path(target_cfg["checkpoint"], config)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Local T5Gemma2 checkpoint does not exist: {checkpoint}. Downloads are disabled; "
                "set target.checkpoint to a model directory already present on this machine."
            )
        tokenizer_path = target_cfg.get("tokenizer")
        if tokenizer_path:
            tokenizer_path = resolve_path(tokenizer_path, config)
        else:
            tokenizer_path = checkpoint
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Local T5Gemma2 tokenizer path does not exist: {tokenizer_path}")
        common = {
            "local_files_only": True,
            "trust_remote_code": bool(target_cfg.get("trust_remote_code", False)),
        }
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), **common)
        model_kwargs = dict(common)
        model_kwargs["torch_dtype"] = torch_dtype(target_cfg.get("dtype", "bfloat16"))
        attention_implementation = target_cfg.get("attn_implementation")
        if attention_implementation:
            model_kwargs["attn_implementation"] = str(attention_implementation)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(checkpoint), **model_kwargs)
        device = torch.device(target_cfg.get("device", "cuda"))
        model.to(device)
        return cls(model=model, tokenizer=tokenizer, config=config, checkpoint=checkpoint)

    @staticmethod
    def _as_id_set(value: Any) -> set[int]:
        if value is None:
            return set()
        if isinstance(value, (tuple, list, set)):
            return {int(token_id) for token_id in value if token_id is not None}
        return {int(value)}

    def _resolve_start_token(self) -> int:
        candidates = (
            getattr(self.model.generation_config, "decoder_start_token_id", None),
            getattr(getattr(self.model.config, "decoder", None), "decoder_start_token_id", None),
            getattr(self.model.generation_config, "bos_token_id", None),
            self.tokenizer.bos_token_id,
            self.tokenizer.pad_token_id,
        )
        for token_id in candidates:
            if token_id is not None:
                return int(token_id)
        raise RuntimeError("Could not determine the T5Gemma2 decoder start token ID")

    def _validate_generation_config(self) -> None:
        generation_config = self.model.generation_config
        unsupported = (
            "forced_bos_token_id",
            "forced_eos_token_id",
            "forced_decoder_ids",
            "suppress_tokens",
            "begin_suppress_tokens",
            "bad_words_ids",
            "force_words_ids",
            "sequence_bias",
            "constraints",
        )
        for name in unsupported:
            value = getattr(generation_config, name, None)
            if value not in (None, [], {}):
                raise ValueError(
                    f"The greedy adapter does not implement generation_config.{name}; "
                    "remove that constraint before using this runtime"
                )
        if int(getattr(generation_config, "encoder_no_repeat_ngram_size", 0) or 0) > 0:
            raise ValueError("encoder_no_repeat_ngram_size is not implemented by the greedy adapter")
        if int(getattr(generation_config, "min_length", 0) or 0) > 0:
            raise ValueError("generation_config.min_length is unsupported; use generation.min_new_tokens")

    @property
    def vocab_size(self) -> int:
        return int(self.output_head.weight.shape[0])

    @property
    def decoder_width(self) -> int:
        width = getattr(getattr(self.model.config, "decoder", None), "hidden_size", None)
        if width is None:
            width = self.input_embedding.weight.shape[1]
        return int(width)

    @property
    def encoder_width(self) -> int:
        config = getattr(self.model.config, "encoder", None)
        config = getattr(config, "text_config", config)
        width = getattr(config, "hidden_size", None)
        if width is None:
            width = getattr(self.model.config, "d_model", None)
        if width is None:
            width = getattr(self.model.config, "hidden_size", None)
        if width is None:
            raise RuntimeError("Could not infer the T5Gemma2 encoder width from its local config")
        return int(width)

    @property
    def attention_implementation(self) -> Optional[str]:
        return getattr(self.model.config, "_attn_implementation", None)

    def tokenize_source(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        data_cfg = self.config["data"]
        full_text = str(data_cfg.get("source_prefix", "")) + text
        encoded = self.tokenizer(
            full_text,
            max_length=int(data_cfg.get("max_source_tokens", 8192)),
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def encode_ids(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> SourceState:
        input_ids = input_ids.to(self.device, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.device, dtype=torch.long)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
        if input_ids.shape[0] != 1:
            raise ValueError("The first runtime supports one request at a time")
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        digest = hashlib.sha256()
        digest.update(str(self.checkpoint).encode("utf-8"))
        digest.update(input_ids.detach().cpu().contiguous().numpy().tobytes())
        digest.update(attention_mask.detach().cpu().contiguous().numpy().tobytes())
        return SourceState(
            source_id=digest.hexdigest()[:16],
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            fingerprint=digest.hexdigest(),
            metadata={
                "source_tokens": int(attention_mask[0].sum().item()),
                "checkpoint": str(self.checkpoint),
            },
        )

    def encode_text(self, text: str, source_id: str = "") -> SourceState:
        input_ids, attention_mask = self.tokenize_source(text)
        source = self.encode_ids(input_ids, attention_mask)
        if source_id:
            source.source_id = str(source_id)
        return source

    def new_cache(self) -> FullHistoryCache:
        return FullHistoryCache.create()

    def forward_decoder(
        self,
        source: SourceState,
        decoder_input_ids: Sequence[int] | torch.Tensor,
        cache: FullHistoryCache,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(decoder_input_ids, torch.Tensor):
            input_ids = decoder_input_ids.to(self.device, dtype=torch.long)
        else:
            input_ids = torch.tensor([list(decoder_input_ids)], dtype=torch.long, device=self.device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("Decoder input must contain at least one token for a single request")
        total_length = cache.length + input_ids.shape[1]
        decoder_attention_mask = torch.ones((1, total_length), dtype=torch.long, device=self.device)
        output = self.model.model(
            encoder_outputs=source.encoder_outputs,
            attention_mask=source.attention_mask,
            decoder_input_ids=input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=cache.cache,
            use_cache=True,
            return_dict=True,
        )
        cache.cache = output.past_key_values
        hidden_states = output.last_hidden_state
        logits = self.output_head(hidden_states)
        softcap = getattr(getattr(self.model.config, "decoder", None), "final_logit_softcapping", None)
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits, hidden_states

    def decoder_features(self, source: SourceState, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        decoder_input_ids = decoder_input_ids.to(self.device, dtype=torch.long)
        if decoder_input_ids.ndim == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        output = self.model.model(
            encoder_outputs=source.encoder_outputs,
            attention_mask=source.attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return output.last_hidden_state

    def native_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        generation = self.config["generation"]
        effective_max_new_tokens = int(max_new_tokens if max_new_tokens is not None else generation["max_new_tokens"])
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": effective_max_new_tokens,
            "min_new_tokens": min(int(generation.get("min_new_tokens", 0)), effective_max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
            "repetition_penalty": float(generation.get("repetition_penalty", 1.0)),
            "no_repeat_ngram_size": int(generation.get("no_repeat_ngram_size", 0)),
            "use_cache": True,
            "return_dict_in_generate": False,
            "decoder_start_token_id": self.start_token_id,
            "pad_token_id": self.pad_token_id,
        }
        if self.eos_token_ids:
            kwargs["eos_token_id"] = sorted(self.eos_token_ids)
        with torch.inference_mode():
            result = self.model.generate(**kwargs)
        return result[0]

    def target_manifest(self) -> Dict[str, Any]:
        if self._manifest_cache is not None:
            return dict(self._manifest_cache)
        config_json = self.model.config.to_dict()
        config_json = self._strip_runtime_config(config_json)
        config_sha256 = hashlib.sha256(json.dumps(config_json, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        tokenizer_manifest = {
            "class": self.tokenizer.__class__.__qualname__,
            "vocab_size": len(self.tokenizer),
            "special_tokens_map": getattr(self.tokenizer, "special_tokens_map", {}),
            "vocab_sha256": self._tokenizer_vocab_fingerprint(),
        }
        tokenizer_sha256 = hashlib.sha256(
            json.dumps(tokenizer_manifest, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        checkpoint_metadata_sha256 = self._checkpoint_metadata_fingerprint(self.checkpoint)
        target_fingerprint = (
            self.target_identity
            or hashlib.sha256(
                f"{checkpoint_metadata_sha256}:{tokenizer_sha256}:{config_sha256}".encode("utf-8")
            ).hexdigest()
        )
        manifest = {
            "checkpoint": str(self.checkpoint),
            "target_identity": self.target_identity or None,
            "target_fingerprint": target_fingerprint,
            "fingerprint_method": "explicit target.identity"
            if self.target_identity
            else "local checkpoint file metadata + tokenizer metadata + model config",
            "model_type": getattr(self.model.config, "model_type", None),
            "transformers_version": self._package_version("transformers"),
            "torch_version": torch.__version__,
            "dtype": str(next(self.model.parameters()).dtype),
            "device": str(next(self.model.parameters()).device),
            "attention_implementation": self.attention_implementation,
            "encoder_width": self.encoder_width,
            "decoder_width": self.decoder_width,
            "vocab_size": self.vocab_size,
            "config_sha256": config_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "checkpoint_metadata_sha256": checkpoint_metadata_sha256,
        }
        self._manifest_cache = manifest
        return dict(manifest)

    @staticmethod
    def _strip_runtime_config(value: Any) -> Any:
        ignored = {"_name_or_path", "_attn_implementation", "_attn_implementation_internal"}
        if isinstance(value, dict):
            return {
                key: T5Gemma2Adapter._strip_runtime_config(child) for key, child in value.items() if key not in ignored
            }
        if isinstance(value, list):
            return [T5Gemma2Adapter._strip_runtime_config(child) for child in value]
        return value

    def _tokenizer_vocab_fingerprint(self) -> str:
        vocab = self.tokenizer.get_vocab()
        digest = hashlib.sha256()
        for token, token_id in sorted(vocab.items()):
            digest.update(token.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(token_id).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    @staticmethod
    def _checkpoint_metadata_fingerprint(checkpoint: Path) -> str:
        relevant_suffixes = {".safetensors", ".bin", ".json"}
        if checkpoint.is_file():
            files = [checkpoint]
            base = checkpoint.parent
        else:
            base = checkpoint
            files = [
                path for path in checkpoint.rglob("*") if path.is_file() and path.suffix.lower() in relevant_suffixes
            ]
        digest = hashlib.sha256()
        for path in sorted(files):
            try:
                stat = path.stat()
            except OSError:
                continue
            relative = path.relative_to(base).as_posix()
            digest.update(f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def _package_version(name: str) -> Optional[str]:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
